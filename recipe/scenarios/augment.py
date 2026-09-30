"""Augmentation scenario: generate artificial patients for every class, train a
dropout selector, score artificials by MC-dropout uncertainty, and keep the
top-X% (per metric x strategy x percentage) added to the real training set.

Pipeline (all per fold, the selector/scoring/final-train per architecture):
    generate  -> per-class GMM signature + Gaussian sampling
    selector  -> train baseline model with dropout enabled (train_5fold)
    score     -> MC-dropout uncertainty for every artificial patient
    protocols -> select top-X% and combine with real train
    train     -> final models on each combined protocol
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from .. import dataset_spec as ds
from ..clustering import build_codebook
from ..config import ExperimentConfig
from ..core import protocols as proto
from ..core import selection as sel
from ..core.generator import GenerationInputs, build_cell_pool, generate_gaussian_patients
from ..core.signature import compute_signature
from ..core.train_runner import Arm, TrainJob, build_command, run_job
from ..data_io import load_features
from ..seeding import seed_everything

# Unlike transfer/size -- whose folds are repeated draws over one held-out
# split -- the augment scenario is real 5-fold cross-validation, so val/test
# must move with the training fold.
CV_ARGS = ("--eval_fold", "per_fold")

SELECTOR_MODE = "selector"   # the MC-dropout scoring model
BASELINE_MODE = "baseline"   # the no-augmentation comparison arm


def _artificial_dir(cfg: ExperimentConfig, fold: int) -> str:
    return os.path.join(cfg.artificial_patients_dir, f"fold_{fold}")


def _records_csv(cfg: ExperimentConfig, fold: int) -> str:
    return os.path.join(cfg.output_root, "artificial_records", f"fold_{fold}.csv")


def _uncertainty_csv(cfg: ExperimentConfig, arch: str, fold: int) -> str:
    return os.path.join(cfg.output_root, "uncertainty", arch, f"fold_{fold}.csv")


def _dataset_name(arch: str) -> str:
    return f"augment_{arch}"


def _final_mode(cfg: ExperimentConfig, metric: str, strategy: str, pct: int) -> str:
    return f"fitep{cfg.augment.fit_ep}_{metric}_{sel.STRATEGY_TOKEN[strategy]}_top{pct}pct"


def _selector_result_dir(cfg: ExperimentConfig, arch: str) -> str:
    """Reconstruct train_5fold's result folder for the selector run."""
    name = (f"{arch}_k{cfg.protocol.k_value}_{SELECTOR_MODE}"
            f"_lr{cfg.train.lr}_ep{cfg.augment.fit_ep}_seed{cfg.train.seed}")
    # must mirror train_5fold's layout: <results_root>/<arch>/<dataset_name>/<name>
    return os.path.join(cfg.output_root, arch, "augment", name)


# ---------------------------------------------------------------------------
# Stage 1: generation (per fold, per class) -- architecture-independent
# ---------------------------------------------------------------------------
def generate(cfg: ExperimentConfig) -> Dict[int, str]:
    seed_everything(cfg.generator.seed)
    label_to_id = ds.prepare_label_map(cfg.data)
    out = {}
    for fold in cfg.train.folds:
        # Each fold starts from a freshly seeded global RNG, so generation for
        # fold k does not depend on the folds generated before it.
        seed_everything(cfg.generator.seed)
        records = ds.fold_train_records(cfg.data, fold, label_to_id, eval_fold=fold)
        X = np.concatenate([r["features"] for r in records], axis=0)
        codebook = build_codebook(cfg.clustering).fit(X)
        cell_pool = build_cell_pool(
            [{"patient_id": r["patient_id"], "label": r["label"], "features": r["features"]}
             for r in records], codebook)

        out_dir = _artificial_dir(cfg, fold)
        all_recs: List[dict] = []
        for name, cid in label_to_id.items():
            class_pool = cell_pool[cell_pool["label"] == cid]
            if class_pool.empty:
                continue
            sig = compute_signature(class_pool, cfg.clustering.n_clusters)
            gi = GenerationInputs(
                stats_mean=sig.mean, stats_std=sig.std, n_clusters=cfg.clustering.n_clusters,
                cell_pool=class_pool, n_patients=cfg.augment.artificial_patient_count,
                target_label=cid, output_dir=out_dir, label_name=str(name),
                seed=cfg.generator.seed, max_bag_size=cfg.generator.max_bag_size)
            all_recs.extend(generate_gaussian_patients(gi))
        os.makedirs(os.path.dirname(_records_csv(cfg, fold)), exist_ok=True)
        pd.DataFrame(all_recs).to_csv(_records_csv(cfg, fold), index=False)
        out[fold] = _records_csv(cfg, fold)
        print(f"[recipe] fold {fold}: generated {len(all_recs)} artificial patients -> {out_dir}")
    # baseline protocol for the selector (real train, with dropout at train time).
    for fold in cfg.train.folds:
        # identical rows, two modes: the selector trains with dropout for
        # `fit_ep` epochs, the baseline arm trains like every other arm.
        proto.build_baseline_only(cfg, label_to_id, fold, dataset_name="augment", mode=SELECTOR_MODE)
        for arch in cfg.train.architectures:
            proto.build_baseline_only(cfg, label_to_id, fold,
                                      dataset_name=_dataset_name(arch), mode=BASELINE_MODE)
    return out


# ---------------------------------------------------------------------------
# Stage 2: selector training (per arch) -- baseline model with dropout
# ---------------------------------------------------------------------------
def selector_jobs(cfg: ExperimentConfig) -> List[TrainJob]:
    arm = Arm("selector", "augment", SELECTOR_MODE,
              extra_args=["--dropout_p", str(cfg.augment.dropout_p),
                          "--ep", str(cfg.augment.fit_ep),
                          "--eval_fold", "per_fold"])
    return [build_command(cfg, arch, arm) for arch in cfg.train.architectures]


def train_selector(cfg: ExperimentConfig, dry_run: bool = False) -> int:
    rc = 0
    for job in selector_jobs(cfg):
        rc |= run_job(job, cfg.train.mil_dir, dry_run=dry_run)
    return rc


# ---------------------------------------------------------------------------
# Stage 3: MC-dropout scoring (per arch, per fold)
# ---------------------------------------------------------------------------
def score(cfg: ExperimentConfig) -> None:
    sys.path.insert(0, cfg.train.mil_dir)  # make `classifier` importable
    from ..core.mc_dropout import load_model, mc_dropout_scores

    label_to_id = ds.prepare_label_map(cfg.data)
    class_count = len(label_to_id)
    for arch in cfg.train.architectures:
        for fold in cfg.train.folds:
            recs = pd.read_csv(_records_csv(cfg, fold))
            art_dir = _artificial_dir(cfg, fold)
            ext = cfg.data.file_extension
            patients = [(os.path.join(art_dir, f"{r.patient_id}.{ext}"), int(r.label))
                        for r in recs.itertuples()]
            emb_dim = load_features(patients[0][0]).shape[-1]
            ckpt = os.path.join(_selector_result_dir(cfg, arch),
                                f"fold_{fold}", "cAItomorph_best_weights.pth")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Selector checkpoint missing: {ckpt} (train selector first)")
            model = load_model(ckpt, arch, class_count, emb_dim, dropout_p=cfg.augment.dropout_p)
            df = mc_dropout_scores(model, patients, cfg.augment.mc_dropout)
            out = _uncertainty_csv(cfg, arch, fold)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            df.to_csv(out, index=False)
            print(f"[recipe] scored {len(df)} artificials -> {out}")


# ---------------------------------------------------------------------------
# Stage 4: protocol building (per arch, per fold, per metric/strategy/pct)
# ---------------------------------------------------------------------------
def build_protocols(cfg: ExperimentConfig) -> None:
    label_to_id = ds.prepare_label_map(cfg.data)
    highest = cfg.augment.uncertainty_mode == "highest"
    for arch in cfg.train.architectures:
        for fold in cfg.train.folds:
            unc = pd.read_csv(_uncertainty_csv(cfg, arch, fold))
            real_train = ds.fold_train_table(cfg.data, fold, label_to_id)
            real_n = len(real_train)
            real_class_counts = real_train["labels"].value_counts().to_dict()
            art_dir = _artificial_dir(cfg, fold)
            for metric in cfg.augment.metrics:
                for strategy in cfg.augment.strategies:
                    for pct in cfg.augment.percentages:
                        chosen = sel.select(unc, metric=metric, strategy=strategy,
                                            top_percentage=pct, highest=highest,
                                            real_patient_count=real_n,
                                            real_class_counts=real_class_counts)
                        proto.build_augment_protocol(
                            cfg, fold, _final_mode(cfg, metric, strategy, pct),
                            real_train, chosen, art_dir, dataset_name=_dataset_name(arch))
        print(f"[recipe] built combined protocols for arch={arch}")


def run(cfg: ExperimentConfig, dry_run: bool = False) -> None:
    generate(cfg)
    train_selector(cfg, dry_run=dry_run)
    if not dry_run:
        score(cfg)
        build_protocols(cfg)


# ---------------------------------------------------------------------------
# Final-training jobs (arch-coupled: each arch trains on its own protocols)
# ---------------------------------------------------------------------------
def build_jobs(cfg: ExperimentConfig) -> List[TrainJob]:
    jobs = []
    for arch in cfg.train.architectures:
        # the no-augmentation reference: real training set only, trained exactly
        # like the augmented arms -- the arm every augmented arm is compared against.
        jobs.append(build_command(cfg, arch,
                                  Arm("baseline", _dataset_name(arch), BASELINE_MODE,
                                      extra_args=list(CV_ARGS))))
        for metric in cfg.augment.metrics:
            for strategy in cfg.augment.strategies:
                for pct in cfg.augment.percentages:
                    mode = _final_mode(cfg, metric, strategy, pct)
                    arm = Arm(f"{metric}_{strategy}_top{pct}", _dataset_name(arch), mode,
                              extra_args=list(CV_ARGS))
                    jobs.append(build_command(cfg, arch, arm))
    return jobs
