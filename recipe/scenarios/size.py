"""Size / few-shot scenario: keep k real patients of the transfer class and fill
the class back to full size with artificial patients drawn from those k.

For each k in ``size.k_stats_values`` and each fold (a seeded re-selection of the
k kept patients), a GMM is fit on the pool (diseased + the k kept patients), the
signature is taken from the k patients (exact counts when k=1), and
(total - k) artificial patients are generated. Two arms per (k, fold): baseline
(diseased + k real) and augmented (+ artificials).

Generation and protocol selection are coupled (the artificials must match the k
kept patients), so this scenario's ``generate`` builds both the bags and the
arms; ``protocols``/``run`` are aliases of it.
"""
from __future__ import annotations

import json
import os
import random
from typing import List

import numpy as np
import pandas as pd

from .. import dataset_spec as ds
from ..clustering import build_codebook
from ..config import ExperimentConfig
from ..core import protocols as proto
from ..core.generator import GenerationInputs, build_cell_pool, generate_gaussian_patients
from ..core.signature import compute_signature
from ..core.train_runner import Arm
from ..seeding import seed_everything


def _artificial_dir(cfg: ExperimentConfig, k_stats: int, fold: int) -> str:
    return os.path.join(cfg.artificial_patients_dir, f"kstats_{k_stats}", f"fold_{fold}")


def _selection_csv(cfg: ExperimentConfig) -> str:
    return os.path.join(cfg.output_root, "size_selection.json")


def generate(cfg: ExperimentConfig) -> dict:
    if cfg.transfer_label is None:
        raise ValueError("size scenario requires `transfer_label` (the few-shot class)")
    seed_everything(cfg.generator.seed)
    os.makedirs(cfg.output_root, exist_ok=True)
    base_fold = cfg.protocol.injection_base_fold
    label_to_id = ds.prepare_label_map(cfg.data, cfg.transfer_label)
    tid = label_to_id[cfg.transfer_label]
    col = cfg.data.out_patient_col

    records = ds.fold_train_records(cfg.data, base_fold, label_to_id)
    transfer = [r for r in records if r["label_name"] == cfg.transfer_label]
    diseased = [r for r in records if r["label_name"] != cfg.transfer_label]
    total_transfer = len(transfer)
    print(f"[recipe] base fold {base_fold}: {len(diseased)} diseased, {total_transfer} '{cfg.transfer_label}'")

    diseased_rows = pd.DataFrame({col: [r["feature_path"] for r in diseased],
                                  "labels": [r["label"] for r in diseased]})
    pid_to_rec = {r["patient_id"]: r for r in transfer}

    selection = {}
    for k_stats in cfg.size.k_stats_values:
        for fold in range(cfg.protocol.n_folds):
            random.seed(42 + fold)
            kept = random.sample(sorted(pid_to_rec), min(k_stats, total_transfer))
            kept_recs = [pid_to_rec[p] for p in kept]

            # pool = diseased + kept; fit GMM on the pool's cells.
            pool_recs = diseased + kept_recs
            X = np.concatenate([r["features"] for r in pool_recs], axis=0)
            codebook = build_codebook(cfg.clustering).fit(X)
            cell_pool = build_cell_pool(
                [{"patient_id": r["patient_id"], "label": r["label"], "features": r["features"]}
                 for r in pool_recs], codebook)

            kept_set = set(kept)
            sig = compute_signature(
                cell_pool[cell_pool["patient_id"].isin(kept_set)], cfg.clustering.n_clusters)

            out_dir = _artificial_dir(cfg, k_stats, fold)
            n_art = max(0, total_transfer - len(kept))
            gi = GenerationInputs(
                stats_mean=sig.mean, stats_std=sig.std, n_clusters=cfg.clustering.n_clusters,
                cell_pool=cell_pool, n_patients=n_art, target_label=tid, output_dir=out_dir,
                label_name=str(cfg.transfer_label), seed=cfg.generator.seed,
                exact_counts=sig.exact, max_bag_size=cfg.generator.max_bag_size)
            art_records = generate_gaussian_patients(gi)

            kept_rows = pd.DataFrame({col: [r["feature_path"] for r in kept_recs],
                                      "labels": [tid] * len(kept_recs)})
            real_rows = pd.concat([diseased_rows, kept_rows], ignore_index=True)
            proto.build_size_protocols(cfg, label_to_id, fold, k_stats,
                                       real_rows, art_records, out_dir)
            selection[f"{k_stats}_{fold}"] = {"k_stats": k_stats, "fold": fold,
                                              "n_artificial": len(art_records)}
            print(f"[recipe] k={k_stats} fold={fold}: kept {len(kept)}, generated {len(art_records)}")
    with open(_selection_csv(cfg), "w") as fh:
        json.dump(selection, fh, indent=2)
    return selection


# generation and protocol building are inseparable here.
protocols = generate
run = generate


def arms(cfg: ExperimentConfig) -> List[Arm]:
    out = []
    for k_stats in cfg.size.k_stats_values:
        name = f"kstats_{k_stats}"
        out.append(Arm(f"k{k_stats}_baseline", name, "baseline"))
        out.append(Arm(f"k{k_stats}_augmented", name, "augmented"))
    return out
