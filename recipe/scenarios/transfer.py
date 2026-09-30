"""Transfer scenario: supply a target's missing class by transferring a source
cohort's per-cluster composition signature onto the target's donor cells.

Stages: fit-codebook -> generate -> protocols -> train -> evaluate.
"""
from __future__ import annotations

import os
from typing import List

import pandas as pd

from .. import dataset_spec as ds
from ..clustering import build_codebook
from ..codebook import load_codebook, save_codebook
from ..config import ExperimentConfig
from ..core import protocols as proto
from ..core.generator import GenerationInputs, build_cell_pool, generate_gaussian_patients
from ..core.signature import compute_signature
from ..core.train_runner import Arm
from ..seeding import seed_everything


def _artificial_dir(cfg: ExperimentConfig) -> str:
    return os.path.join(cfg.artificial_patients_dir, "transfer")


def _records_csv(cfg: ExperimentConfig) -> str:
    return os.path.join(cfg.output_root, "artificial_patient_labels.csv")


def fit_codebook(cfg: ExperimentConfig) -> dict:
    """Fit the codebook on the source cohort and derive the transfer-class signature."""
    if cfg.source is None:
        raise ValueError("transfer scenario requires a `source:` data block in the config")
    seed_everything(cfg.generator.seed)
    os.makedirs(cfg.output_root, exist_ok=True)
    gcfg = cfg.clustering

    X = ds.source_fit_matrix(cfg.source, max_samples=gcfg.fit_sample_size, seed=gcfg.random_state)
    print(f"[recipe] fitting codebook '{gcfg.algorithm}' (k={gcfg.n_clusters}) on {X.shape}")
    codebook = build_codebook(gcfg).fit(X)
    del X
    save_codebook(codebook, cfg.codebook_path)

    records = ds.source_transfer_records(cfg.source, cfg.transfer_label)
    print(f"[recipe] source '{cfg.transfer_label}' patients: {len(records)}")
    df = build_cell_pool(records, codebook)
    sig = compute_signature(df, gcfg.n_clusters)
    pd.DataFrame({"cluster": range(gcfg.n_clusters), "mean": sig.mean, "std": sig.std}
                 ).to_csv(cfg.signature_stats_csv, index=False)
    sig.pivot.to_csv(cfg.signature_counts_csv, index_label="patient_id")
    print(f"[recipe] signature saved -> {cfg.signature_stats_csv}")
    return {"n_transfer_patients": len(sig.pivot)}


def generate(cfg: ExperimentConfig) -> List[dict]:
    seed_everything(cfg.generator.seed)
    label_to_id = ds.prepare_label_map(cfg.data, cfg.transfer_label)
    tid = label_to_id[cfg.transfer_label]
    codebook = load_codebook(cfg.codebook_path)
    donor = ds.target_donor_records(
        cfg.data, cfg.protocol.injection_base_fold, cfg.transfer_label, label_to_id,
        max_cells_per_patient=cfg.clustering.donor_cells_per_patient,
        subsample_seed=cfg.clustering.random_state)
    cell_pool = build_cell_pool(donor, codebook)
    stats = pd.read_csv(cfg.signature_stats_csv)

    out_dir = _artificial_dir(cfg)
    gi = GenerationInputs(
        stats_mean=stats["mean"].values, stats_std=stats["std"].values,
        n_clusters=cfg.clustering.n_clusters, cell_pool=cell_pool,
        n_patients=cfg.generator.n_artificial, target_label=tid, output_dir=out_dir,
        label_name=str(cfg.transfer_label), seed=cfg.generator.seed,
        max_bag_size=cfg.generator.max_bag_size)
    records = generate_gaussian_patients(gi)
    pd.DataFrame(records).to_csv(_records_csv(cfg), index=False)
    print(f"[recipe] generated {len(records)} artificial '{cfg.transfer_label}' patients -> {out_dir}")
    return records


def protocols(cfg: ExperimentConfig) -> dict:
    label_to_id = ds.prepare_label_map(cfg.data, cfg.transfer_label)
    records = pd.read_csv(_records_csv(cfg)).to_dict("records")
    out = proto.build_transfer_protocols(cfg, label_to_id, records, _artificial_dir(cfg))
    for arm, paths in out.items():
        print(f"[recipe] protocols[{arm}]: {len(paths)} folds")
    return out


def run(cfg: ExperimentConfig) -> dict:
    fit_codebook(cfg)
    generate(cfg)
    return protocols(cfg)


def arms(cfg: ExperimentConfig) -> List[Arm]:
    a = [Arm("full_dataset", "transfer", "full_dataset"),
         Arm("baseline", "transfer", "baseline"),
         Arm("synthetic", "transfer", "synthetic")]
    if cfg.source is not None:
        a.insert(2, Arm("direct_transfer", "transfer", "direct_transfer"))
    return a
