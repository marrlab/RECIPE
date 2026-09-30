"""Training-protocol CSV builders for all three scenarios.

Every arm of every scenario writes the same artifact -- a
``train_fold_{fold}_k_{k}_{mode}.csv`` under ``{protocols_root}/{dataset_name}/k_{k}/``
(absolute feature path + integer ``labels``), exactly what ``train_5fold.py``
reads. One ``_write_protocol`` helper does the I/O; the scenarios decide what rows
go in each arm.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import dataset_spec as ds
from ..config import ExperimentConfig


def _protocol_dir(protocols_root: str, dataset_name: str, k_value: int) -> str:
    out = os.path.join(protocols_root, dataset_name, f"k_{k_value}")
    os.makedirs(out, exist_ok=True)
    return out


def _write_protocol(cfg: ExperimentConfig, df: pd.DataFrame, dataset_name: str,
                    fold: int, mode: str) -> str:
    out_dir = _protocol_dir(cfg.protocols_root, dataset_name, cfg.protocol.k_value)
    path = os.path.join(out_dir, f"train_fold_{fold}_k_{cfg.protocol.k_value}_{mode}.csv")
    df.to_csv(path, index=False)
    return path


# ===========================================================================
# transfer scenario (full_dataset / baseline / direct_transfer / synthetic)
# ===========================================================================
def _base_train(cfg: ExperimentConfig, label_to_id: Dict[str, int]) -> pd.DataFrame:
    return ds.fold_train_table(cfg.data, cfg.protocol.injection_base_fold, label_to_id)


def _n_inject(cfg: ExperimentConfig, base_df: pd.DataFrame, tid: int) -> int:
    if cfg.protocol.transfer_inject_count is not None:
        return int(cfg.protocol.transfer_inject_count)
    n = int((base_df["labels"] == tid).sum())
    if n == 0:
        raise ValueError(
            f"Target has no '{cfg.transfer_label}' patients to match; set "
            f"protocol.transfer_inject_count to choose how many to inject.")
    return n


def build_transfer_protocols(cfg: ExperimentConfig, label_to_id: Dict[str, int],
                             artificial_records: List[dict], artificial_dir: str,
                             dataset_name: str = "transfer") -> Dict[str, List[str]]:
    """All four arms for the transfer scenario."""
    pc = cfg.protocol
    col = cfg.data.out_patient_col
    tid = label_to_id[cfg.transfer_label]
    base = _base_train(cfg, label_to_id)
    n_inject = _n_inject(cfg, base, tid)
    base_no_transfer = base[base["labels"] != tid].copy()

    out: Dict[str, List[str]] = {"full_dataset": [], "baseline": [],
                                 "direct_transfer": [], "synthetic": []}

    # full_dataset: base fold train unchanged (oracle reference).
    for fold in range(pc.n_folds):
        out["full_dataset"].append(_write_protocol(cfg, base, dataset_name, fold, "full_dataset"))

    # baseline: transfer class removed (per-fold CV or fixed base fold).
    for fold in range(pc.n_folds):
        src_fold = fold if pc.baseline_per_fold_cv else pc.injection_base_fold
        df = ds.fold_train_table(cfg.data, src_fold, label_to_id)
        df = df[df["labels"] != tid].copy()
        out["baseline"].append(_write_protocol(cfg, df, dataset_name, fold, "baseline"))

    # direct_transfer: real source transfer-class patients, by absolute path.
    if cfg.source is not None:
        source_patients = ds.source_transfer_patients(cfg.source, cfg.transfer_label)
        for fold in range(pc.n_folds):
            random.seed(pc.seed + fold)
            sel = (random.sample(source_patients, n_inject)
                   if len(source_patients) >= n_inject
                   else random.choices(source_patients, k=n_inject))
            rows = pd.DataFrame([{col: p["full_path"], "labels": tid} for p in sel])
            df = pd.concat([base_no_transfer, rows], ignore_index=True)
            df = df.sample(frac=1, random_state=pc.seed + fold).reset_index(drop=True)
            out["direct_transfer"].append(_write_protocol(cfg, df, dataset_name, fold, "direct_transfer"))

    # synthetic: generated artificial transfer-class patients, by absolute path.
    ext = cfg.data.file_extension
    artificial = [(r["patient_id"], os.path.join(artificial_dir, f"{r['patient_id']}.{ext}"))
                  for r in artificial_records]
    for fold in range(pc.n_folds):
        random.seed(pc.seed + fold)
        np.random.seed(pc.seed + fold)
        sel = (random.sample(artificial, n_inject) if len(artificial) >= n_inject
               else random.choices(artificial, k=n_inject))
        rows = pd.DataFrame([{col: path, "labels": tid} for _, path in sel])
        df = pd.concat([base_no_transfer, rows], ignore_index=True)
        df = df.sample(frac=1, random_state=pc.seed + fold).reset_index(drop=True)
        out["synthetic"].append(_write_protocol(cfg, df, dataset_name, fold, "synthetic"))
    return out


# ===========================================================================
# size scenario (baseline / augmented) -- per fold, namespaced by k_stats
# ===========================================================================
def build_size_protocols(cfg: ExperimentConfig, label_to_id: Dict[str, int], fold: int,
                          k_stats: int, real_rows: pd.DataFrame,
                          artificial_records: List[dict], artificial_dir: str) -> Dict[str, str]:
    """``real_rows`` = the fold's diseased rows + the k kept transfer-class rows
    ([out_patient_col, labels]). baseline = real_rows; augmented = + artificials."""
    dataset_name = f"kstats_{k_stats}"
    col = cfg.data.out_patient_col
    tid = label_to_id[cfg.transfer_label]
    ext = cfg.data.file_extension

    baseline_path = _write_protocol(cfg, real_rows, dataset_name, fold, "baseline")

    art_rows = pd.DataFrame([
        {col: os.path.join(artificial_dir, f"{r['patient_id']}.{ext}"), "labels": tid}
        for r in artificial_records])
    augmented = pd.concat([real_rows, art_rows], ignore_index=True)
    augmented_path = _write_protocol(cfg, augmented, dataset_name, fold, "augmented")
    return {"baseline": baseline_path, "augmented": augmented_path}


# ===========================================================================
# augment scenario -- combine real train + uncertainty-selected artificials
# ===========================================================================
def build_augment_protocol(cfg: ExperimentConfig, fold: int, mode: str,
                           real_train: pd.DataFrame, selected: pd.DataFrame,
                           artificial_dir: str,
                           dataset_name: str = "augment") -> str:
    """``real_train`` = fold train rows ([out_patient_col, labels]); ``selected`` =
    chosen artificial rows ([patient_id, label]). Writes one combined protocol."""
    col = cfg.data.out_patient_col
    ext = cfg.data.file_extension
    art_rows = pd.DataFrame({
        col: [os.path.join(artificial_dir, f"{pid}.{ext}") for pid in selected["patient_id"]],
        "labels": selected["label"].values,
    })
    combined = pd.concat([real_train, art_rows], ignore_index=True)
    return _write_protocol(cfg, combined, dataset_name, fold, mode)


def build_baseline_only(cfg: ExperimentConfig, label_to_id: Dict[str, int],
                        fold: int, dataset_name: str = "augment", mode: str = "baseline") -> str:
    """No-augmentation reference: the fold's real train rows."""
    df = ds.fold_train_table(cfg.data, fold, label_to_id)
    return _write_protocol(cfg, df, dataset_name, fold, mode)
