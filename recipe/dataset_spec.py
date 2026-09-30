"""Generic dataset access driven entirely by a :class:`DataConfig`.

There is no dataset registry and no dataset-specific branch anywhere: every
accessor reads the wide split CSV named in the config (absolute feature path per
patient, a string label, and a ``train/val/test`` role per fold in
``kfold0..k{n-1}``) and harmonizes labels through the config's optional alias
map. Swapping datasets means editing the YAML ``data:``/``source:`` block.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import tqdm

from . import labels as lh
from .config import DataConfig
from .data_io import load_features, load_features_for_paths


# --------------------------------------------------------------------------
# split_csv helpers
# --------------------------------------------------------------------------
def split_table(cfg: DataConfig) -> pd.DataFrame:
    """Normalized table: [patient_id, feature_path, label (canonical), kfold0..k]."""
    df = pd.read_csv(cfg.splits_csv)
    out = pd.DataFrame({
        "patient_id": df[cfg.id_col].astype(str),
        "feature_path": df[cfg.path_col].astype(str),
        "label": [lh.canonical_label(v, cfg.aliases) for v in df[cfg.label_col]],
    })
    for f in range(cfg.n_fold_cols):
        out[f"kfold{f}"] = df[f"kfold{f}"]
    return out


def split_fold_rows(cfg: DataConfig, fold: int, role: str) -> pd.DataFrame:
    tbl = split_table(cfg)
    return tbl[tbl[f"kfold{fold}"] == role].reset_index(drop=True)


# --------------------------------------------------------------------------
# Label space
# --------------------------------------------------------------------------
def prepare_label_map(cfg: DataConfig, transfer_label: Optional[str] = None) -> Dict[str, int]:
    """Canonical-name -> id over the target's present labels (plus the transfer
    label, if given). train_5fold rebuilds this exact map from the split CSV."""
    tbl = split_table(cfg)
    names = set(tbl["label"])
    if transfer_label is not None:
        names = names | {transfer_label}
    return lh.build_label_map(names)


def eval_feature_paths(cfg: DataConfig, eval_fold: int = 0) -> set:
    """Feature-bag paths held out for evaluation in ``eval_fold`` (val/test).
    The donor pool must be disjoint from these -- the anti-leakage guarantee.

    Defaults to fold 0 because `transfer` and `size` always evaluate there; the
    `augment` scenario is real cross-validation and passes the fold it is on."""
    tbl = split_table(cfg)
    col = f"kfold{eval_fold}"
    return set(tbl[tbl[col].isin(("val", "test"))]["feature_path"].astype(str))


# --------------------------------------------------------------------------
# Target training rows (base for protocol arms)
# --------------------------------------------------------------------------
def fold_train_table(cfg: DataConfig, fold: int, label_to_id: Dict[str, int]) -> pd.DataFrame:
    """Train rows for ``fold`` as [out_patient_col, labels:int]."""
    rows = split_fold_rows(cfg, fold, "train")
    return pd.DataFrame({
        cfg.out_patient_col: rows["feature_path"].values,
        "labels": [label_to_id[l] for l in rows["label"]],
    })


def fold_train_records(cfg: DataConfig, fold: int, label_to_id: Dict[str, int],
                       labels_keep: Optional[set] = None,
                       max_cells_per_patient: Optional[int] = None,
                       subsample_seed: int = 0,
                       enforce_no_leak: bool = True,
                       eval_fold: Optional[int] = None) -> List[dict]:
    """Per-patient {patient_id, label:int, label_name, feature_path, features} for
    a fold's train split, optionally restricted to ``labels_keep`` (canonical
    names). Used by the size/augment scenarios that fit a GMM on the fold itself."""
    rows = split_fold_rows(cfg, fold, "train")
    if labels_keep is not None:
        rows = rows[rows["label"].isin(labels_keep)].reset_index(drop=True)
    if enforce_no_leak:
        ef = 0 if eval_fold is None else eval_fold
        leak = set(rows["feature_path"].astype(str)) & eval_feature_paths(cfg, ef)
        if leak:
            raise AssertionError(
                f"Slide leakage: {len(leak)} train slide(s) also in val/test of "
                f"kfold{ef}, e.g. {sorted(leak)[:2]}.")
    feats = load_features_for_paths(
        rows["feature_path"].tolist(), cap=max_cells_per_patient, seed=subsample_seed)
    out = []
    for pid, lbl, fp, feat in zip(rows["patient_id"], rows["label"],
                                  rows["feature_path"], feats):
        if feat is None:
            continue
        out.append({"patient_id": str(pid), "label": label_to_id[lbl],
                    "label_name": lbl, "feature_path": str(fp), "features": feat})
    return out


# --------------------------------------------------------------------------
# Source-side: codebook fit set, signature records, real transfer patients
# --------------------------------------------------------------------------
def _h5_n_cells(path: str) -> int:
    with h5py.File(path, "r") as hf:
        s = hf["features"].shape
    return s[-2] if len(s) >= 2 else s[0]


def _h5_sample(path: str, cap: int, seed: int, n_blocks: int = 16) -> np.ndarray:
    """Read ~``cap`` cells as a few contiguous blocks spread across the slide
    (fast sequential reads over networked HDF5, with slide-wide coverage)."""
    with h5py.File(path, "r") as hf:
        dset = hf["features"]
        ndim = dset.ndim
        n = dset.shape[-2] if ndim >= 2 else dset.shape[0]
        if cap >= n:
            arr = dset[()]
        else:
            nb = int(min(n_blocks, cap))
            per = cap // nb
            rng = np.random.default_rng(seed)
            starts = np.linspace(0, n - per, nb).astype(int)
            parts = []
            for s in starts:
                s = int(min(max(s + rng.integers(-per, per + 1), 0), n - per))
                parts.append(dset[0, s:s + per] if ndim == 3 else dset[s:s + per])
            arr = np.concatenate(parts, axis=0)
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    return arr


def source_fit_matrix(cfg: DataConfig, max_samples: Optional[int] = None,
                      seed: int = 42, max_workers: int = 16) -> np.ndarray:
    """Codebook-fit matrix from a cohort's fold-0 train split. Streams bags into a
    preallocated array; ``max_samples`` reads only a per-bag slice (WSI scale)."""
    rows = split_fold_rows(cfg, 0, "train")
    paths = rows["feature_path"].tolist()

    if max_samples is not None:
        cap = max(1, int(np.ceil(max_samples / len(paths))))
        def _samp(i_p):
            i, p = i_p
            return _h5_sample(p, cap, seed + i)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            parts = list(tqdm.tqdm(ex.map(_samp, enumerate(paths)), total=len(paths),
                                   desc=f"Sampling <= {cap}/bag"))
        return np.concatenate(parts, axis=0)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        counts = list(tqdm.tqdm(ex.map(_h5_n_cells, paths), total=len(paths),
                                desc="Reading bag sizes"))
    with h5py.File(paths[0], "r") as hf:
        dim = hf["features"].shape[-1]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    total = int(offsets[-1])
    X = np.empty((total, dim), dtype=np.float32)

    def _load_into(i: int) -> None:
        X[offsets[i]:offsets[i + 1]] = load_features(paths[i])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_load_into, i) for i in range(len(paths))]
        for _ in tqdm.tqdm(as_completed(futs), total=len(futs), desc="Streaming source bags"):
            _.result()
    return X


def source_transfer_records(cfg: DataConfig, transfer_label: str) -> List[dict]:
    """Source transfer-class patients only: {patient_id, label, features} (cells
    needed for the per-patient cluster-count signature)."""
    rows = split_fold_rows(cfg, 0, "train")
    rows = rows[rows["label"] == transfer_label]
    feats = load_features_for_paths(rows["feature_path"].tolist())
    return [{"patient_id": str(p), "label": transfer_label, "features": f}
            for p, f in zip(rows["patient_id"], feats) if f is not None]


def source_transfer_patients(cfg: DataConfig, transfer_label: str) -> List[dict]:
    """Real source transfer-class patients as {patient_id, full_path} (the donors
    for the direct-transfer arm)."""
    rows = split_fold_rows(cfg, 0, "train")
    sel = rows[rows["label"] == transfer_label]
    return [{"patient_id": str(p), "full_path": str(fp)}
            for p, fp in zip(sel["patient_id"], sel["feature_path"])]


# --------------------------------------------------------------------------
# Target donor cells (non-transfer cells of the base fold)
# --------------------------------------------------------------------------
def target_donor_records(cfg: DataConfig, base_fold: int, transfer_label: str,
                         label_to_id: Dict[str, int],
                         max_cells_per_patient: Optional[int] = None,
                         subsample_seed: int = 0) -> List[dict]:
    """Target non-transfer patients of ``base_fold``: {patient_id, label:int,
    features}. These cells are the donor pool the generator draws from."""
    rows = split_fold_rows(cfg, base_fold, "train")
    rows = rows[rows["label"] != transfer_label]

    leak = set(rows["feature_path"].astype(str)) & eval_feature_paths(cfg)
    if leak:
        raise AssertionError(
            f"Slide leakage: {len(leak)} donor slide(s) also in val/test (kfold0), "
            f"e.g. {sorted(leak)[:2]}.")

    feats = load_features_for_paths(
        rows["feature_path"].tolist(), cap=max_cells_per_patient, seed=subsample_seed)
    out = []
    for pid, lbl, feat in zip(rows["patient_id"], rows["label"], feats):
        if feat is None:
            continue
        out.append({"patient_id": str(pid), "label": label_to_id[lbl], "features": feat})
    return out
