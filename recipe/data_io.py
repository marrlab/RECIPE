"""Feature/CSV loading helpers (mirrors the notebooks' h5 access)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import tqdm


def load_features(filepath: str) -> np.ndarray:
    """Load a patient's feature matrix, squeezing a leading singleton axis."""
    with h5py.File(filepath, "r") as hf:
        feat = hf["features"][()]
    if feat.ndim == 3:
        feat = feat.squeeze(0)
    return feat


def load_features_with_label(filepath: str) -> Tuple[np.ndarray, Optional[object]]:
    with h5py.File(filepath, "r") as hf:
        feat = hf["features"][()]
        if feat.ndim == 3:
            feat = feat.squeeze(0)
        label = hf["labels"][()] if "labels" in hf else None
    return feat, label


def parallel_load_features(
    patient_ids: List[str], ext: str, path_of_dataset: str, max_workers: int = 16
) -> List[Optional[np.ndarray]]:
    """Load features for many patients in parallel, preserving input order."""
    feature_arrays: List[Optional[np.ndarray]] = [None] * len(patient_ids)
    pid_to_index = {pid: i for i, pid in enumerate(patient_ids)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_features, os.path.join(path_of_dataset, f"{pid}.{ext}")): pid
            for pid in patient_ids
        }
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"Loading *.{ext}"):
            pid = futures[fut]
            try:
                feature_arrays[pid_to_index[pid]] = fut.result()
            except Exception as exc:  # pragma: no cover - surfaced, not swallowed silently
                print(f"Failed to load {pid}: {exc}")
    return feature_arrays


def subsample_rows(feat: Optional[np.ndarray], cap: Optional[int], seed: int) -> Optional[np.ndarray]:
    """Randomly keep at most ``cap`` rows of a (cells, dim) matrix (seeded, no
    replacement). ``cap=None`` / small / None inputs pass through unchanged."""
    if cap is None or feat is None or feat.shape[0] <= cap:
        return feat
    idx = np.sort(np.random.default_rng(seed).choice(feat.shape[0], size=cap, replace=False))
    return feat[idx]


def load_features_for_paths(
    paths: List[str], cap: Optional[int] = None, seed: int = 0, max_workers: int = 16
) -> List[Optional[np.ndarray]]:
    """Load features for absolute file paths in parallel, preserving order.

    Mirrors ``parallel_load_features`` but takes complete paths (the split_csv
    datasets store an absolute ``tensor_paths`` column instead of bare ids). When
    ``cap`` is set each bag is subsampled to ``cap`` cells inside the worker, so a
    huge donor target (e.g. a WSI cohort, tens of millions of cells) never lands fully in memory."""
    out: List[Optional[np.ndarray]] = [None] * len(paths)

    def _load(i: int, p: str) -> Optional[np.ndarray]:
        return subsample_rows(load_features(p), cap, seed + i)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load, i, p): i for i, p in enumerate(paths)}
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Loading by path"):
            i = futures[fut]
            try:
                out[i] = fut.result()
            except Exception as exc:  # pragma: no cover
                print(f"Failed to load {paths[i]}: {exc}")
    return out


def load_patient_records(patient_ids: List[str], path_of_dataset: str, ext: str) -> List[dict]:
    """Load (patient_id, features, label) records sequentially, label from the h5."""
    records = []
    for pid in tqdm.tqdm(patient_ids, desc="Loading patient records"):
        filepath = os.path.join(path_of_dataset, f"{pid}.{ext}")
        try:
            feat, label = load_features_with_label(filepath)
            records.append({"patient_id": pid, "filepath": filepath, "features": feat, "label": label})
        except Exception as exc:
            print(f"Failed to load {pid}: {exc}")
    return records


def read_csv_patient_ids(csv_path: str, patient_col: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    ids = df[patient_col].astype(str).tolist()
    return df, ids
