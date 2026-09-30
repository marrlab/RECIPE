"""Per-cluster cell-count signature -- the composition profile the generator
samples from. Used identically by all scenarios; only the cells fed in differ.

* transfer : cells of the source transfer-class patients.
* size     : cells of the k real target patients that remain.
* augment  : cells of one class's patients (called once per class).

When a single reference patient is given (size, k=1) the signature is that
patient's exact per-cluster counts; otherwise it is the mean/std across patients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Signature:
    mean: np.ndarray                 # per-cluster mean count
    std: np.ndarray                  # per-cluster std count
    pivot: pd.DataFrame              # patients x clusters count matrix
    exact: Optional[np.ndarray] = None  # set iff a single reference patient


def cluster_count_pivot(df_cells: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """patients x clusters count matrix from a per-cell table [patient_id, cluster]."""
    counts = (df_cells.groupby(["patient_id", "cluster"]).size()
              .reset_index(name="count")
              .pivot_table(index="patient_id", columns="cluster", values="count", fill_value=0))
    for c in range(n_clusters):
        if c not in counts.columns:
            counts[c] = 0
    return counts[sorted(counts.columns)]


def compute_signature(df_cells: pd.DataFrame, n_clusters: int) -> Signature:
    """Mean/std of per-patient per-cluster counts. ``exact`` is set when there is
    exactly one reference patient (the few-shot k=1 case copies it verbatim)."""
    pivot = cluster_count_pivot(df_cells, n_clusters)
    mean = pivot.mean(axis=0).values
    std = pivot.std(axis=0).values
    exact = pivot.iloc[0].values.astype(int) if len(pivot) == 1 else None
    return Signature(mean=mean, std=std, pivot=pivot, exact=exact)


def cluster_stats_per_patient(df: pd.DataFrame, n_clusters: int
                              ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Back-compat tuple form ``(mean, std, pivot)`` used by the transfer pipeline."""
    sig = compute_signature(df, n_clusters)
    return sig.mean, sig.std, sig.pivot
