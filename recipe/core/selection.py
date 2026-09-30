"""Uncertainty-driven selection of artificial patients (augment scenario).

Pure pandas over an uncertainty table with columns [patient_id, label, <metric>].
Three strategies, each adding a number of artificials proportional to the real
training-set size:

* global    : top X% of all artificials by ``metric``.
* per_class : top X% within each class (X% of that class's real count).
* mixed     : half of X% globally + half per-class (de-duplicated).

``highest=True`` keeps the most uncertain; ``False`` the least.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def select_global_top(df: pd.DataFrame, metric: str, top_percentage: float,
                      highest: bool, real_patient_count: int) -> pd.DataFrame:
    sorted_df = df.sort_values(metric, ascending=not highest)
    n = int(real_patient_count * (top_percentage / 100))
    return sorted_df.head(n)


def select_per_class_top(df: pd.DataFrame, metric: str, top_percentage: float,
                         real_class_counts: Dict[int, int], highest: bool) -> pd.DataFrame:
    parts = []
    for label, group in df.groupby("label"):
        sorted_group = group.sort_values(metric, ascending=not highest)
        n = int(real_class_counts.get(label, 0) * (top_percentage / 100))
        parts.append(sorted_group.head(n))
    return pd.concat(parts) if parts else df.iloc[0:0]


def select_mixed_top(df: pd.DataFrame, metric: str, top_percentage: float,
                     highest: bool, real_patient_count: int,
                     real_class_counts: Dict[int, int]) -> pd.DataFrame:
    half = top_percentage / 2
    g = select_global_top(df, metric, half, highest, real_patient_count)
    pc = select_per_class_top(df, metric, half, real_class_counts, highest)
    return pd.concat([g, pc]).drop_duplicates()


# Canonical strategy name -> short token used in protocol filenames.
STRATEGY_TOKEN = {"global": "global", "mixed": "mixed", "per_class": "perclass"}


def select(df: pd.DataFrame, *, metric: str, strategy: str, top_percentage: float,
           highest: bool, real_patient_count: int,
           real_class_counts: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    """Dispatch to one strategy. Returns the selected subset of ``df``."""
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in uncertainty table columns {list(df.columns)}")
    real_class_counts = real_class_counts or {}
    if strategy == "global":
        return select_global_top(df, metric, top_percentage, highest, real_patient_count)
    if strategy == "per_class":
        return select_per_class_top(df, metric, top_percentage, real_class_counts, highest)
    if strategy == "mixed":
        return select_mixed_top(df, metric, top_percentage, highest,
                                real_patient_count, real_class_counts)
    raise ValueError(f"Unknown selection strategy '{strategy}'")
