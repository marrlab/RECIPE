"""Clustering codebook I/O: save/load a fitted codebook and assign cells to clusters.

The codebook itself is fitted by :func:`recipe.clustering.build_codebook` (a
diagonal-covariance GMM); this module only persists it and uses its ``predict``
to label cells. Any object with a ``predict(X) -> cluster ids`` API works.
"""
from __future__ import annotations

import os
import pickle
from typing import List

import pandas as pd


def save_codebook(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def load_codebook(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def assign_clusters_dataframe(records: List[dict], model) -> pd.DataFrame:
    """One row per cell: (patient_id, label, embedding_index, cluster, embedding)."""
    rows = []
    for rec in records:
        pid = rec["patient_id"]
        label = rec["label"]
        feat = rec["features"]
        clusters = model.predict(feat)
        for emb_idx, (embedding, cluster) in enumerate(zip(feat, clusters)):
            rows.append(
                {
                    "patient_id": pid,
                    "label": label,
                    "embedding_index": emb_idx,
                    "cluster": int(cluster),
                    "embedding": embedding,
                }
            )
    return pd.DataFrame(rows)
