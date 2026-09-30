"""The one artificial-patient generator: per-cluster Gaussian count sampling.

A signature gives per-cluster cell-count statistics ``(mean, std)`` (or an exact
count vector). For each synthetic patient we draw a count per cluster from
``N(mean, std)`` (floored at zero, optionally clamped to a max bag size), then
resample that many real cells from the donor pool for that cluster and write the
bag as ``.h5``. All three scenarios use this one generator -- the only thing that
differs is where the signature and the donor pool come from (see the scenario
modules).

Reproducibility note: the per-draw seed is ``seed + patient_idx + cluster``, so
draws collide whenever those two indices sum to the same value (patient 0 /
cluster 5 samples identically to patient 5 / cluster 0). This is kept verbatim
deliberately, because changing it alters every generated bag.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
import tqdm


@dataclass
class GenerationInputs:
    stats_mean: np.ndarray          # per-cluster mean cell count
    stats_std: np.ndarray           # per-cluster std cell count
    n_clusters: int
    cell_pool: pd.DataFrame         # donor cells: cols [cluster, embedding, patient_id, label, embedding_index]
    n_patients: int
    target_label: int               # integer label assigned to synthetic bags
    output_dir: str
    label_name: str = "artificial"  # used in synthetic patient ids / filenames
    seed: int = 42
    # Per-cluster exact counts (used when there is a single reference patient, k=1):
    # if set, every synthetic patient copies this composition instead of sampling.
    exact_counts: Optional[np.ndarray] = None
    max_bag_size: Optional[int] = None


def clamp_bag_size(counts: np.ndarray, max_bag_size: Optional[int], seed: int) -> np.ndarray:
    """Downscale a per-cluster count vector to total <= ``max_bag_size``,
    preserving composition (multinomial re-draw on the same probabilities)."""
    counts = np.asarray(counts, dtype=int)
    if max_bag_size is None:
        return counts
    total = int(counts.sum())
    if total <= int(max_bag_size):
        return counts
    probs = counts / total
    return np.random.default_rng(seed).multinomial(int(max_bag_size), probs).astype(int)


def build_cell_pool(records: List[dict], codebook) -> pd.DataFrame:
    """One row per donor cell: [patient_id, label, embedding_index, cluster, embedding].
    ``records`` are {patient_id, label, features}; clusters come from the codebook."""
    rows = []
    for rec in records:
        pid, label, feat = rec["patient_id"], rec["label"], rec["features"]
        for emb_idx, (embedding, cluster) in enumerate(zip(feat, codebook.predict(feat))):
            rows.append({"patient_id": pid, "label": label, "embedding_index": emb_idx,
                         "cluster": int(cluster), "embedding": embedding})
    return pd.DataFrame(rows)


def generate_gaussian_patients(gi: GenerationInputs) -> List[dict]:
    """Create synthetic patients on disk; return [{patient_id, label, label_name,
    num_embeddings}]. Deterministic given ``gi.seed``."""
    os.makedirs(gi.output_dir, exist_ok=True)
    df_source = gi.cell_pool
    pool_by_cluster = {c: df_source[df_source["cluster"] == c].reset_index(drop=True)
                       for c in range(gi.n_clusters)}

    records: List[dict] = []
    for patient_idx in tqdm.tqdm(range(gi.n_patients),
                                 desc=f"Generating {gi.label_name} patients"):
        if gi.exact_counts is not None:
            counts = np.asarray(gi.exact_counts, dtype=int)
        else:
            np.random.seed(gi.seed + patient_idx)
            counts = np.random.normal(gi.stats_mean, gi.stats_std, gi.n_clusters).astype(int)
            counts = counts * (counts > 0)
        counts = clamp_bag_size(counts, gi.max_bag_size, gi.seed + patient_idx)

        selected_embeddings = []
        metadata = []
        for cluster in range(gi.n_clusters):
            n = int(counts[cluster])
            if n <= 0:
                continue
            df_cluster = pool_by_cluster[cluster]
            if df_cluster.empty:
                continue
            sampled = df_cluster.sample(
                n=n, replace=bool(n > len(df_cluster)),
                random_state=gi.seed + patient_idx + cluster)
            selected_embeddings.extend(sampled["embedding"].tolist())
            for _, row in sampled.iterrows():
                metadata.append({
                    "image_index": row.get("embedding_index"),
                    "original_patient_id": row.get("patient_id"),
                    "original_label": row.get("label"),
                    "cluster": cluster,
                    "target_label": gi.target_label,
                })

        if not selected_embeddings:
            print(f"Warning: no embeddings sampled for {gi.label_name} patient {patient_idx}")
            continue

        feature_array = np.vstack(selected_embeddings)
        patient_id = f"artificial_{gi.label_name}_{patient_idx}"
        saving_path = os.path.join(gi.output_dir, f"{patient_id}.h5")
        metadata_csv = pd.DataFrame(metadata).to_csv(index=False)
        with h5py.File(saving_path, "w") as hf:
            hf.create_dataset("features", data=feature_array)
            hf.create_dataset("labels", data=gi.target_label)
            hf.create_dataset("metadata_csv", data=np.bytes_(metadata_csv))

        records.append({"patient_id": patient_id, "label": gi.target_label,
                        "label_name": gi.label_name, "num_embeddings": len(feature_array)})
    return records
