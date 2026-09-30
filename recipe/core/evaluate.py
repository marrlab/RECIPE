"""Aggregate per-fold confusion matrices into balanced accuracy.

Balanced accuracy per fold = mean per-class recall = nanmean(diag / row_sum) of
``test_conf_matrix.npy``. Results live in this run's own directory
(``_runs/<run_name>/<arch>/<result>/fold_*``), so evaluation just globs them and
parses the protocol mode from each result folder name -- arm-agnostic, so it
works for every scenario (transfer arms, size baseline/augmented, or the augment
grid's metric/strategy/percentage modes).
"""
from __future__ import annotations

import glob
import os
import re
from typing import List

import numpy as np
import pandas as pd

from ..config import ExperimentConfig


def balanced_accuracy_from_confusion(cm: np.ndarray) -> float:
    # A class absent from the test set has an all-zero row; its recall is nan by
    # design and is skipped by nanmean rather than counted as zero.
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(cm) / cm.sum(axis=1)
    return float(np.nanmean(recall))


def _parse_mode(result_dir: str, arch: str) -> str:
    """Extract the protocol mode from a result folder ``{arch}_k{k}_{mode}_lr..._seed...``."""
    name = os.path.basename(result_dir)
    m = re.match(rf"^{re.escape(arch)}_k\d+_(.*)_lr", name)
    return m.group(1) if m else name


# Intermediate models that share the results tree but are not experimental arms.
# The augment scenario's MC-dropout selector is trained only to score artificial
# patients; reporting it beside the arms would invite comparing the two.
INTERNAL_MODES = frozenset({"selector"})


def collect_all(cfg: ExperimentConfig) -> pd.DataFrame:
    records = []
    for arch in cfg.train.architectures:
        arch_root = os.path.join(cfg.output_root, arch)
        # <arch>/<dataset_name>/<result_name>/fold_*/
        for result_dir in sorted(glob.glob(os.path.join(arch_root, "*", "*"))):
            cms = sorted(glob.glob(os.path.join(result_dir, "fold_*", "test_conf_matrix.npy")))
            if not cms:
                continue
            if _parse_mode(result_dir, arch) in INTERNAL_MODES:
                continue
            baccs = [balanced_accuracy_from_confusion(np.load(p)) for p in cms]
            records.append({
                "architecture": arch,
                "dataset": os.path.basename(os.path.dirname(result_dir)),
                "mode": _parse_mode(result_dir, arch),
                "bacc_mean": float(np.mean(baccs)),
                "bacc_std": float(np.std(baccs, ddof=1)) if len(baccs) > 1 else 0.0,
                "n_folds": len(baccs),
                "per_fold": baccs,
            })
    return pd.DataFrame(records)
