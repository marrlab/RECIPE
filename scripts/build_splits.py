#!/usr/bin/env python
"""Build the wide split CSV that RECIPE reads, from data you already have.

No cohort file ships with this repository -- the split CSVs contain patient
identifiers and site-specific paths. This script regenerates one from a feature
directory plus either an existing fold layout or a flat label table.

Output schema (one row per patient), as consumed by ``recipe/dataset_spec.py``::

    patient_id,label,tensor_paths,kfold0,kfold1,...,kfold{n-1}

``label`` is a free-text class name; ``tensor_paths`` is the absolute path of the
patient's feature bag; each ``kfold{i}`` is ``train``, ``val`` or ``test``.
``kfold0`` is the fixed evaluation fold -- its ``val``/``test`` rows are held out
of every arm, and the donor pool is asserted disjoint from them.

Two modes
---------
1. ``--folds-dir`` -- convert an existing ``data_fold_{i}/{train,val,test}.csv``
   layout (columns ``patient_name,labels``) plus a label map. This preserves an
   existing fold assignment exactly::

     python scripts/build_splits.py \
       --folds-dir   /path/to/features/folds \
       --feature-dir /path/to/features \
       --out datasets/aml_hehr_splits.csv

2. ``--make-folds`` -- derive rotating stratified folds from a flat
   ``patient_id,label`` table when no fold assignment exists yet::

     python scripts/build_splits.py \
       --labels-csv  /path/to/labels.csv \
       --feature-dir /path/to/features \
       --make-folds --out datasets/my_cohort_splits.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

import pandas as pd

ROLES = ("train", "val", "test")


# --------------------------------------------------------------------------
# label map
# --------------------------------------------------------------------------
# Column-name pairs seen in the wild, tried in order when not given explicitly.
_LABEL_MAP_COLS = (("label_id", "label_name"), ("label", "diagnose"), ("label", "label_name"))


def read_label_map(path: Optional[str], id_col: Optional[str] = None,
                   name_col: Optional[str] = None) -> Optional[Dict[int, str]]:
    """``{label_id: label_name}`` from a two-column CSV. Column names are
    auto-detected across the layouts this project has produced unless given."""
    if path is None:
        return None
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    if id_col and name_col:
        pair = (id_col, name_col)
        if not set(pair) <= set(df.columns):
            raise SystemExit(f"{path}: no columns {pair}, got {list(df.columns)}")
    else:
        pair = next((( lower[a], lower[b]) for a, b in _LABEL_MAP_COLS
                     if a in lower and b in lower), None)
        if pair is None:
            raise SystemExit(
                f"{path}: could not find a label-id/label-name column pair in "
                f"{list(df.columns)}; pass --label-id-col and --label-name-col")
    return {int(r[pair[0]]): str(r[pair[1]]) for _, r in df.iterrows()}


def _default_label_map(folds_dir: str) -> Optional[str]:
    for name in ("label_map.csv", "label_mapping.csv", "label_to_diagnose.csv"):
        p = os.path.join(folds_dir, name)
        if os.path.exists(p):
            return p
    return None


# --------------------------------------------------------------------------
# mode 1: an existing data_fold_{i}/{train,val,test}.csv layout
# --------------------------------------------------------------------------
def from_folds_dir(folds_dir: str, n_folds: int, id_col: str,
                   label_map: Optional[Dict[int, str]]) -> pd.DataFrame:
    roles: Dict[str, Dict[str, str]] = {}      # patient -> {kfold_i: role}
    raw_label: Dict[str, object] = {}

    for fold in range(n_folds):
        fold_dir = os.path.join(folds_dir, f"data_fold_{fold}")
        if not os.path.isdir(fold_dir):
            raise SystemExit(f"missing fold directory: {fold_dir}")
        for role in ROLES:
            csv = os.path.join(fold_dir, f"{role}.csv")
            if not os.path.exists(csv):
                raise SystemExit(f"missing split file: {csv}")
            df = pd.read_csv(csv)
            if id_col not in df.columns:
                raise SystemExit(
                    f"{csv}: no '{id_col}' column (got {list(df.columns)}); use --id-col")
            label_col = "labels" if "labels" in df.columns else "label"
            if label_col not in df.columns:
                raise SystemExit(f"{csv}: no 'labels'/'label' column")
            for pid, lbl in zip(df[id_col].astype(str), df[label_col]):
                prev = roles.setdefault(pid, {})
                key = f"kfold{fold}"
                if key in prev and prev[key] != role:
                    raise SystemExit(
                        f"patient '{pid}' is both '{prev[key]}' and '{role}' in fold "
                        f"{fold}. Roles must partition the cohort within each fold.")
                prev[key] = role
                if pid in raw_label and raw_label[pid] != lbl:
                    raise SystemExit(
                        f"patient '{pid}' has conflicting labels {raw_label[pid]} / {lbl}")
                raw_label[pid] = lbl

    rows = []
    for pid in sorted(roles):
        per_fold = roles[pid]
        missing = [f"kfold{f}" for f in range(n_folds) if f"kfold{f}" not in per_fold]
        if missing:
            raise SystemExit(f"patient '{pid}' has no role in {', '.join(missing)}")
        lbl = raw_label[pid]
        if label_map is not None:
            try:
                lbl = label_map[int(lbl)]
            except (KeyError, ValueError, TypeError):
                raise SystemExit(f"label {lbl!r} of patient '{pid}' is not in the label map")
        row = {"patient_id": pid, "label": str(lbl)}
        row.update(per_fold)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# mode 2: rotating stratified folds from a flat label table
# --------------------------------------------------------------------------
def make_folds(labels_csv: str, n_folds: int, id_col: str, label_col: str,
               seed: int) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv(labels_csv)
    for c in (id_col, label_col):
        if c not in df.columns:
            raise SystemExit(f"{labels_csv}: no '{c}' column (got {list(df.columns)})")
    df = df[[id_col, label_col]].astype(str).drop_duplicates(subset=[id_col])
    df = df.sort_values(id_col).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    blocks = [test_idx for _, test_idx in skf.split(df, df[label_col])]

    out = pd.DataFrame({"patient_id": df[id_col].values, "label": df[label_col].values})
    for fold in range(n_folds):
        # rotate: block `fold` is test, the next one is val, the rest train
        role = pd.Series(["train"] * len(df))
        role.iloc[blocks[fold]] = "test"
        role.iloc[blocks[(fold + 1) % n_folds]] = "val"
        out[f"kfold{fold}"] = role.values
    return out


# --------------------------------------------------------------------------
def attach_paths(df: pd.DataFrame, feature_dir: str, ext: str,
                 check_missing: bool) -> pd.DataFrame:
    ext = ext.lstrip(".")
    feature_dir = os.path.abspath(feature_dir)
    df = df.copy()
    df["tensor_paths"] = [os.path.join(feature_dir, f"{pid}.{ext}") for pid in df["patient_id"]]
    missing = [p for p in df["tensor_paths"] if not os.path.exists(p)]
    if missing:
        msg = f"{len(missing)} feature bag(s) not found, e.g. {missing[:3]}"
        if check_missing:
            raise SystemExit(msg + "\n(pass --allow-missing to write the CSV anyway)")
        print(f"WARNING: {msg}", file=sys.stderr)
    return df


def summarize(df: pd.DataFrame, n_folds: int) -> None:
    print(f"\n{len(df)} patients, {df['label'].nunique()} labels")
    for name, n in df["label"].value_counts().sort_index().items():
        print(f"  {name:<28} {n}")
    counts = df["kfold0"].value_counts()
    print("kfold0 (the fixed evaluation fold): "
          + " / ".join(f"{counts.get(r, 0)} {r}" for r in ROLES))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--folds-dir", help="directory holding data_fold_{i}/{train,val,test}.csv")
    src.add_argument("--labels-csv", help="flat table of patients and labels (with --make-folds)")

    ap.add_argument("--feature-dir", required=True, help="directory of per-patient feature bags")
    ap.add_argument("--out", required=True, help="split CSV to write")
    ap.add_argument("--file-extension", default="h5", choices=["h5", "pt"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--label-map", default=None,
                    help="CSV mapping integer labels to names "
                         "(default: <folds-dir>/label_map.csv)")
    ap.add_argument("--label-id-col", default=None, help="label-map id column (auto-detected)")
    ap.add_argument("--label-name-col", default=None, help="label-map name column (auto-detected)")
    ap.add_argument("--id-col", default=None,
                    help="patient column (default: patient_name for --folds-dir, "
                         "patient_id for --labels-csv)")
    ap.add_argument("--label-col", default="label", help="label column of --labels-csv")
    ap.add_argument("--make-folds", action="store_true",
                    help="derive rotating stratified folds (requires --labels-csv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-missing", action="store_true",
                    help="write the CSV even if some feature bags are absent")
    args = ap.parse_args(argv)

    if args.folds_dir:
        id_col = args.id_col or "patient_name"
        lm_path = args.label_map or _default_label_map(args.folds_dir)
        label_map = read_label_map(lm_path, args.label_id_col, args.label_name_col)
        df = from_folds_dir(args.folds_dir, args.n_folds, id_col, label_map)
        print(f"folds:      {args.folds_dir}")
        print(f"label map:  {lm_path or '(none -- integer labels kept as-is)'}")
    else:
        if not args.make_folds:
            raise SystemExit("--labels-csv requires --make-folds")
        id_col = args.id_col or "patient_id"
        df = make_folds(args.labels_csv, args.n_folds, id_col, args.label_col, args.seed)
        print(f"labels:     {args.labels_csv} (rotating stratified {args.n_folds}-fold, "
              f"seed {args.seed})")

    df = attach_paths(df, args.feature_dir, args.file_extension, not args.allow_missing)
    df = df[["patient_id", "label", "tensor_paths"]
            + [f"kfold{f}" for f in range(args.n_folds)]]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    summarize(df, args.n_folds)
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
