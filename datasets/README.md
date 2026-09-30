# The dataset contract

RECIPE never hard-codes a cohort. A dataset is one **wide split CSV** plus the
`data:` / `source:` block of a YAML config that points at it. Swapping cohorts
means editing YAML, never Python.

**No cohort file is committed to this repository.** The split CSVs carry patient
identifiers and site-specific paths, so they are `.gitignore`d
(`datasets/*_splits.csv`). Generate your own with `scripts/build_splits.py`.
Only `example_splits.csv` is tracked — it describes the synthetic cohort the
tests build, and contains no real data.

## Schema

One row per patient:

| column | meaning |
|---|---|
| `patient_id` | cohort-unique identifier |
| `label` | free-text class name (`aliases:` in the config can remap it) |
| `tensor_paths` | absolute path of that patient's feature bag |
| `kfold0 … kfold{n-1}` | that patient's role in each fold: `train`, `val` or `test` |

```csv
patient_id,label,tensor_paths,kfold0,kfold1,kfold2,kfold3,kfold4
example_classA_0,classA,/path/to/features/example_classA_0.h5,val,val,val,val,val
example_classA_2,classA,/path/to/features/example_classA_2.h5,train,train,train,train,train
```

Two rules the code relies on:

- **`kfold0` is the fixed evaluation fold.** Its `val`/`test` rows are held out of
  every arm of every scenario, and `recipe/dataset_spec.py` asserts the donor pool
  is disjoint from them. A patient that is `train` in `kfold0` and `test` in
  `kfold3` is fine; a patient in both roles of `kfold0` is not.
- **Roles must partition the cohort within each fold** — exactly one role per
  patient per fold. The builder rejects anything else.

## Feature bags

A bag is one file per patient holding that patient's instance embeddings.

- `.h5` — dataset `features` of shape `(n_instances, dim)`. Any other dataset in
  the file (`labels`, `img_paths`, …) is ignored; the label always comes from the
  split CSV.
- `.pt` — a dict with a `features` key, same shape.

All patients in one cohort must share `dim`. Feature extraction is out of scope
for this repository: bring embeddings from whatever encoder suits your modality.

## Building a split CSV

**From an existing fold layout** (`data_fold_{i}/{train,val,test}.csv` with columns
`patient_name,labels`, plus a `label_map.csv`) — this preserves an existing fold
assignment exactly, so published splits stay intact:

```bash
python scripts/build_splits.py \
  --folds-dir   /path/to/features/folds \
  --feature-dir /path/to/features \
  --out datasets/my_cohort_splits.csv
```

**From a flat label table**, when no fold assignment exists yet — rotating
stratified folds, block `i` is `test`, block `i+1` is `val`, the rest `train`:

```bash
python scripts/build_splits.py \
  --labels-csv  /path/to/labels.csv \
  --feature-dir /path/to/features \
  --make-folds --out datasets/my_cohort_splits.csv
```

The script prints the patient count, the per-class counts and the `kfold0` split
sizes, so you can sanity-check the cohort before training.

## Label aliasing

Cohorts name the same biological class differently. Rather than editing CSVs, map
them in the config:

```yaml
data:
  name: aml_hehr
  splits_csv: datasets/aml_hehr_splits.csv
  aliases: {"SCD": healthy}        # stem-cell donor controls are the healthy class
source:
  name: beluga
  splits_csv: datasets/beluga_splits.csv
  aliases: {"No malignancy": healthy}
```

The map is partial: labels not listed pass through unchanged. The shared label id
space is the union of the target's labels and the transfer label.
