# RECIPE

**Re-mixing Embeddings via Clustering for Patient augmEntation** — reference
implementation for *Re-mixing Embeddings for Patient Augmentation in Data Scarce
Multiple Instance Learning* ([arXiv:2606.25770](https://arxiv.org/abs/2606.25770)).

RECIPE generates artificial patients for multiple-instance-learning (MIL) cohorts
directly in embedding space. A Gaussian Mixture Model over pooled instance
embeddings defines unsupervised clusters; a class's **recipe** is the distribution
of its instances across those clusters. New patients are made by drawing a
per-cluster instance count from the recipe and resampling that many *real*
embeddings from the matching clusters. Nothing is synthesised in feature space —
every instance in an artificial bag is a real instance from a real patient.

One dataset-agnostic implementation covers three scenarios. Nothing about a
cohort is hard-coded — a dataset is a CSV plus a YAML block, so applying RECIPE
to your own data means writing a config, not editing Python:

| Scenario | Question | Where the recipe comes from | Where the instances come from |
|---|---|---|---|
| **transfer** | a cohort is missing a whole class | an **external** cohort's class recipe | the target cohort's other classes |
| **size** | only *k* real patients of a class exist | those *k* patients | the target cohort |
| **augment** | does adding artificial patients help? | each class's own patients | the target cohort |

In `augment`, the generated patients are then ranked by **MC-dropout uncertainty**
and only the most informative top-X% are kept for retraining.

**To use it on your data** you need two things: one feature bag per patient, and
a CSV saying which class each patient is and which fold they fall in
(`scripts/build_splits.py` writes that CSV for you). Then copy one of the example
configs and change the paths. There is no dataset registry and no per-cohort
branch anywhere in the code.

---

## Install

```bash
conda create -n recipe -c conda-forge --override-channels python=3.11 -y
conda activate recipe
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

`torch` is pinned to a CUDA 12.4 build because it works across older and newer
GPU drivers; adjust the index URL for your hardware. If your environment has a
different name, set `RECIPE_ENV` (the emitted SLURM scripts honour it) or change
`train.conda_env` in the config.

## Data

**No cohort data ships with this repository.** RECIPE starts from *pre-extracted*
instance embeddings — one feature bag per patient — plus a CSV describing labels
and fold membership. Feature extraction is out of scope — bring your own
embeddings (a pathology/cytology foundation model for image cohorts, PCA or a
learned latent for single-cell modalities). Any modality works: the pipeline
only ever sees an `(n_instances, dim)` matrix per patient.

Build the split CSV for your cohort first:

```bash
python scripts/build_splits.py \
  --folds-dir   /path/to/features/folds \      # data_fold_{i}/{train,val,test}.csv
  --feature-dir /path/to/features \            # <patient_id>.h5
  --out datasets/my_cohort_splits.csv
```

`datasets/README.md` documents the schema, the `--make-folds` mode for cohorts
with no fold assignment yet, and the label-aliasing mechanism.
`datasets/example_splits.csv` is a tiny synthetic example.

## Running an experiment

```bash
python -m recipe <scenario> <stage> -c configs/<config>.yaml
```

`scenario` is `transfer | size | augment`; `stage` is one of `run` (chain all the
preparation stages), `fit-codebook`, `generate`, `protocols`, `train-selector`,
`select`, `train`, `evaluate`.

```bash
# Scenario 1 -- transfer: an external healthy recipe fills a missing control class
python -m recipe transfer run      -c configs/transfer.yaml
python -m recipe transfer train    -c configs/transfer.yaml
python -m recipe transfer evaluate -c configs/transfer.yaml

# Scenario 2 -- size: keep k real patients, refill the class from their recipe
python -m recipe size run      -c configs/size.yaml
python -m recipe size train    -c configs/size.yaml
python -m recipe size evaluate -c configs/size.yaml

# Scenario 3 -- augment: generate per class, then keep the most uncertain
python -m recipe augment run      -c my_augment.yaml
python -m recipe augment train    -c my_augment.yaml
python -m recipe augment evaluate -c my_augment.yaml
```

> **The shipped configs are worked examples, not fixed experiments.** Copy one,
> point `data.splits_csv` at your own cohort and edit the class names — that is
> the whole porting effort. `configs/transfer.yaml` and `configs/size.yaml` are
> complete examples; for `augment`, write the `augment:` block documented in
> [datasets/README.md](datasets/README.md).

`--dry-run` prints the training commands instead of running them — always worth
doing before a large job. On a SLURM cluster, one script prepares the run and
submits the training array:

```bash
sbatch scripts/run_scenario.sh transfer configs/transfer.yaml
sbatch scripts/run_scenario.sh size     configs/size.yaml
sbatch scripts/run_scenario.sh augment  my_augment.yaml
```

Partition, QoS, walltime, CPUs and memory for the training array come from the
`train.slurm_*` fields of the config — the defaults are for the cluster this was
developed on and will need changing.

> If training dies with `RuntimeError: torch_shm_manager ... Invalid argument`,
> your `TMPDIR` is too long: the trainer uses torch's `file_system` sharing
> strategy, whose unix socket path is capped at 108 bytes. Point `TMPDIR`
> somewhere short, or pass `--num_workers 0`.

## Method

1. **Codebook** — fit a diagonal-covariance GMM (K=50) over the pooled instance
   embeddings and assign every instance a hard cluster.
2. **Signature** — per cluster, the mean and standard deviation of the instance
   count across the reference patients. With a single reference patient the
   signature is that patient's exact count vector.
3. **Generate** — per artificial patient, draw a count per cluster from
   `N(mean, std)` (floored at zero), then resample that many real instances from
   the donor pool for that cluster and write the bag.
4. **Select** (`augment` only) — score each artificial patient by MC-dropout
   uncertainty (predictive entropy, BALD, or max per-class std over T passes) and
   keep the top-X% globally, per class, or mixed.
5. **Train / evaluate** — ABMIL / Transformer / Mean / Max through
   `mil/train_5fold.py`; balanced accuracy is `nanmean(diag(cm) / cm.sum(1))` per
   fold, averaged over folds.

Training hyperparameters (all config-settable): AdamW, lr 5e-5, weight decay
0.01, 150 epochs, early stopping patience 15 on validation loss,
`ReduceLROnPlateau`, batch size 1 with gradient accumulation 20, seed 38.

> Bags vary in length, so the trainer runs with `batch_size=1` and accumulates
> gradients instead: `--grad_accum` (default 20) is the effective batch size.

## Layout

```
recipe/            the method
  core/            generator · signature · protocols · selection · mc_dropout
                   · train_runner · evaluate
  scenarios/       transfer.py · size.py · augment.py  (thin wiring)
  config.py        one dataclass tree, fully driven by YAML
  dataset_spec.py  split-CSV access + the leakage guard
mil/               the MIL trainer (Mean / Max / ABMIL / Transformer)
configs/           example configs to copy
datasets/          the data contract (no cohort data)
scripts/           build_splits.py · run_scenario.sh
```

## Citation

See [CITATION.cff](CITATION.cff). Licensed under the MIT License.
