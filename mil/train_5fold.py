from platform import architecture

from torch.utils.data import DataLoader, WeightedRandomSampler 
import torch.multiprocessing
import torch
import os
import time
import argparse as ap
import pickle
import numpy as np
from pathlib import Path

from model_train import ModelTrainer
from classifier import cAItomorph
from dataset import MILDataset
from plot_confusion import plot_confusion_matrix
from scheduler import BuildScheduler
import json
import pandas as pd
import random  # ← NEW

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# mil lives inside the recipe repo; import its shared, config-driven label
# helpers so the trainer derives the SAME label space the protocols were built with.
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from recipe.labels import canonical_label, build_label_map


def main():
    parser = ap.ArgumentParser()
    # Training parameters
    parser.add_argument('--lr', required=False, type=float, default=5e-5)
    parser.add_argument('--grad_accum', required=False, type=int, default=20,
                        help='Batches accumulated before a backward pass (batch size is 1). '
                             'This is the effective batch size.')
    parser.add_argument('--eval_fold', required=False, default='fixed',
                        choices=['fixed', 'per_fold'],
                        help="Which split column supplies val/test. 'fixed' (default) "
                             "always uses kfold0 -- correct when the folds are repeated "
                             "draws over one held-out split (transfer, size). 'per_fold' "
                             "uses kfold{f}, i.e. real cross-validation (augment).")
    parser.add_argument('--num_workers', required=False, type=int, default=4,
                        help='DataLoader worker processes. Use 0 if torch_shm_manager '
                             'fails (long TMPDIR); see the README.')
    parser.add_argument('--scheduler', required=False, default='ReduceLROnPlateau')
    parser.add_argument('--ep', required=False, type=int, default=150)
    parser.add_argument('--es', required=False, type=int, default=15)
    parser.add_argument('--wd', required=False, type=float, default=0.01)
    parser.add_argument('--metric', required=False, choices=['loss', 'f1'], type=str, default='loss')
    parser.add_argument('--seed', required=False, type=int, default=38)
    # Model architecture
    parser.add_argument('--arch', help='Filters patients with sub-standard sample quality', required=True)
    # Continue from checkpoint
    parser.add_argument('--checkpoint', required=False, default=None)
    # Eval-only recovery: skip training, load each fold's own saved best-val
    # checkpoint (cAItomorph_best_weights.pth) and write its test_conf_matrix.npy.
    # Recovers results from runs that timed out after training but before the
    # test step. No retraining is performed.
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training; evaluate each fold\'s saved best '
                             'checkpoint on the test set and write the conf matrix.')
    # Unified split CSV the val/test set + label space are read from directly
    # (no materialized val.csv / test.csv / label_map.csv).
    parser.add_argument('--splits_csv', type=str, required=True,
                        help='Target dataset wide split CSV (patient_id, label, '
                             'tensor_paths, kfold0..k).')
    parser.add_argument('--aliases_json', type=str, default="{}",
                        help='JSON dict of raw-label -> canonical-name harmonization '
                             '(from the config data.aliases). "{}" = identity.')
    parser.add_argument('--path_col', type=str, default='tensor_paths',
                        help='Absolute feature-path column name in the split CSV.')
    parser.add_argument('--label_col', type=str, default='label',
                        help='Raw label column name in the split CSV.')
    parser.add_argument('--transfer_label', type=str, required=False, default=None,
                        help='Canonical transfer/few-shot class name to include in the '
                             'label space (transfer/size scenarios). Omit for augment.')
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='Head dropout rate (>0 enables MC-dropout for the augment selector).')
    parser.add_argument('--results_root', type=str, default=None,
                        help="Root for results (<results_root>/<arch>/...). Defaults to "
                             "recipe/_runs/<saving_name_extension>.")
    # Training protocol parameters (all required)
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name folder for the protocol path')
    parser.add_argument('--k_value', type=int, required=True,
                        help='GMM k-value used for protocol generation')
    parser.add_argument('--protocol_mode', type=str, required=True,
                        help='Protocol mode used in CSV filename: '
                             'train_fold_{fold}_k_{k}_{protocol_mode}.csv')
    parser.add_argument('--training_protocols_path', type=str, required=True,
                        help='Root directory where training protocols are stored')
    # Other parameters
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                        help='List of folds to process')
    parser.add_argument('--complete_token', default='test_conf_matrix.npy',
                        help='File token to mark fold completion')
    parser.add_argument('--saving_name_extension', required=True,
                        help='Experiment name for result folder')
    parser.add_argument('--use_weighted_sampler', action='store_true',
                        help='Use weighted random sampler with inverse class frequency')
    parser.add_argument('--file_extension', type=str, required=False, choices=['h5', 'pt'],
                        default='h5', help='File extension for embeddings (h5 or pt)')
    def fold_complete(root: Path, token: str):
        return (root / token).exists()

    args = parser.parse_args()

    # ------------------ Reproducibility: minimal additions ------------------
    seed = args.seed
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def reseed():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    reseed()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    # -----------------------------------------------------------------------

    checkpoint = args.checkpoint
    start = time.time()

    # Results live under the run's output_root (recipe passes --results_root); falls
    # back to recipe/_runs/<saving_name_extension> so the script is runnable alone.
    results_root = args.results_root or os.path.join(_REPO_ROOT, "_runs", args.saving_name_extension)
    # dataset_name is part of the path, not just the protocol lookup: the size
    # sweep reuses one GMM k across every k_stats value, so without it all six
    # sweep points would land in the same folder and overwrite each other.
    base_result_path = os.path.join(results_root, args.arch, args.dataset_name)

    # Simplified result folder naming
    result_name = f"{args.arch}_k{args.k_value}_{args.protocol_mode}"
    if args.use_weighted_sampler:
        result_name += "_weighted"
    result_name += f"_lr{args.lr}_ep{args.ep}_seed{seed}"

    RESULT_FOLDER_ROOT = os.path.join(base_result_path, result_name)
    os.makedirs(RESULT_FOLDER_ROOT, exist_ok=True)

    RESULT_FOLDER_ROOT = Path(RESULT_FOLDER_ROOT)
    print('Results are saved in: ', RESULT_FOLDER_ROOT)

    print("\nInitialize datasets...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()
    print("Found device: ", ngpu, "x ", device)

    datasets = {}

    # --- Label space + val/test read straight from the unified split CSV --------
    # Canonicalize the target's labels, add the transfer class, and build the same
    # integer map (sorted -> id) the protocols were generated with. No dummy
    # label_map.csv / val.csv / test.csv are materialized anywhere.
    aliases = json.loads(args.aliases_json)
    splits = pd.read_csv(args.splits_csv)
    splits["_canon"] = [canonical_label(v, aliases) for v in splits[args.label_col]]
    label_names = set(splits["_canon"])
    if args.transfer_label:
        label_names = label_names | {args.transfer_label}
    label_to_id = build_label_map(label_names)
    class_count = len(label_to_id)
    print(f"Label space ({class_count}): {label_to_id}")

    def _eval_rows(role: str, eval_fold: int) -> pd.DataFrame:
        """val/test rows from the evaluation fold: absolute path + int label."""
        r = splits[splits[f"kfold{eval_fold}"] == role]
        return pd.DataFrame({"patient_files": r[args.path_col].values,
                             "labels": [label_to_id[c] for c in r["_canon"]]})

    for fold in args.folds:
        # Every fold starts from the same RNG state, so fold k's weights do not
        # depend on whether folds 0..k-1 ran first in the same process. Without
        # this, `--folds 3` and fold 3 of `--folds 0 1 2 3` train different
        # models -- enough on a small cohort to collapse a fold entirely.
        reseed()
        dl_gen = torch.Generator()
        dl_gen.manual_seed(seed)

        eval_fold = fold if args.eval_fold == 'per_fold' else 0
        val_files = _eval_rows("val", eval_fold)
        test_files = _eval_rows("test", eval_fold)
        eval_paths = (set(val_files["patient_files"].astype(str))
                      | set(test_files["patient_files"].astype(str)))

        RESULT_FOLDER = RESULT_FOLDER_ROOT / f'fold_{fold}'
        RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

        # Skip this fold if completion token already exists
        if fold_complete(RESULT_FOLDER, args.complete_token):
            print(f"[SKIP] fold {fold} already complete (found {args.complete_token})")
            continue

        # Eval-only recovery needs a saved best-val checkpoint for this fold.
        eval_ckpt = RESULT_FOLDER / 'cAItomorph_best_weights.pth'
        if args.eval_only and not eval_ckpt.exists():
            print(f"[SKIP] fold {fold} eval_only: no checkpoint at {eval_ckpt}")
            continue

        # Construct training protocol path (no fallback)
        train_filename = f"train_fold_{fold}_k_{args.k_value}_{args.protocol_mode}.csv"
        train_file_path = os.path.join(
            args.training_protocols_path,
            args.dataset_name,
            f"k_{args.k_value}",
            train_filename
        )

        if not os.path.exists(train_file_path):
            raise FileNotFoundError(
                f"Training protocol not found: {train_file_path}\n"
                f"Run generate_patients_and_protocols.py first to create protocols."
            )

        print(f"Loading training protocol: {train_file_path}")
        train_files = pd.read_csv(train_file_path)
        train_path_col = next((c for c in ("patient_files", "patient_name", "tensor_paths")
                               if c in train_files.columns), None)

        # ---- HARD ANTI-LEAKAGE GUARANTEE (slide-level) ------------------------
        # Synthetic patients are built from the REAL donor SLIDES in the training
        # set; no such slide may appear in val/test. The split is slide-level (a
        # patient may have other slides in eval -- that is allowed), so we compare
        # the actual feature-bag paths. Synthetic bags are new files
        # (artificial_patients/...) and are excluded. Any overlap aborts the run.
        real_train_paths = {str(p) for p in train_files[train_path_col]
                            if "artificial_patients" not in str(p)}
        leak = real_train_paths & eval_paths
        assert not leak, (f"LEAKAGE: {len(leak)} training/donor slide(s) also in "
                          f"val/test, e.g. {sorted(leak)[:3]}")

        print(f"Validation/test read from split kfold{eval_fold} (val/test)")
        datasets['train'] = MILDataset(train_files, ext=args.file_extension)
        datasets['val'] = MILDataset(val_files, ext=args.file_extension)
        datasets['test'] = MILDataset(test_files, ext=args.file_extension)
        embedding_dim = datasets['train'].get_dimension()
        print(f"Embedding dimension: {embedding_dim}")

        print("Initialize dataloaders...")
        dataloaders = {}
        num_workers = int(args.num_workers)

        label_col = 'labels' if 'labels' in train_files.columns else ('label' if 'label' in train_files.columns else None)
        if label_col is None:
            raise KeyError(f"No 'labels' or 'label' column in {train_file_path}")

        label_counts = train_files[label_col].value_counts().to_dict()

        # Setup weighted sampler if requested
        if args.use_weighted_sampler:
            # Calculate class weights (inverse frequency)
            total_samples = sum(label_counts.values())
            class_weights = {label: total_samples / count for label, count in label_counts.items()}

            # Assign weight to each sample based on its label
            sample_weights = [class_weights[label] for label in train_files[label_col]]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=dl_gen
            )
            print(f"Using weighted sampler with class weights: {class_weights}")

            dataloaders['train'] = DataLoader(
                datasets['train'],
                batch_size=1,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=False,
                worker_init_fn=seed_worker
            )
        else:
            dataloaders['train'] = DataLoader(
                datasets['train'],
                batch_size=1,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=dl_gen
            )
        dataloaders['val'] = DataLoader(
            datasets['val'], batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=False,
            worker_init_fn=seed_worker, generator=dl_gen
        )
        dataloaders['test'] = DataLoader(
            datasets['test'], batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=False,
            worker_init_fn=seed_worker, generator=dl_gen
        )
        print("Dataloaders are ready..")

        if torch.cuda.is_available():
            print(f"GPU memory after dataloaders: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        model = cAItomorph(
            class_count=class_count,
            arch=args.arch,
            embedding_dim=embedding_dim,
            dropout_p=args.dropout_p)

        if torch.cuda.is_available():
            print(f"GPU memory after model creation (CPU): {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # In eval_only mode load this fold's own saved best checkpoint; otherwise
        # honour the optional global --checkpoint warm-start.
        load_from = str(eval_ckpt) if args.eval_only else checkpoint
        if load_from is not None:
            pre = torch.load(load_from)
            vit_state_dict = {k.replace("module.", ""): v for k, v in pre.items()}
            model.load_state_dict(vit_state_dict, strict=True)
            print(f"Using weights from {load_from}")

        if (ngpu > 1):
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        print(model.eval())
        print("Setup complete.\n")

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=args.wd)
        sched_builder = BuildScheduler(optimizer, args)
        print("Using scheduler:", sched_builder.name)

        train_obj = ModelTrainer(
            model=model,
            dataloaders=dataloaders,
            epochs=int(args.ep),
            optimizer=optimizer,
            sched_builder=sched_builder,
            class_count=class_count,
            device=device,
            early_stop=int(args.es),
            grad_accum=int(args.grad_accum),
            save_path=RESULT_FOLDER)
        if args.eval_only:
            print("Eval-only: evaluating saved checkpoint on the test set (no training)")
            model, conf_matrix = train_obj.launch_eval_only()
        else:
            print("Starting training")
            model, conf_matrix = train_obj.launch_training()

        np.save(RESULT_FOLDER / 'test_conf_matrix.npy', conf_matrix)
        plot_confusion_matrix(conf_matrix, RESULT_FOLDER, {i: n for n, i in label_to_id.items()})

        end = time.time()
        runtime = end - start
        time_str = str(int(runtime // 3600)) + "h" + str(int((runtime % 3600) // 60)) + "min" + str(int(runtime % 60)) + "s"

        print("")
        print("------------------------Final report--------------------------")
        print('Runtime', time_str)
        print('Fold', fold)
        print('Architecture', args.arch)
        print('Splits CSV', args.splits_csv)
        print('Result folder', RESULT_FOLDER)
        print('Scheduler', args.scheduler)
        print('Weight decay', args.wd)
        print('Early stopping', args.es)
        print('Gradient accumulation', args.grad_accum)
        print('Seed', seed)
        print('max. Epochs', args.ep)
        print('Learning rate', args.lr)


if __name__ == "__main__":
    main()
