"""Unified command-line interface.

    python -m recipe <scenario> <stage> -c config.yaml [options]

scenario : transfer | size | augment
stage    : run | generate | protocols | fit-codebook | select |
           train-selector | train | evaluate

Examples
--------
    python -m recipe transfer run      -c configs/transfer.yaml
    python -m recipe transfer train    -c configs/transfer.yaml --emit-slurm scripts/run_transfer.sh
    python -m recipe size     run      -c configs/size.yaml
    python -m recipe augment  run      -c my_augment.yaml     # generate + selector + score + protocols
    python -m recipe augment  train    -c my_augment.yaml     # final training grid
    python -m recipe <scenario> evaluate -c config.yaml
"""
from __future__ import annotations

import argparse
import os

from .config import SCENARIOS, ExperimentConfig
from .core import evaluate as ev
from .core import train_runner as tr
from .scenarios import REGISTRY

STAGES = ["run", "generate", "protocols", "fit-codebook", "select",
          "train-selector", "train", "evaluate"]


def _load_cfg(args) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg.scenario = args.scenario           # positional overrides the file
    if args.output_root:
        cfg.output_root_override = args.output_root
    return cfg


def _require(mod, attr, scenario, stage):
    if not hasattr(mod, attr):
        raise SystemExit(f"stage '{stage}' is not available for scenario '{scenario}'")
    return getattr(mod, attr)


def _train(cfg: ExperimentConfig, mod, args) -> int:
    jobs = mod.build_jobs(cfg) if hasattr(mod, "build_jobs") else tr.all_jobs(cfg, mod.arms(cfg))
    if args.emit_slurm:
        path = tr.emit_jobs_slurm(cfg, jobs, args.emit_slurm)
        print(f"SLURM array script written -> {path} ({len(jobs)} jobs)")
        return 0
    rc = 0
    for job in jobs:
        rc |= tr.run_job(job, cfg.train.mil_dir, dry_run=args.dry_run)
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="recipe", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", choices=SCENARIOS)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("-c", "--config", default=None, help="YAML config path")
    p.add_argument("--output-root", default=None, help="Override output_root")
    p.add_argument("--dry-run", action="store_true", help="print training commands only")
    p.add_argument("--emit-slurm", default=None, metavar="PATH", help="write a SLURM array script")
    args = p.parse_args(argv)

    cfg = _load_cfg(args)
    mod = REGISTRY[args.scenario]
    stage = args.stage

    if stage == "run":
        if args.scenario == "augment":
            mod.run(cfg, dry_run=args.dry_run)
        else:
            mod.run(cfg)
    elif stage == "generate":
        mod.generate(cfg)
    elif stage == "protocols":
        _require(mod, "protocols", args.scenario, stage)(cfg)
    elif stage == "fit-codebook":
        _require(mod, "fit_codebook", args.scenario, stage)(cfg)
    elif stage == "train-selector":
        _require(mod, "train_selector", args.scenario, stage)(cfg, dry_run=args.dry_run)
    elif stage == "select":
        _require(mod, "score", args.scenario, stage)(cfg)
        _require(mod, "build_protocols", args.scenario, stage)(cfg)
    elif stage == "train":
        return _train(cfg, mod, args)
    elif stage == "evaluate":
        df = ev.collect_all(cfg)
        if df.empty:
            print("No results found yet (train first).")
            return 1
        out = os.path.join(cfg.output_root, "balanced_accuracy_summary.csv")
        os.makedirs(cfg.output_root, exist_ok=True)
        df.drop(columns=["per_fold"]).to_csv(out, index=False)
        print(df.drop(columns=["per_fold"]).to_string(index=False))
        print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
