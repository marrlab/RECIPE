"""Drive the shared mil/train_5fold.py -- one job per (arch, arm).

Training is never re-implemented: scenarios describe their arms as
(label, dataset_name, protocol_mode, extra_args) and this builds/runs/emits the
exact train_5fold.py command. Data-agnostic: the target split CSV and its label
aliases are passed straight through from the config.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import ExperimentConfig


@dataclass
class Arm:
    label: str                      # human label (for logs)
    dataset_name: str               # protocol folder
    protocol_mode: str              # protocol filename mode token
    extra_args: List[str] = field(default_factory=list)


@dataclass
class TrainJob:
    arch: str
    arm: str
    cmd: List[str]


def build_command(cfg: ExperimentConfig, arch: str, arm: Arm) -> TrainJob:
    tc = cfg.train
    cmd = [
        "python", tc.train_script,
        "--arch", arch,
        "--splits_csv", cfg.data.splits_csv,
        "--aliases_json", json.dumps(cfg.data.aliases or {}),
        "--path_col", cfg.data.path_col,
        "--label_col", cfg.data.label_col,
        "--dataset_name", arm.dataset_name,
        "--k_value", str(cfg.protocol.k_value),
        "--protocol_mode", arm.protocol_mode,
        "--training_protocols_path", cfg.protocols_root,
        "--results_root", cfg.output_root,
        "--folds", *[str(f) for f in tc.folds],
        "--saving_name_extension", cfg.saving_name_extension,
        "--file_extension", cfg.data.file_extension,
        "--lr", str(tc.lr),
        "--ep", str(tc.epochs),
        "--seed", str(tc.seed),
        "--num_workers", str(tc.num_workers),
    ]
    if cfg.transfer_label:
        cmd += ["--transfer_label", cfg.transfer_label]
    cmd += list(arm.extra_args)
    return TrainJob(arch=arch, arm=arm.label, cmd=cmd)


def all_jobs(cfg: ExperimentConfig, arms: List[Arm]) -> List[TrainJob]:
    return [build_command(cfg, arch, arm) for arm in arms for arch in cfg.train.architectures]


def run_job(job: TrainJob, mil_dir: str, dry_run: bool = False) -> int:
    print(f"[recipe] ({job.arch}/{job.arm}) cd {mil_dir} && {' '.join(job.cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.call(job.cmd, cwd=mil_dir)


def emit_jobs_slurm(cfg: ExperimentConfig, jobs: List[TrainJob], path: str,
                    job_name: str = "recipe") -> str:
    tc = cfg.train
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=logs_recipe/recipe_%A_%a.out",
        "#SBATCH --error=logs_recipe/recipe_%A_%a.err",
        f"#SBATCH --array=0-{len(jobs) - 1}",
        f"#SBATCH --partition={tc.slurm_partition}",
        f"#SBATCH --qos={tc.slurm_qos}",
        f"#SBATCH --gres={tc.slurm_gres}",
        f"#SBATCH --time={tc.slurm_time}",
        f"#SBATCH --cpus-per-task={tc.slurm_cpus}",
        f"#SBATCH --mem={tc.slurm_mem}",
        "",
        "source ~/.bashrc",
        # RECIPE_ENV lets a site point at a differently-named environment
        # without editing the config.
        f'conda activate "${{RECIPE_ENV:-{tc.conda_env}}}"',
        "mkdir -p logs_recipe",
        "",
        f"cd {tc.mil_dir}",
        "",
        "case $SLURM_ARRAY_TASK_ID in",
    ]
    for i, job in enumerate(jobs):
        lines.append(f"  {i})")
        lines.append(f"    # {job.arch} / {job.arm}")
        lines.append("    " + " ".join(shlex.quote(c) for c in job.cmd) + " ;;")
    lines += ["esac", ""]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # SLURM opens the log files before the script body runs, so the directory has
    # to exist by submit time -- not just after the `mkdir -p` inside the script.
    os.makedirs(os.path.join(os.getcwd(), "logs_recipe"), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    os.chmod(path, 0o755)
    return path
