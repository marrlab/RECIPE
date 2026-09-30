"""Experiment configuration -- one dataclass tree, fully driven by YAML.

The config carries no dataset identity beyond free-text names and paths supplied
by the user. ``data`` describes the target cohort; ``source`` (transfer scenario
only) describes the donor cohort. The three scenarios reuse the same
clustering/generator/protocol/train blocks and add one small scenario block each
(``size`` / ``augment``).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Repo root (recipe/) derived from this file -> recipe/recipe/config.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIL_DIR = os.path.join(REPO_ROOT, "mil")

SCENARIOS = ("transfer", "size", "augment")


def _resolve(path: str) -> str:
    """Make a config path absolute.

    Training runs as a subprocess with ``cwd=mil/``, so a relative ``splits_csv``
    would be resolved against the wrong directory. Relative paths are taken
    against the current working directory, falling back to the repository root
    (what ``datasets/<cohort>_splits.csv`` in the shipped configs means).
    """
    if os.path.isabs(path):
        return path
    from_cwd = os.path.abspath(path)
    if os.path.exists(from_cwd):
        return from_cwd
    from_root = os.path.join(REPO_ROOT, path)
    return from_root if os.path.exists(from_root) else from_cwd


@dataclass
class DataConfig:
    """One wide split CSV describing a cohort. Absolute feature path per patient,
    a string label, and a ``train/val/test`` role per fold (``kfold0..k{n-1}``).
    No dataset is special-cased: every field is a value from the YAML."""
    name: str = "target"
    splits_csv: str = ""
    file_extension: str = "h5"
    id_col: str = "patient_id"
    path_col: str = "tensor_paths"
    label_col: str = "label"
    n_fold_cols: int = 5
    out_patient_col: str = "patient_files"  # column name written into protocols
    # Optional raw-label -> canonical-name harmonization (e.g. {"SCD": "healthy"}).
    # Empty = labels are already canonical.
    aliases: Dict[str, str] = field(default_factory=dict)


@dataclass
class ClusteringConfig:
    n_clusters: int = 50
    algorithm: str = "gaussian_mixture"   # vector-quantizer backend (clustering.py)
    random_state: int = 42
    verbose: int = 1
    covariance_type: str = "diag"
    init_params: str = "kmeans"
    params: Dict[str, Any] = field(default_factory=dict)
    fit_sample_size: Optional[int] = None
    donor_cells_per_patient: Optional[int] = None


@dataclass
class GeneratorConfig:
    """The single artificial-patient generator: per-cluster Gaussian count
    sampling + within-cluster cell resampling (core/generator.py)."""
    n_artificial: int = 50
    seed: int = 42
    max_bag_size: Optional[int] = None


@dataclass
class ProtocolConfig:
    seed: int = 42
    n_folds: int = 5
    k_value: int = 50
    baseline_per_fold_cv: bool = False
    injection_base_fold: int = 0
    # How many transfer-class patients to inject (direct/synthetic). None ->
    # match the target's own count of the transfer class.
    transfer_inject_count: Optional[int] = None
    # Dataset-name folder used in protocol paths (per scenario default below).
    dataset_name: str = "protocols"


@dataclass
class TrainConfig:
    architectures: List[str] = field(default_factory=lambda: ["Transformer", "ABMIL", "Mean"])
    lr: float = 5e-5
    epochs: int = 150
    seed: int = 38
    folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_workers: int = 4       # DataLoader workers; 0 if torch_shm_manager fails
    train_script: str = os.path.join(MIL_DIR, "train_5fold.py")
    mil_dir: str = MIL_DIR

    # --- site-specific: everything below only shapes the emitted SLURM script.
    # The defaults are the Helmholtz Munich cluster's; override them in YAML.
    conda_env: str = "recipe"
    slurm_partition: str = "gpu_p"
    slurm_qos: str = "gpu_normal"
    slurm_gres: str = "gpu:1"
    slurm_time: str = "12:00:00"
    slurm_cpus: int = 8
    slurm_mem: str = "64G"


@dataclass
class SizeConfig:
    """Few-shot scenario: keep k real patients of the transfer class, fill the
    rest with artificials. The signature comes from those k patients."""
    k_stats_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])


@dataclass
class AugmentConfig:
    """Augment scenario: generate artificials per class, score with MC dropout,
    keep the top-X% by an uncertainty metric and selection strategy."""
    class_count: Optional[int] = None
    artificial_patient_count: int = 23
    mc_dropout: int = 20
    dropout_p: float = 0.2
    fit_ep: int = 150          # selector training epochs (named in result paths)
    metrics: List[str] = field(default_factory=lambda: ["epistemic_entropy", "bald_score", "uncertainty_std"])
    strategies: List[str] = field(default_factory=lambda: ["global", "mixed"])
    percentages: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30, 40])
    uncertainty_mode: str = "highest"   # "highest" | "lowest"


@dataclass
class ExperimentConfig:
    scenario: str = "transfer"
    data: DataConfig = field(default_factory=DataConfig)
    source: Optional[DataConfig] = None          # transfer scenario only
    transfer_label: Optional[str] = None         # transfer/size: the class in play
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    size: SizeConfig = field(default_factory=SizeConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    output_root_override: Optional[str] = None

    # ---- run identity & layout ----------------------------------------
    @property
    def run_name(self) -> str:
        if self.scenario == "transfer":
            src = self.source.name if self.source else "source"
            return f"transfer_S-{src}_T-{self.data.name}_C-{self.transfer_label}"
        return f"{self.scenario}_{self.data.name}"

    @property
    def output_root(self) -> str:
        return self.output_root_override or os.path.join(REPO_ROOT, "_runs", self.run_name)

    @property
    def saving_name_extension(self) -> str:
        return self.run_name

    # ---- derived paths -------------------------------------------------
    @property
    def codebook_path(self) -> str:
        return os.path.join(self.output_root, f"codebook_k{self.clustering.n_clusters}.pkl")

    @property
    def signature_stats_csv(self) -> str:
        return os.path.join(self.output_root, "transfer_cluster_stats.csv")

    @property
    def signature_counts_csv(self) -> str:
        return os.path.join(self.output_root, "transfer_cluster_counts.csv")

    @property
    def artificial_patients_dir(self) -> str:
        return os.path.join(self.output_root, "artificial_patients")

    @property
    def protocols_root(self) -> str:
        return os.path.join(self.output_root, "training_protocols")

    # ---- (de)serialization --------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "ExperimentConfig":
        cfg = cls()
        if not path:
            return cfg
        if yaml is None:
            raise RuntimeError("pyyaml is not installed but a YAML config was requested")
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls._merge(cfg, data)

    @staticmethod
    def _merge(cfg: "ExperimentConfig", data: Dict[str, Any]) -> "ExperimentConfig":
        # `source` is an optional nested DataConfig; instantiate before merging.
        if "source" in data and data["source"] is not None and cfg.source is None:
            cfg.source = DataConfig()
        for key, val in data.items():
            if isinstance(getattr(type(cfg), key, None), property):
                # Derived paths (output_root, run_name, ...) are computed, not settable.
                # Silently ignoring them here would make the whole block a no-op.
                raise KeyError(
                    f"'{key}' is a derived property and cannot be set from YAML. "
                    f"Use the --output-root flag to relocate a run.")
            if not hasattr(cfg, key):
                raise KeyError(f"Unknown config key: {key}")
            cur = getattr(cfg, key)
            if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    if not hasattr(cur, sub_k):
                        raise KeyError(f"Unknown config key: {key}.{sub_k}")
                    setattr(cur, sub_k, sub_v)
            else:
                setattr(cfg, key, val)
        if cfg.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got '{cfg.scenario}'")
        for block in (cfg.data, cfg.source):
            if block is not None and block.splits_csv:
                block.splits_csv = _resolve(block.splits_csv)
        return cfg
