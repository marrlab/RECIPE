"""recipe -- Artificial-Patient GENeration for MIL cohorts.

A single, dataset-agnostic framework for three scenarios that all share
the same primitives (GMM codebook -> per-cluster count signature -> Gaussian
cell-sampling generator -> training-protocol CSVs -> MIL train + balanced
accuracy):

    transfer  -- supply a missing class by transferring a source cohort's
                 cluster-composition signature onto the target's donor cells.
    size      -- fill a few-shot class back to full size from the signature of
                 the k real patients that remain.
    augment   -- generate artificial patients per class, then keep the top-X%
                 by MC-dropout uncertainty.

No dataset names are hard-coded anywhere: a dataset is described entirely by the
``data:`` block of a YAML config (see ``configs/``).
"""
from .config import ExperimentConfig

__all__ = ["ExperimentConfig"]
__version__ = "1.0.0"
