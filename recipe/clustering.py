"""The clustering codebook: a Gaussian Mixture Model used as a vector quantizer.

The pipeline only ever needs a *hard* cluster label per cell (``predict``); the
GMM's covariances, responsibilities and likelihood are computed and thrown away.
The codebook is fit once (on the source cohort for `transfer`, on the fold's own
training pool for `size`/`augment`) and then assigns every cell of the donor pool.

The default is a diagonal-covariance GMM with K=50 and k-means initialisation
(set `clustering.n_clusters` and friends in the config). That is the only
algorithm shipped; the returned object exposes ``fit(X)`` /
``predict(X)`` and is picklable, so ``save_codebook``/``load_codebook`` work on it.
"""
from __future__ import annotations

ALGORITHMS = ("gaussian_mixture",)


def available_algorithms() -> list:
    return list(ALGORITHMS)


def build_codebook(gcfg):
    """Construct an unfitted codebook from a :class:`ClusteringConfig`.

    ``gcfg.params`` overrides any GaussianMixture keyword argument.
    """
    if gcfg.algorithm not in ALGORITHMS:
        raise KeyError(
            f"Unknown algorithm '{gcfg.algorithm}'. Available: {available_algorithms()}")

    from sklearn.mixture import GaussianMixture

    params = dict(getattr(gcfg, "params", {}) or {})
    params.setdefault("n_components", gcfg.n_clusters)
    params.setdefault("covariance_type", gcfg.covariance_type)
    params.setdefault("init_params", gcfg.init_params)
    params.setdefault("random_state", gcfg.random_state)
    params.setdefault("verbose", gcfg.verbose)
    return GaussianMixture(**params)
