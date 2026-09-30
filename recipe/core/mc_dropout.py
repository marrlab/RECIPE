"""Monte-Carlo dropout uncertainty scoring of artificial patients.

No separate "dropout" model fork: ABMIL, Mean and Max expose a ``dropout_p``
that puts a Dropout layer in their classifier head. Transformer takes no
``dropout_p`` -- it is already stochastic through its built-in ``emb_dropout``
(0.1) and attention dropout, so the effective rate differs by architecture.
``enable_mc_dropout`` flips every Dropout module to train mode at inference so
repeated forward passes are stochastic. From the per-pass softmax
predictions we compute three uncertainty metrics:

* uncertainty_std    -- max over classes of the per-class std across MC passes.
* epistemic_entropy  -- entropy of the mean prediction.
* bald_score         -- mutual information = epistemic_entropy - mean(per-pass entropy).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _entropy_np(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def metrics_from_predictions(preds: np.ndarray) -> dict:
    """The three uncertainty metrics from a [T, C] stack of MC softmax predictions.

    * uncertainty_std   = max over classes of the per-class std across MC passes.
    * epistemic_entropy = entropy of the mean prediction.
    * bald_score        = epistemic_entropy - mean(per-pass entropy)  (>= 0).
    """
    preds = np.asarray(preds, dtype=float)
    mean_pred = preds.mean(axis=0)
    epistemic_entropy = float(_entropy_np(mean_pred))
    bald_score = float(epistemic_entropy - np.mean(_entropy_np(preds)))
    return {
        "uncertainty_std": float(preds.std(axis=0).max()),
        "epistemic_entropy": epistemic_entropy,
        "bald_score": bald_score,
    }


def enable_mc_dropout(model) -> None:
    """Set every Dropout module to train mode (rest of the net stays in eval)."""
    import torch.nn as nn
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()


def load_model(weights_path: str, arch: str, class_count: int, embedding_dim: int,
               dropout_p: float = 0.2, device=None):
    """Load a cAItomorph checkpoint with dropout enabled in the head."""
    import torch
    from classifier import cAItomorph  # resolved via mil on sys.path
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = cAItomorph(class_count=class_count, arch=arch,
                       embedding_dim=embedding_dim, dropout_p=dropout_p)
    state = torch.load(weights_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device)


def mc_dropout_scores(model, patients: Sequence[Tuple[str, int]], mc_iterations: int,
                      device=None) -> pd.DataFrame:
    """Score each (feature_path, label) by the three uncertainty metrics.

    Returns a DataFrame [patient_id, label, uncertainty_std, epistemic_entropy,
    bald_score]; ``patient_id`` is the bag filename stem (matches the records the
    protocol builder writes)."""
    import os
    import torch
    import torch.nn.functional as F
    from ..data_io import load_features

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    enable_mc_dropout(model)

    rows = []
    with torch.no_grad():
        for path, label in patients:
            feat = load_features(path)
            bag = torch.as_tensor(np.asarray(feat), dtype=torch.float32, device=device).unsqueeze(0)
            preds = []
            for _ in range(mc_iterations):
                logits = model(bag)
                preds.append(F.softmax(logits, dim=1).cpu().numpy()[0])
            m = metrics_from_predictions(np.stack(preds))   # [T, C]
            rows.append({"patient_id": os.path.basename(path).split(".")[0],
                         "label": int(label), **m})
    return pd.DataFrame(rows)
