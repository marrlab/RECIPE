"""Config-driven label handling -- no dataset names baked in.

A dataset's label column may already be canonical (the common case), or it may
need light harmonization (e.g. a control class named differently in two cohorts
that are transferred between each other). Both are expressed purely as data: an
optional ``aliases`` mapping (raw label -> canonical name) carried on the
dataset's config. With no aliases, ``canonical_label`` is the identity, so the
code stays dataset-agnostic.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional


def canonical_label(raw: object, aliases: Optional[Mapping[str, str]] = None) -> str:
    """Map a raw label to its canonical name through an ``aliases`` table.

    ``aliases`` is the (raw -> canonical) dict from the dataset's config; it is a
    *partial* remap -- only the listed labels are renamed (e.g. to align a control
    class named differently in two cohorts), and any label not in the table passes
    through unchanged. Empty/None aliases means every label is already canonical.
    """
    key = str(raw)
    if not aliases:
        return key
    return aliases.get(key, key)


def build_label_map(canonical_labels: Iterable[str]) -> Dict[str, int]:
    """Stable canonical-name -> integer id over a deduplicated, sorted set.

    Callers pass the union of the target's present labels and the transfer label,
    so a source and a target agree on the integers (both go through this map).
    """
    names: List[str] = sorted(set(canonical_labels))
    return {name: i for i, name in enumerate(names)}
