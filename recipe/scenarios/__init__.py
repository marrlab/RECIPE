"""Thin per-scenario wiring over the shared core primitives."""
from . import augment, size, transfer

REGISTRY = {"transfer": transfer, "size": size, "augment": augment}
