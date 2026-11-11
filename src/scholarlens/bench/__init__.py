"""Bench — datasets and metrics tying Lens (detection) and Quill (editing) together."""

from scholarlens.bench.metrics import auroc, evaluate, threshold_at_fpr, tpr_at_fpr
from scholarlens.bench.schema import Record, read_jsonl, write_jsonl
from scholarlens.bench.splits import assign_splits, leakage, split_of

__all__ = ["Record", "read_jsonl", "write_jsonl", "assign_splits", "leakage", "split_of", "auroc",
           "evaluate", "threshold_at_fpr", "tpr_at_fpr"]
