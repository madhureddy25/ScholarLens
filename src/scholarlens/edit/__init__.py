"""Quill — citation-safe language editing of an author's own manuscript."""

from scholarlens.edit.engine import EditConfig, EditSession, NodeEdit, Quill, Status
from scholarlens.edit.report import write_outputs
from scholarlens.edit.similarity import EmbeddingSimilarity, sentence_transformer_similarity

__all__ = ["EditConfig", "EditSession", "NodeEdit", "Quill", "Status", "write_outputs",
           "EmbeddingSimilarity", "sentence_transformer_similarity"]
