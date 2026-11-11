"""Phase 2 — citation, reference, math, quantity and term protection."""

from scholarlens.masking.kinds import Kind
from scholarlens.masking.masker import (
    SENTINEL_RE,
    IntegrityError,
    IntegrityReport,
    MaskConfig,
    Masker,
    MaskResult,
    Span,
)
from scholarlens.masking.terms import SpacyTermDetector

__all__ = ["Kind", "IntegrityError", "IntegrityReport", "MaskConfig", "Masker", "MaskResult",
           "SENTINEL_RE", "Span", "SpacyTermDetector"]
