"""ScholarLens: integrity-first tooling for academic text.

* ``scholarlens.ingest``   – Phase 1: LaTeX / Markdown / DOCX / PDF(GROBID) → Document
* ``scholarlens.masking``  – Phase 2: citation, math, reference, quantity and term protection
* ``scholarlens.features`` – statistical + structural signals (PPL, entropy, burstiness, transitions)
* ``scholarlens.detect``   – paraphrase-robust AI-text detection (Lens)
* ``scholarlens.edit``     – citation-safe editing assistant (Quill)
* ``scholarlens.bench``    – robustness and fairness benchmarks tying both together
"""

__version__ = "0.1.0"

# Hugging Face download defaults must be in the environment before huggingface_hub
# is first imported, so they are applied when the package itself is imported.
from scholarlens import _hub  # noqa: E402, F401
