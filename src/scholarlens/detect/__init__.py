"""Lens — AI-text detection for academic prose, built and evaluated for robustness."""

from scholarlens.detect.featurize import Featurizer, normalize
from scholarlens.detect.fusion import FusionDetector
from scholarlens.detect.retrieval import RetrievalDetector
from scholarlens.detect.zero_shot import Binoculars, baseline_scores

__all__ = ["Featurizer", "normalize", "FusionDetector", "RetrievalDetector", "Binoculars", "baseline_scores"]


def load_featurizer_from_signature(signature: str, cache=None) -> Featurizer:
    """Rebuild the featurizer a model was trained with ("lm=gpt2|emb=<model>")."""
    parts = dict(p.split("=", 1) for p in signature.split("|"))
    lm = emb = None
    if parts.get("lm", "none") != "none":
        from scholarlens.features.lm import CausalLMScorer

        lm = CausalLMScorer(parts["lm"])
    if parts.get("emb") and parts["emb"] not in ("HashingEmbedder",):
        from scholarlens.features.transitions import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder(parts["emb"])
    return Featurizer(lm=lm, embedder=emb, cache=cache)
