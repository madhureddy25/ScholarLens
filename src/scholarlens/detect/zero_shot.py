"""Zero-shot baselines (no training data needed). Scores: higher = more likely AI.

* LM statistics from one model: Fast-DetectGPT sampling discrepancy (Bao et al.,
  2024), negative log-perplexity, negative mean log-rank, negative entropy.
* Binoculars (Hans et al., 2024): ratio of a performer model's log-perplexity to
  the observer/performer cross-entropy; both models must share a tokenizer.
"""

from __future__ import annotations

import numpy as np

# feature name in Featurizer output → sign so that higher = more AI
LM_BASELINES = {
    "fast_detectgpt": ("lm_fast_detectgpt", 1.0),
    "neg_log_ppl": ("lm_log_ppl", -1.0),
    "neg_log_rank": ("lm_log_rank_mean", -1.0),
    "neg_entropy": ("lm_entropy_mean", -1.0),
}


def baseline_scores(X: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Zero-shot baselines read straight from a feature matrix (no extra model passes)."""
    out = {}
    for name, (col, sign) in LM_BASELINES.items():
        if col in names:
            out[name] = sign * X[:, names.index(col)]
    return out


class Binoculars:
    name = "binoculars"

    def __init__(self, observer: str = "gpt2", performer: str = "distilgpt2", device: str = "cpu",
                 max_length: int = 1024) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer


        self.torch, self.device, self.max_length = torch, device, max_length
        self.tok = AutoTokenizer.from_pretrained(observer)
        if AutoTokenizer.from_pretrained(performer).get_vocab() != self.tok.get_vocab():
            raise ValueError("observer and performer must share a tokenizer")
        self.observer = AutoModelForCausalLM.from_pretrained(observer).to(device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(performer).to(device).eval()
        self.model_name = f"{observer}|{performer}"

    def binoculars(self, text: str) -> float:
        torch = self.torch
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_length).input_ids
        if self.tok.bos_token_id is not None:
            ids = torch.cat([torch.tensor([[self.tok.bos_token_id]]), ids], dim=1)
        ids = ids.to(self.device)
        with torch.no_grad():
            obs = self.observer(ids).logits[0, :-1].float()
            perf = self.performer(ids).logits[0, :-1].float()
        targets = ids[0, 1:]
        perf_logp = torch.log_softmax(perf, -1)
        log_ppl = -perf_logp.gather(-1, targets[:, None]).mean()
        x_ent = -(torch.softmax(obs, -1) * perf_logp).sum(-1).mean()
        return float(log_ppl / x_ent)

    def score(self, texts: list[str]) -> np.ndarray:
        return np.array([-self.binoculars(t) for t in texts])  # low Binoculars ⇒ AI
