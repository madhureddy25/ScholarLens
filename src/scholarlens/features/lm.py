"""Language-model signals: perplexity, entropy dynamics, rank statistics and the
Fast-DetectGPT sampling-discrepancy score.

All are computed from a *single* forward pass per window:

    PPL(X)  = exp(−1/N Σ log P(x_i | x_<i))
    H_i     = −Σ_v P(v | x_<i) log P(v | x_<i)
    FDG(X)  = (Σ log P(x_i) − Σ μ_i) / sqrt(Σ σ²_i)

where μ_i = Σ_v P(v) log P(v) and σ²_i = Σ_v P(v) (log P(v))² − μ_i² are the
analytic mean/variance of a token's log-probability under the model's own
distribution (Bao et al., 2024, "Fast-DetectGPT", ICLR). Higher FDG means the
text is more "typical" for the model than its own samples would be.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class LMStats:
    n_tokens: int
    log_ppl: float
    ppl: float
    logprob_mean: float
    logprob_std: float
    entropy_mean: float
    entropy_std: float
    entropy_window_var: float
    log_rank_mean: float
    top1_frac: float
    top10_frac: float
    fast_detectgpt: float

    def to_dict(self, prefix: str = "lm_") -> dict[str, float]:
        return {prefix + k: float(v) for k, v in asdict(self).items()}


class CausalLMScorer:
    def __init__(self, model_name: str = "gpt2", device: str = "cpu", max_length: int = 1024,
                 stride: int = 512, entropy_window: int = 16) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer


        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.max_length = min(max_length, getattr(self.model.config, "max_position_embeddings", max_length))
        self.stride = stride
        self.entropy_window = entropy_window

    def token_arrays(self, text: str) -> dict[str, np.ndarray]:
        """Per-token arrays: logprob, entropy, rank, analytic mean and variance."""
        torch = self.torch
        ids = self.tok(text, return_tensors="pt").input_ids[0]
        if self.tok.bos_token_id is not None and (len(ids) == 0 or ids[0] != self.tok.bos_token_id):
            ids = torch.cat([torch.tensor([self.tok.bos_token_id]), ids])
        cols: dict[str, list] = {k: [] for k in ("logprob", "entropy", "rank", "mu", "var")}
        scored_upto = 0  # index of the last target token already scored
        start = 0
        with torch.no_grad():
            while True:
                end = min(start + self.max_length, len(ids))
                chunk = ids[start:end].unsqueeze(0).to(self.device)
                logits = self.model(chunk).logits[0, :-1].float()
                targets = chunk[0, 1:]
                logp = torch.log_softmax(logits, dim=-1)
                p = logp.exp()
                tok_lp = logp.gather(-1, targets[:, None]).squeeze(-1)
                mu = (p * logp).sum(-1)
                var = (p * logp.square()).sum(-1) - mu.square()
                rank = (logp > tok_lp[:, None]).sum(-1) + 1
                first_new = max(0, scored_upto - start)  # skip targets already scored
                for name, arr in (("logprob", tok_lp), ("entropy", -mu), ("rank", rank), ("mu", mu), ("var", var)):
                    cols[name].append(arr[first_new:].cpu().numpy())
                scored_upto = end - 1
                if end >= len(ids):
                    break
                start = end - (self.max_length - self.stride)
        return {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}

    def score(self, text: str) -> LMStats:
        a = self.token_arrays(text)
        n = len(a["logprob"])
        if n == 0:
            return LMStats(0, *([0.0] * 11))
        lp, ent = a["logprob"], a["entropy"]
        w = self.entropy_window
        win = np.convolve(ent, np.ones(w) / w, mode="valid") if n >= w else ent
        fdg = (lp.sum() - a["mu"].sum()) / np.sqrt(max(a["var"].sum(), 1e-12))
        return LMStats(
            n_tokens=n,
            log_ppl=float(-lp.mean()),
            ppl=float(np.exp(-lp.mean())),
            logprob_mean=float(lp.mean()),
            logprob_std=float(lp.std()),
            entropy_mean=float(ent.mean()),
            entropy_std=float(ent.std()),
            entropy_window_var=float(np.var(win)),
            log_rank_mean=float(np.log(a["rank"]).mean()),
            top1_frac=float((a["rank"] == 1).mean()),
            top10_frac=float((a["rank"] <= 10).mean()),
            fast_detectgpt=float(fdg),
        )
