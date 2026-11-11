"""Surface statistics: burstiness, lexical diversity, connective and marker usage.

All functions take plain prose (ideally ``MaskResult.render()`` output, so that
LaTeX markup and citation formats do not skew the statistics).
"""

from __future__ import annotations

import math
import re

import numpy as np

from scholarlens.core.text import split_sentences, words

# Sentence-initial connectives that templated machine prose tends to over-use.
CONNECTIVES = (
    "furthermore", "moreover", "additionally", "in addition", "consequently", "overall", "notably",
    "importantly", "in conclusion", "in summary", "thus", "hence", "therefore", "however",
    "nevertheless", "ultimately", "specifically", "collectively", "taken together", "crucially",
)

# Heuristic list of words/phrases whose frequency in scientific abstracts rose
# sharply after 2023, in the spirit of excess-vocabulary analyses such as
# Kobak et al. (2024) and Liang et al. (2024). It is a weak signal: humans use
# all of these words, so it must never be used on its own to judge a text.
LLM_MARKERS = (
    "delve", "delves", "delved", "delving", "underscore", "underscores", "underscored", "underscoring",
    "showcase", "showcases", "showcased", "showcasing", "intricate", "intricacies", "pivotal",
    "meticulous", "meticulously", "realm", "tapestry", "multifaceted", "nuanced", "commendable",
    "noteworthy", "garner", "garnered", "unveil", "unveils", "unveiled", "elucidate", "elucidates",
    "leverage", "leverages", "leveraging", "harness", "harnessing", "bolster", "bolsters", "foster",
    "fosters", "fostering", "paramount", "holistic", "seamless", "seamlessly", "invaluable",
    "profound", "landscape", "navigate", "navigating", "endeavor", "endeavour", "testament",
    "encompass", "encompasses", "encompassing", "groundbreaking", "transformative", "comprehensive",
    "crucial", "notably",
)
_MARKER_PHRASES = re.compile(
    r"\b(?:shed(?:s|ding)? light on|pav(?:e|es|ing) the way|in the realm of|plays? a (?:crucial|pivotal|vital) role"
    r"|a deeper understanding|it is (?:important|worth) (?:to note|noting))\b",
    re.I,
)
FUNCTION_WORDS = frozenset(
    "a an the and or but if of in on at by for with to from as is are was were be been being this that "
    "these those it its we our they their which who whom whose not no than then there here also such "
    "can could may might will would should must do does did has have had into over under between".split()
)


def burstiness(lengths: list[int] | np.ndarray) -> float:
    """B = (σ_L − μ_L) / (σ_L + μ_L) over sentence lengths; in [−1, 1]."""
    arr = np.asarray(lengths, dtype=float)
    if arr.size == 0:
        return 0.0
    mu, sigma = arr.mean(), arr.std()
    return float((sigma - mu) / (sigma + mu)) if sigma + mu > 0 else 0.0


def mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio (length-robust lexical diversity)."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts: dict[str, int] = {}
    for t in tokens[:window]:
        counts[t] = counts.get(t, 0) + 1
    ratios = [len(counts) / window]
    for i in range(window, len(tokens)):
        out, inc = tokens[i - window], tokens[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[inc] = counts.get(inc, 0) + 1
        ratios.append(len(counts) / window)
    return float(np.mean(ratios))


def stylometric_features(text: str, sentences: list[str] | None = None) -> dict[str, float]:
    sentences = sentences if sentences is not None else split_sentences(text)
    lengths = [len(words(s)) for s in sentences]
    toks = [w.lower() for w in words(text) if not w.startswith("⟦")]
    n_words = max(len(toks), 1)
    n_sent = max(len(sentences), 1)
    arr = np.asarray(lengths or [0], dtype=float)
    counts: dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    starts = [s.lstrip("\"'“(").lower() for s in sentences]
    connective_starts = sum(any(s.startswith(c + ",") or s.startswith(c + " ") for c in CONNECTIVES) for s in starts)
    marker_hits = sum(counts.get(m, 0) for m in LLM_MARKERS) + len(_MARKER_PHRASES.findall(text))
    return {
        "n_sentences": float(len(sentences)),
        "n_words": float(len(toks)),
        "sent_len_mean": float(arr.mean()),
        "sent_len_std": float(arr.std()),
        "sent_len_cv": float(arr.std() / arr.mean()) if arr.mean() else 0.0,
        "sent_len_max": float(arr.max()),
        "sent_len_min": float(arr.min()),
        "burstiness": burstiness(lengths),
        "ttr": len(counts) / n_words,
        "mattr": mattr(toks),
        "hapax_ratio": sum(1 for c in counts.values() if c == 1) / n_words,
        "avg_word_len": float(np.mean([len(t) for t in toks])) if toks else 0.0,
        "function_word_ratio": sum(counts.get(w, 0) for w in FUNCTION_WORDS) / n_words,
        "connective_start_rate": connective_starts / n_sent,
        "llm_marker_per_1k": 1000.0 * marker_hits / n_words,
        "comma_per_sentence": text.count(",") / n_sent,
        "semicolon_per_1k": 1000.0 * text.count(";") / n_words,
        "colon_per_1k": 1000.0 * text.count(":") / n_words,
        "paren_per_1k": 1000.0 * text.count("(") / n_words,
        "dash_per_1k": 1000.0 * (text.count("—") + text.count(" – ") + text.count("---")) / n_words,
        "log_n_words": math.log1p(len(toks)),
    }
