from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy import stats


_WORD_RE = re.compile(r"\b\w+\b")


@dataclass(frozen=True)
class ZipfResult:
    slope: float
    intercept: float
    r2: float
    p_value: float
    std_err: float
    word_counts: Counter


def zipf_fit(texts: pd.Series) -> ZipfResult:
    """
    Fit Zipf's law on a corpus (log-log rank vs frequency).
    Returns slope, intercept, R^2, and word counts.
    """
    all_text = " ".join(texts.dropna().astype(str)).lower()
    words = _WORD_RE.findall(all_text)

    word_counts = Counter(words)
    freqs = np.array(sorted(word_counts.values(), reverse=True), dtype=float)
    if len(freqs) == 0:
        return ZipfResult(
            slope=float("nan"),
            intercept=float("nan"),
            r2=float("nan"),
            p_value=float("nan"),
            std_err=float("nan"),
            word_counts=word_counts,
        )

    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
    return ZipfResult(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r_value**2),
        p_value=float(p_value),
        std_err=float(std_err),
        word_counts=word_counts,
    )


def build_whitespace_token_probs(texts: pd.Series) -> Dict[str, float]:
    corpus = " ".join(texts.dropna().astype(str)).strip()
    if not corpus:
        return {}

    tokens = corpus.split()
    counts = Counter(tokens)
    total = sum(counts.values())
    return {tok: cnt / total for tok, cnt in counts.items()}


def build_tiktoken_id_probs(texts: pd.Series, encoder) -> Dict[int, float]:
    token_ids: list[int] = []
    for txt in texts.dropna().astype(str).tolist():
        token_ids.extend(encoder.encode(txt))

    if not token_ids:
        return {}

    counts = Counter(token_ids)
    total = sum(counts.values())
    return {tid: cnt / total for tid, cnt in counts.items()}


def js_distance_from_prob_dicts(p: Dict, q: Dict, base: float = 2.0) -> float:
    """
    SciPy jensenshannon returns a *distance* (sqrt(JS divergence)).
    """
    support = sorted(set(p.keys()) | set(q.keys()))
    if not support:
        return float("nan")

    p_vec = np.array([p.get(k, 0.0) for k in support], dtype=float)
    q_vec = np.array([q.get(k, 0.0) for k in support], dtype=float)
    return float(jensenshannon(p_vec, q_vec, base=base))
