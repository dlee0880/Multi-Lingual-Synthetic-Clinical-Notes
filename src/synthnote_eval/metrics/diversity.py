from __future__ import annotations

from typing import List

import sacrebleu

from synthnote_eval.utils.text import normalize_for_bleu


def sentence_self_bleu(hypothesis: str, references: List[str]) -> float:
    """
    Self-BLEU for diversity: BLEU(hypothesis, references).
    Uses deterministic normalization and sacrebleu with tokenize="none".
    """
    hyp = normalize_for_bleu(hypothesis)
    refs = [normalize_for_bleu(r) for r in references]
    return float(sacrebleu.sentence_bleu(hyp, refs, tokenize="none").score)
