from __future__ import annotations

import re
from typing import List

HANGUL_RE = re.compile(r"[가-힣]")
KOREAN_CHAR_RE = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]")

# Tokens for BLEU normalization across mixed KO/EN, numbers, punctuation.
TOK_RE = re.compile(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s]")

# Simple mixed-language tokens for density denominator.
MIXED_TOKEN_RE = re.compile(r"\w+|[가-힣]+")

# Sentence boundary:
#  1) period after alphabetic or hangul (not numeric)
#  2) blank-line (double newline)
SENT_BOUNDARY_RE = re.compile(r"(?<=[A-Za-z가-힣])\.|(?:\r?\n)\s*(?:\r?\n)")


def has_hangul(s: str) -> bool:
    return bool(s) and bool(HANGUL_RE.search(s))


def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = SENT_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_for_bleu(text: str) -> str:
    tokens = TOK_RE.findall(str(text))
    return " ".join(tokens)


def count_mixed_tokens(text: str) -> int:
    if not text:
        return 0
    return len([t for t in MIXED_TOKEN_RE.findall(text) if t.strip()])


def korean_char_proportion(text: str) -> float:
    if not isinstance(text, str) or len(text) == 0:
        return float("nan")
    korean_chars = KOREAN_CHAR_RE.findall(text)
    return len(korean_chars) / len(text)
