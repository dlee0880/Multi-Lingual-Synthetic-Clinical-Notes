from __future__ import annotations

import re
from typing import Sequence

import pandas as pd
import tiktoken

from synthnote_eval.utils.text import split_sentences


def token_length_tiktoken(texts: pd.Series, encoding_name: str = "cl100k_base") -> pd.Series:
    enc = tiktoken.get_encoding(encoding_name)
    return texts.fillna("").astype(str).apply(lambda x: len(enc.encode(x)))


def sentence_count(texts: pd.Series) -> pd.Series:
    return texts.fillna("").astype(str).apply(lambda x: len(split_sentences(x)))


def extended_special_char_count(texts: pd.Series, extended_special_chars: Sequence[str]) -> pd.Series:
    if not extended_special_chars:
        # If not provided, return zeros rather than failing.
        return pd.Series([0] * len(texts), index=texts.index)

    pattern = "[" + "".join(re.escape(ch) for ch in extended_special_chars) + "]"
    return texts.fillna("").astype(str).str.count(pattern)
