from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvalConfig:
    tiktoken_encoding: str = "cl100k_base"
    modernbert_model: str = "thomas-sounack/BioClinical-ModernBERT-base"
    korean_med_ner_model: str = "SungJoo/medical-ner-koelectra"
    korean_fallback_ner_model: str = "monologg/koelectra-base-v3-naver-ner"

    # Special characters set should be provided by user (project-specific).
    extended_special_chars: Sequence[str] = ()
