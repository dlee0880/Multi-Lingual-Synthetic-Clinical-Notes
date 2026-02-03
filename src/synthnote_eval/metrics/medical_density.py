from __future__ import annotations

from typing import Any, Optional, Tuple

import pandas as pd

from synthnote_eval.utils.text import count_mixed_tokens


def build_scispacy_english_pipeline(
    model_name: str = "en_core_sci_sm",
    try_add_umls_linker: bool = False,
):
    """
    Build an English scispaCy pipeline.
    If try_add_umls_linker=True, attempts to add the UMLS linker.
    If linker isn't available/configured, code falls back to NER-only counting.
    """
    import spacy

    nlp_en = spacy.load(model_name)

    if try_add_umls_linker:
        try:
            nlp_en.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
            )
        except Exception:
            # Silent fallback: still usable as NER-only.
            pass

    return nlp_en


def _count_english_medical_entities(text: str, nlp_en) -> int:
    if not text:
        return 0

    doc = nlp_en(text)

    # If UMLS linker is present, entities may have ent._.kb_ents
    has_kb = any(hasattr(ent._, "kb_ents") for ent in doc.ents)
    if has_kb:
        return sum(1 for ent in doc.ents if getattr(ent._, "kb_ents", []))

    return len(doc.ents)


def build_korean_ner_pipeline(model_name: str, device: int = -1):
    """
    Build a HuggingFace NER pipeline (token-classification) with simple aggregation.
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline(
        task="token-classification",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy="simple",
        device=device,
    )


def _count_korean_entities(text: str, ner_ko) -> int:
    if not text:
        return 0

    ents = ner_ko(text)
    seen = set()
    for e in ents:
        label = e.get("entity_group") or e.get("entity")
        key = (e.get("start"), e.get("end"), label)
        seen.add(key)
    return len(seen)


def calculate_medical_density_bilingual(
    text: Any,
    nlp_en,
    ner_ko,
) -> float:
    """
    medical_density = (english_entity_count + korean_entity_count) / total_mixed_tokens
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0.0

    text = str(text)
    total_tokens = count_mixed_tokens(text)
    if total_tokens == 0:
        return 0.0

    en_cnt = _count_english_medical_entities(text, nlp_en)
    ko_cnt = _count_korean_entities(text, ner_ko)
    return float((en_cnt + ko_cnt) / total_tokens)
