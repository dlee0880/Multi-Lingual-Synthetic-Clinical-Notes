"""
synthnote_eval_single_page.py

One-file, GitHub-shareable evaluation utilities for synthetic clinical notes.
Includes:
  - Surface stats: tiktoken token length, sentence count, special character frequency
  - Diversity: sentence-level BLEU (self-BLEU helper)
  - Medical term density (bilingual): English scispaCy + Korean HF NER (with fallback)
  - Type-token ratio (TTR)
  - Distributional analysis: Zipf's law fit + plotting
  - Similarity to real notes: Jensen–Shannon distance (whitespace + tiktoken-id)
  - Semantics: CLS embeddings using BioClinical ModernBERT
  - Korean ratio: Korean character proportion

Notes
-----
- This is a pure Python script (no notebook/Colab magics).
- Heavy dependencies (scispaCy / transformers / torch) are imported lazily.
- UMLS linker in scispaCy is optional and may require extra setup.

Recommended installs (example):
  pip install numpy pandas scipy matplotlib tiktoken sacrebleu
  pip install torch transformers
  pip install spacy scispacy
  python -m spacy download en_core_sci_sm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Dict
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

import tiktoken
import sacrebleu


# =========================
# Configuration
# =========================

@dataclass(frozen=True)
class EvalConfig:
    tiktoken_encoding: str = "cl100k_base"
    english_scispacy_model: str = "en_core_sci_sm"
    korean_medical_ner_model: str = "SungJoo/medical-ner-koelectra"
    korean_fallback_ner_model: str = "monologg/koelectra-base-v3-naver-ner"
    modernbert_model: str = "thomas-sounack/BioClinical-ModernBERT-base"


# =========================
# Shared regex utilities
# =========================

SENT_BOUNDARY_RE = re.compile(r"(?<=[A-Za-z가-힣])\.|(?:\r?\n)\s*(?:\r?\n)")
TOK_RE = re.compile(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s]")
MIXED_TOKEN_RE = re.compile(r"\w+|[가-힣]+")
KOREAN_CHAR_RE = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]")


def split_sentences(text: str) -> List[str]:
    """Split into sentences using (a) period after alpha/hangul, (b) blank line."""
    if not isinstance(text, str):
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = SENT_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_for_bleu(text: str) -> str:
    """Stable normalization across mixed KO/EN for BLEU computation."""
    tokens = TOK_RE.findall(str(text))
    return " ".join(tokens)


def count_mixed_tokens(text: str) -> int:
    """Simple mixed token count used as denominator for bilingual density."""
    if not text:
        return 0
    return len([t for t in MIXED_TOKEN_RE.findall(str(text)) if t.strip()])


def korean_char_proportion(text: Any) -> float:
    """Ratio of Korean characters to total characters."""
    if not isinstance(text, str) or len(text) == 0:
        return float("nan")
    korean_chars = KOREAN_CHAR_RE.findall(text)
    return len(korean_chars) / len(text)


# =========================
# Surface statistics
# =========================

def token_length_tiktoken(texts: pd.Series, encoding_name: str = "cl100k_base") -> pd.Series:
    """Compute token length using tiktoken for each row."""
    enc = tiktoken.get_encoding(encoding_name)
    return texts.fillna("").astype(str).apply(lambda x: len(enc.encode(x)))


def sentence_count(texts: pd.Series) -> pd.Series:
    """Count sentences using split_sentences."""
    return texts.fillna("").astype(str).apply(lambda x: len(split_sentences(x)))


def extended_special_char_count(texts: pd.Series, extended_special_chars: Sequence[str]) -> pd.Series:
    """
    Count occurrences of user-defined special characters in each text.
    If list is empty, returns zeros.
    """
    if not extended_special_chars:
        return pd.Series([0] * len(texts), index=texts.index)

    pattern = "[" + "".join(re.escape(ch) for ch in extended_special_chars) + "]"
    return texts.fillna("").astype(str).str.count(pattern)


# =========================
# Diversity (self-BLEU helper)
# =========================

def sentence_bleu(hyp: str, refs: List[str]) -> float:
    """
    Compute sentence-level BLEU score of hyp against refs (normalized, tokenize='none').
    Returns sacreBLEU score (0-100).
    """
    hyp_n = normalize_for_bleu(hyp)
    refs_n = [normalize_for_bleu(r) for r in refs]
    return float(sacrebleu.sentence_bleu(hyp_n, refs_n, tokenize="none").score)


# =========================
# Medical term density (bilingual)
# =========================

def build_scispacy_english_pipeline(model_name: str, try_add_umls_linker: bool = False):
    """
    Build an English scispaCy pipeline.
    If try_add_umls_linker=True, attempts to add scispaCy UMLS linker;
    otherwise uses NER-only entities.
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
            # Fall back to NER-only silently.
            pass
    return nlp_en


def count_english_medical_entities(text: str, nlp_en) -> int:
    """
    Counts English medical entities.
    - If UMLS linker present: count only entities with kb_ents
    - Else: count all detected entities (NER-only fallback)
    """
    if not text:
        return 0

    doc = nlp_en(text)

    has_kb = any(hasattr(ent._, "kb_ents") for ent in doc.ents)
    if has_kb:
        return sum(1 for ent in doc.ents if getattr(ent._, "kb_ents", []))

    return len(doc.ents)


def build_korean_ner_pipeline(model_name: str, device: int = -1):
    """Build HF token-classification pipeline with simple aggregation."""
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline(
        task="token-classification",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy="simple",
        device=device,
    )


def count_korean_medical_entities(text: str, ner_ko) -> int:
    """Count Korean NER entities deduplicated by (start, end, label)."""
    if not text:
        return 0

    ents = ner_ko(text)
    seen = set()
    for e in ents:
        label = e.get("entity_group") or e.get("entity")
        key = (e.get("start"), e.get("end"), label)
        seen.add(key)
    return len(seen)


def calculate_medical_density_bilingual(text: Any, nlp_en, ner_ko) -> float:
    """
    medical_density = (english_entities + korean_entities) / total_mixed_tokens
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0.0

    text = str(text)
    denom = count_mixed_tokens(text)
    if denom == 0:
        return 0.0

    en_cnt = count_english_medical_entities(text, nlp_en)
    ko_cnt = count_korean_medical_entities(text, ner_ko)
    return float((en_cnt + ko_cnt) / denom)


def load_bilingual_med_density_pipelines(
    cfg: EvalConfig,
    device: int = -1,
    try_add_umls_linker: bool = False,
) -> Tuple[Any, Any]:
    """
    Convenience loader:
      - English scispaCy pipeline
      - Korean medical NER pipeline with fallback to general KO NER
    """
    nlp_en = build_scispacy_english_pipeline(cfg.english_scispacy_model, try_add_umls_linker=try_add_umls_linker)

    try:
        ner_ko = build_korean_ner_pipeline(cfg.korean_medical_ner_model, device=device)
    except Exception:
        ner_ko = build_korean_ner_pipeline(cfg.korean_fallback_ner_model, device=device)

    return nlp_en, ner_ko


# =========================
# Type-token ratio (TTR)
# =========================

def calculate_ttr(text: Any) -> float:
    """
    Type-token ratio using a simple regex tokenization that is stable across KO/EN.
    (Avoids requiring a global spaCy 'nlp' object.)
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0.0

    tokens = [t.lower() for t in TOK_RE.findall(str(text)) if t.strip() and not re.fullmatch(r"\W+", t)]
    if not tokens:
        return 0.0
    return float(len(set(tokens)) / len(tokens))


# =========================
# Distributional analysis: Zipf
# =========================

_WORD_RE = re.compile(r"\b\w+\b")


def zipf_fit(texts: pd.Series) -> Dict[str, float]:
    """
    Fit Zipf's law on corpus (log10 rank vs log10 frequency).
    Returns dict with slope, intercept, r2, p_value, std_err.
    """
    all_text = " ".join(texts.dropna().astype(str)).lower()
    words = _WORD_RE.findall(all_text)
    counts = Counter(words)

    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    if len(freqs) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "p_value": np.nan, "std_err": np.nan}

    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }


def plot_zipf(texts: pd.Series, title: str = "Zipf's Law Assessment (Log-Log Scale)") -> Dict[str, float]:
    """
    Plot Zipf distribution and fitted line. Returns fit metrics.
    """
    all_text = " ".join(texts.dropna().astype(str)).lower()
    words = _WORD_RE.findall(all_text)
    counts = Counter(words)

    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)

    if len(freqs) == 0:
        metrics = {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "p_value": np.nan, "std_err": np.nan}
        return metrics

    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freqs)

    plt.figure(figsize=(10, 6))
    plt.scatter(log_ranks, log_freqs, alpha=0.6, label="Observed Frequencies")
    plt.plot(
        log_ranks,
        slope * log_ranks + intercept,
        linestyle="--",
        linewidth=2,
        label=f"Linear Fit (R^2: {r_value**2:.3f})",
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Frequency)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.15)
    plt.show()

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }


# =========================
# Similarity vs real notes: Jensen–Shannon distance (JSD)
# =========================

def build_whitespace_token_probs(texts: pd.Series) -> Dict[str, float]:
    """Empirical token distribution using whitespace splitting."""
    corpus = " ".join(texts.dropna().astype(str)).strip()
    if not corpus:
        return {}
    toks = corpus.split()
    counts = Counter(toks)
    total = sum(counts.values())
    return {tok: cnt / total for tok, cnt in counts.items()}


def build_tiktoken_id_probs(texts: pd.Series, encoder) -> Dict[int, float]:
    """Empirical token-id distribution using tiktoken."""
    token_ids: List[int] = []
    for txt in texts.dropna().astype(str).tolist():
        token_ids.extend(encoder.encode(txt))

    if not token_ids:
        return {}

    counts = Counter(token_ids)
    total = sum(counts.values())
    return {tid: cnt / total for tid, cnt in counts.items()}


def js_distance_from_prob_dicts(p: Dict, q: Dict, base: float = 2.0) -> float:
    """
    SciPy jensenshannon returns a distance: sqrt(JS divergence).
    """
    support = sorted(set(p.keys()) | set(q.keys()))
    if not support:
        return float("nan")
    p_vec = np.array([p.get(k, 0.0) for k in support], dtype=float)
    q_vec = np.array([q.get(k, 0.0) for k in support], dtype=float)
    return float(jensenshannon(p_vec, q_vec, base=base))


def jsd_whitespace(texts_a: pd.Series, texts_b: pd.Series, base: float = 2.0) -> float:
    """Corpus-level JS distance using whitespace tokens."""
    p = build_whitespace_token_probs(texts_a)
    q = build_whitespace_token_probs(texts_b)
    return js_distance_from_prob_dicts(p, q, base=base)


def jsd_tiktoken_ids(texts_a: pd.Series, texts_b: pd.Series, encoding_name: str = "cl100k_base", base: float = 2.0) -> float:
    """Corpus-level JS distance using tiktoken token IDs."""
    enc = tiktoken.get_encoding(encoding_name)
    p = build_tiktoken_id_probs(texts_a, enc)
    q = build_tiktoken_id_probs(texts_b, enc)
    return js_distance_from_prob_dicts(p, q, base=base)


# =========================
# Semantics: BioClinical ModernBERT embeddings
# =========================

def embed_texts_cls_bioclinical_modernbert(
    texts: Iterable[str],
    model_name: str,
    batch_size: int = 16,
    max_length: int = 512,
    device: Optional[str] = None,
    use_fp16: bool = True,
) -> np.ndarray:
    """
    CLS embeddings using a HF encoder model (BioClinical ModernBERT).
    Returns np.ndarray of shape (N, hidden_dim).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    amp_enabled = bool(use_fp16 and device.startswith("cuda"))
    texts_list = pd.Series(list(texts)).fillna("").astype(str).tolist()

    chunks = []
    with torch.inference_mode():
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
            chunks.append(cls.detach().cpu())

    return torch.cat(chunks, dim=0).numpy()


# =========================
# High-level DataFrame runner
# =========================

def evaluate_dataframe(
    df: pd.DataFrame,
    text_col: str,
    cfg: Optional[EvalConfig] = None,
    extended_special_chars: Sequence[str] = (),
    compute_med_density: bool = False,
    med_density_device: int = -1,
    try_add_umls_linker: bool = False,
) -> pd.DataFrame:
    """
    Compute a standard set of metrics for df[text_col]. Returns a copy with new columns.

    Adds:
      - n_tokens
      - n_sentences
      - n_extended_special_chars
      - ttr
      - korean_ratio

    Optionally adds:
      - medical_density (requires heavy deps: spacy/scispacy/transformers/torch)
    """
    cfg = cfg or EvalConfig()
    out = df.copy()
    texts = out[text_col].fillna("").astype(str)

    out["n_tokens"] = token_length_tiktoken(texts, cfg.tiktoken_encoding)
    out["n_sentences"] = sentence_count(texts)
    out["n_extended_special_chars"] = extended_special_char_count(texts, extended_special_chars)
    out["ttr"] = texts.apply(calculate_ttr)
    out["korean_ratio"] = texts.apply(korean_char_proportion)

    if compute_med_density:
        nlp_en, ner_ko = load_bilingual_med_density_pipelines(
            cfg=cfg,
            device=med_density_device,
            try_add_umls_linker=try_add_umls_linker,
        )
        out["medical_density"] = texts.apply(lambda t: calculate_medical_density_bilingual(t, nlp_en, ner_ko))

    return out