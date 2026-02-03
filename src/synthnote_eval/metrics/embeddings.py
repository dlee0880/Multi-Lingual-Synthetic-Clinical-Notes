from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


@torch.inference_mode()
def embed_texts_cls_modernbert(
    texts: Iterable[str],
    model_name: str,
    batch_size: int = 16,
    max_length: int = 512,
    device: Optional[str] = None,
    use_fp16: bool = True,
) -> np.ndarray:
    """
    CLS embedding using a HuggingFace encoder model (ModernBERT variant).

    Returns:
        np.ndarray of shape (N, hidden_size)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    amp_enabled = bool(use_fp16 and device.startswith("cuda"))

    texts_list = pd.Series(list(texts)).fillna("").astype(str).tolist()
    out_chunks = []

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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :]
        out_chunks.append(cls.detach().cpu())

    return torch.cat(out_chunks, dim=0).numpy()
