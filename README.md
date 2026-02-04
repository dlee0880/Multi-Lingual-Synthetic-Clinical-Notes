# Multi-Lingual-Synthetic-Clinical-Notes
The development of iterative feedback-loop framework to generate multi-lingual synthetic clinical notes for realistic data augmentation focusing both on semantic and surface-level fidelity with real-world clinical notes.

# synthnote-eval

Evaluation metrics for synthetic clinical notes:
- Surface statistics (token length, sentence count, special character frequency)
- Lexical Diversity (self-BLEU, medical terminology, type-token ratio, Korean character burden)
- Distributional checks (Zipf slope, Jensen–Shannon distance)
- Semantic embeddings (BioClinical ModernBERT CLS embeddings)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

