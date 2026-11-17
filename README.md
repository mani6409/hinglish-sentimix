# Sentiment Analysis of Hinglish (Code-Mixed) Tweets

Course project on SemEval-2020 Task 9 (SentiMix, Hindi-English). Goal: classify tweets as positive, negative or neutral.

## Team
Mani, Monis, Ekta, Akansha

## Approach
1. Baselines: TF-IDF + logistic regression, BiLSTM with word and character embeddings
2. Fine-tuned transformers: mBERT, XLM-R, MuRIL/IndicBERT
3. Zero-shot LLM prompting
4. Ablations: transliteration normalisation and code-mixing index (CMI)
5. Error analysis

## Repository layout
| Path | Purpose |
|---|---|
| `data/raw/` | Hinglish SemEval-2020 Task 9 data (committed, CC-BY 4.0); processed versions are generated, not committed |
| `notebooks/` | EDA and result visualisation |
| `src/` | Preprocessing, metrics, models, training script |
| `configs/` | One file per experiment recording its settings |
| `results/` | Experiment log, predictions, tables, figures |
| `docs/` | Paper summaries, problem statement, daily plan |
| `reports/` | Final report |

## Setup
Python version: **3.11** for everyone (so all installs match). Create the environment with that version, for example `python3.11 -m venv .venv`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Metric
The official metric is **weighted F1** (per-class F1 averaged by class size; Patwa et al., 2020, Sec. 3.1). Always use `src/metrics.py`:

```python
from src.metrics import compute_metrics
compute_metrics(y_true, y_pred)["weighted_f1"]
```

## Code you can use
- `src/data.py`: `load_split("train" | "dev" | "test")` returns one row per tweet (uid, label, text, tokens, langs).
- `src/seed.py`: `set_seed(42)` at the start of every script.
- `src/metrics.py`: `compute_metrics`, `confusion`.
- Tests: `python -m pytest -q`

## Data
The Hinglish data is included in the repo under `data/raw/`. Source, licence and format: see `docs/data_format.md`.

## Reproducing results
To be filled in as experiments are completed.

## References
1. Patwa et al., SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets. https://arxiv.org/abs/2008.04277
2. Conneau et al., Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). https://arxiv.org/abs/1911.02116
