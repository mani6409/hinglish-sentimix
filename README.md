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
Python version: **3.11** for the team (the BiLSTM evaluation was run on 3.12.5; see `requirements-bilstm.txt`) (so all installs match). Create the environment with that version, for example `python3.11 -m venv .venv`.

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
The official SemEval-2020 Task 9 Hinglish data (CC BY 4.0, Zenodo record 3974927) is included in the repository under
`data/raw/zenodo_download/extracted/Semeval_2020_task9_data/Hinglish/`, which is where `src/data.py` looks by default.
Source, licence and format: `docs/data_format.md`. The test labels are in a separate file and are joined by uid.

## BiLSTM (word + character embeddings)
Word embeddings plus a character CNN feed a BiLSTM with max and mean pooling. Three preprocessing modes (raw, clean, normalized) were each
trained with seeds 42, 43, 44; everything except the mode is identical. Details: `docs/bilstm.md`.

**Test results** (mean ± sample std over 3 seeds; official test set, evaluated once under `docs/bilstm_final_evaluation_protocol.md`):

| Mode | Macro F1 | Weighted F1 |
|---|---|---|
| raw | 0.6615 ± 0.0077 | 0.6603 ± 0.0066 |
| clean | 0.6716 ± 0.0030 | 0.6681 ± 0.0023 |
| normalized | 0.6589 ± 0.0061 | 0.6566 ± 0.0059 |

Differences between modes are small compared with seed variation and test-sampling uncertainty; see `results/bilstm_test_analysis/bilstm_final_analysis.md` for the
paired bootstrap intervals, error analysis and the (partly unexplained) gap between dev and test scores. Normalization did not help.

## Reproducing the BiLSTM results
The notebook `notebooks/bilstm.ipynb` shows the preprocessing and all BiLSTM tables and curves from the saved results.

Install the BiLSTM dependencies (`pip install -r requirements-bilstm.txt`; Python 3.12.5 and CPU were used) and run from the repository root:

```bash
python -m pytest -q tests/test_bilstm_data.py tests/test_bilstm_model.py tests/test_bilstm_limits.py tests/test_bilstm_checkpoint.py tests/test_bilstm_outputs.py
python -m src.train_bilstm --config configs/bilstm_default.json --mode clean --seeds 42 43 44     # one mode; never loads the test split
```

The nine checkpoints are git-ignored; their MD5 hashes are in `configs/bilstm_frozen_checkpoints.json`. The final test evaluation
(`--confirm-final-test`) was run once and refuses to overwrite `results/bilstm_final_test/`.

## References
1. Patwa et al., SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets. https://arxiv.org/abs/2008.04277
2. Conneau et al., Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). https://arxiv.org/abs/1911.02116
