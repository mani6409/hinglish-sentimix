# MuRIL and IndicBERT for Hinglish sentiment (issue #7)

## Status (read this first)
| Part of the issue | Status |
|---|---|
| MuRIL fine-tuning code (`src/train_muril.py`) | Implemented and unit-tested on a tiny random model. |
| MuRIL fine-tuning **run** and development predictions | **Not run yet.** Needs a GPU; see "Where to run". No score is reported because none exists. |
| IndicBERT checkpoint and tokenizer investigation | Done (below). |
| IndicBERT fine-tuning code (`src/train_indicbert.py`) | Implemented (same code as MuRIL). **Not run.** |
| Metrics (accuracy, per-class precision/recall, macro and weighted F1) | `src.metrics.compute_metrics` is reused; nothing new. |
| Predictions in common format | The scripts write `uid,label,pred`. Files do not exist until the run is done. |
| Reuse of the shared pipeline of issue #5 | **Not possible yet**: `src/train_transformer.py` is not on `main`. See "Relation to issue #5". |

When the runs are finished, fill in the results table at the end of this file with the numbers from `results/muril_dev_scores_seed*.json`.

## Checkpoints inspected
Run with `python -m src.train_muril --inspect-only` and `python -m src.train_indicbert --checkpoint ai4bharat/IndicBERTv2-MLM-only --inspect-only`. The numbers are measured on the 14,000 training tweets (raw text), subword length includes `[CLS]` and `[SEP]`.

| | MuRIL `google/muril-base-cased` | IndicBERT v2 `ai4bharat/IndicBERTv2-MLM-only` |
|---|---|---|
| Architecture / tokenizer | BERT-base / `BertTokenizerFast` (WordPiece) | BERT-style / `PreTrainedTokenizerFast` |
| Vocabulary size | 197,285 | 250,000 |
| Access | open | open |
| Unknown-token share of subwords | 0.62% | 0.04% |
| Subwords per whitespace word | 1.64 | 1.61 |
| Subword length p50 / p95 / p99 / max | 47 / 61 / 69 / 91 | 46 / 59 / 65 / 100 |

The original `ai4bharat/indic-bert` (ALBERT) is **gated** on the Hugging Face Hub. Without logging in with an account that has accepted its terms, loading it fails with HTTP 401 ("gated repo") – this is what happened on this PC. To use it: create a free Hugging Face account, accept the terms on the model page, then `huggingface-cli login` (or `login()` in Colab), and run `python -m src.train_indicbert` (default checkpoint). It also needs the `sentencepiece` package (added to `requirements.txt`). I did not inspect or run the original `indic-bert` for that reason. The open v2 model above is a different, larger model; if it is used in the report it must be named as IndicBERT **v2**.

**Language coverage (not measured by us; figures are from my recollection of the model cards and papers, so verify them before citing):** MuRIL was pretrained on 17 Indian languages plus English, including *transliterated* (Roman-script) text; IndicBERT models cover about 12 (v1) or 23 (v2) Indian languages in native scripts. Hinglish tweets are mostly Roman-script Hindi, so MuRIL's transliterated pretraining data is the best match in principle. That is an expectation, not a result; only the dev scores can show whether it helps. A low unknown-token share (as for both tokenizers here) means the vocabulary covers the characters, not that the words are understood.

**Maximum sequence length:** 96 is the default (`--max-len`). It covers every training tweet for MuRIL (max 91) and 99%+ for IndicBERT v2 (p99 65, max 100: a few tweets are cut).

## Where to run
Fine-tuning a 237M-parameter BERT on 14,000 tweets for 3 epochs is a GPU job. This PC (Intel i5-1334U CPU, 16 GB RAM, Intel Iris Xe graphics, no NVIDIA GPU) can run the tests and the tokenizer inspection, but a MuRIL run would take hours per epoch on the CPU (not measured; estimate). Use Google Colab with a T4 GPU: `notebooks/colab_muril_indicbert.ipynb` runs the scripts and downloads `results/`.

## How to run
```bash
python -m src.train_muril --inspect-only         # tokenizer report only
python -m src.train_muril                        # seed 42 -> results/muril_dev_predictions.csv + muril_dev_scores_seed42.json
python -m src.train_muril --seed 43              # -> results/predictions/muril_s43_seed43_dev.csv
python -m src.train_indicbert --checkpoint ai4bharat/IndicBERTv2-MLM-only
python -m pytest tests/test_indian_transformer.py -q
```
Settings (identical for both models, all stored in the scores JSON): learning rate 2e-5, batch size 32, 3 epochs, linear warm-up over 10%, weight decay 0.01, max length 96, raw text, mixed precision on GPU, seeds via `src.seed.set_seed`. The best epoch is picked by development weighted F1, which uses the dev set for selection (allowed by the project rules; the test set is refused by the loader). A subset run (`--max-train-samples`) prints results but writes no files, so a smoke test cannot be mistaken for a real result. After the runs, `python -m src.error_analysis_models` and `python -m src.cmi_evaluation` pick up the new prediction files automatically.

## Relation to issue #5
Issue #7 says to reuse the shared training pipeline of issue #5. At the time of writing that pipeline (`src/train_transformer.py`, `src/transformer_utils.py`, `configs/transformer_config.json`) does not exist on `main` or on any pushed branch. `src/indian_transformer.py` is therefore a small self-contained loop that reuses the shared project parts (`src.data`, `src.preprocess`, `src.metrics`, `src.seed`, the `uid,label,pred` format) and keeps all training logic in one function, `fine_tune`, shared by MuRIL and IndicBERT. When #5 is merged, replace `fine_tune` by a call to it so the project has one training loop, and check that the settings above match the ones used for mBERT and XLM-R.

## Limitations
- No fine-tuning result exists yet; nothing here says how well MuRIL or IndicBERT perform.
- One seed is the default; the plan asks for three. Differences smaller than the seed spread must not be claimed as wins.
- The original IndicBERT is gated and was not inspected; IndicBERT v2 is a different model.
- Dev-based epoch selection makes the dev score slightly optimistic.

## Results (to be filled after the Colab runs)
| Model | Seed | Dev weighted F1 | Macro F1 | Accuracy | Best epoch |
|---|---|---|---|---|---|
| MuRIL | 42 | | | | |
| MuRIL | 43 | | | | |
| MuRIL | 44 | | | | |
| IndicBERT (state which checkpoint) | 42 | | | | |
