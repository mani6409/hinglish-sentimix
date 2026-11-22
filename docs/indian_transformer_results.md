# MuRIL and IndicBERT for Hinglish sentiment (issue #7)

## Status
| Part of the issue | Status |
|---|---|
| MuRIL fine-tuning code (`src/train_muril.py`) | Implemented, unit-tested on a tiny random model, and run. |
| MuRIL fine-tuning **run** and development predictions | **Done**: seeds 42, 43, 44 on a Kaggle T4 GPU. Predictions in `results/muril_dev_predictions.csv` (seed 42) and `results/predictions/muril_s43_seed43_dev.csv`, `muril_s44_seed44_dev.csv`. |
| IndicBERT checkpoint and tokenizer investigation | Done (below). |
| IndicBERT fine-tuning code (`src/train_indicbert.py`) | Implemented (same code as MuRIL) and run. |
| IndicBERT fine-tuning **run** | **Done for the open IndicBERT v2 checkpoint, seed 42 only.** `results/indicbert_dev_predictions.csv`. The original gated `ai4bharat/indic-bert` was not run. |
| Metrics (accuracy, per-class precision/recall, macro and weighted F1) | `src.metrics.compute_metrics` is reused; per-class scores are in each `*_scores_seed*.json`. |
| Predictions in common format | `uid,label,pred`. |
| Reuse of the shared pipeline of issue #5 | Not done; see "Relation to issue #5". |

## Results (development set, 3,000 tweets)
Same settings for both models: learning rate 2e-5, batch size 32, 3 epochs, max length 96, raw text. The best epoch was 3 in every run. All numbers are read from the `*_scores_seed*.json` files.

| Model | Seed | Dev weighted F1 | Macro F1 | Accuracy | Best epoch |
|---|---|---|---|---|---|
| MuRIL `google/muril-base-cased` | 42 | 0.6281 | 0.6326 | 0.6307 | 3 |
| MuRIL | 43 | 0.6218 | 0.6261 | 0.6223 | 3 |
| MuRIL | 44 | 0.6314 | 0.6365 | 0.6333 | 3 |
| **MuRIL, mean of 3 seeds** | | **0.6271** | **0.6317** | **0.6288** | |
| IndicBERT v2 `ai4bharat/IndicBERTv2-MLM-only` | 42 | 0.6305 | 0.6360 | 0.6347 | 3 |

For comparison (weighted F1, same dev set, from `results/transformer_comparison.csv`): XLM-R 0.6269 (mean of 3 seeds), mBERT 0.6099 (mean of 3 seeds), BiLSTM and TF-IDF baselines about 0.611 to 0.612.

How to read this:
- MuRIL's three seeds differ by up to 0.010 (0.6218 to 0.6314; standard deviation about 0.005). IndicBERT v2 has one seed, 0.6305, which lies inside that range. **The two models cannot be told apart with this evidence**, and MuRIL is also within 0.002 of XLM-R's mean. Do not claim a winner among MuRIL, IndicBERT v2 and XLM-R.
- MuRIL, IndicBERT v2 and XLM-R are all above mBERT by about 0.02 weighted F1 (0.017 to 0.021), which is several times mBERT's seed standard deviation (0.005). This is the only difference that the data supports.
- Per-class behaviour (dev recall, from `docs/error_analysis.md`): MuRIL negative 0.670, neutral 0.532, positive 0.723; IndicBERT v2 negative 0.737, neutral 0.495, positive 0.703. Both find the neutral class least often, as all other models do.
- Training time on a Kaggle T4 x2: about 2 minutes per epoch, so about 5.5 minutes per MuRIL seed and about 8 minutes for the IndicBERT v2 run (including model download).

## Checkpoints inspected
Run with `python -m src.train_muril --inspect-only` and `python -m src.train_indicbert --checkpoint ai4bharat/IndicBERTv2-MLM-only --inspect-only`. The numbers are measured on the 14,000 training tweets (raw text), subword length includes `[CLS]` and `[SEP]`.

| | MuRIL `google/muril-base-cased` | IndicBERT v2 `ai4bharat/IndicBERTv2-MLM-only` |
|---|---|---|
| Architecture / tokenizer | BERT-base / `BertTokenizerFast` (WordPiece) | BERT-style / `PreTrainedTokenizerFast` |
| Vocabulary size | 197,258 | 250,000 |
| Access | open | open |
| Unknown-token share of subwords | 0.62% | 0.04% |
| Subwords per whitespace word | 1.64 | 1.61 |
| Subword length p50 / p95 / p99 / max | 47 / 61 / 69 / 91 | 46 / 59 / 65 / 100 |

The original `ai4bharat/indic-bert` (ALBERT) is **gated** on the Hugging Face Hub. Without logging in with an account that has accepted its terms, loading it fails with HTTP 401 ("gated repo"). To use it: create a free Hugging Face account, accept the terms on the model page, then `huggingface-cli login` (or `login()` in a notebook), and run `python -m src.train_indicbert` (default checkpoint). It also needs the `sentencepiece` package (added to `requirements.txt`). I did not inspect or run the original `indic-bert` for that reason. The open v2 model above is a different, larger model; wherever it is used in the report it must be named as IndicBERT **v2**.

**Language coverage (not measured by us; figures are from my recollection of the model cards and papers, so verify them before citing):** MuRIL was pretrained on 17 Indian languages plus English, including *transliterated* (Roman-script) text; IndicBERT models cover about 12 (v1) or 23 (v2) Indian languages in native scripts. Hinglish tweets are mostly Roman-script Hindi, so MuRIL's transliterated pretraining data is the best match in principle. The dev scores above do not show a MuRIL advantage over IndicBERT v2 or XLM-R, so the expectation is not confirmed by our data. A low unknown-token share (as for both tokenizers here) means the vocabulary covers the characters, not that the words are understood.

**Maximum sequence length:** 96 is the default (`--max-len`). It covers every training tweet for MuRIL (max 91) and 99%+ for IndicBERT v2 (p99 65, max 100: a few tweets are cut).

## Where the runs were done
Fine-tuning a 237M-parameter BERT on 14,000 tweets for 3 epochs is a GPU job. The development PC has no NVIDIA GPU, so the runs were done on a free Kaggle notebook with a T4 x2 GPU and Internet on. The notebook `notebooks/colab_muril_indicbert.ipynb` in the repository shows the same steps for Colab (clone the repository, run the scripts); no results are stored in it. On Kaggle, the notebooks used for the runs were not added to the repository: each one cloned this repository, ran `python -m src.train_muril` (seeds 42, 43, 44) or `python -m src.train_indicbert`, and pushed `results/` to a branch. Running the commands in "How to run" gives the same result files.

## How to run
```bash
python -m src.train_muril --inspect-only         # tokenizer report only
python -m src.train_muril                        # seed 42 -> results/muril_dev_predictions.csv + muril_dev_scores_seed42.json
python -m src.train_muril --seed 43              # -> results/predictions/muril_s43_seed43_dev.csv
python -m src.train_muril --seed 44              # -> results/predictions/muril_s44_seed44_dev.csv
python -m src.train_indicbert --checkpoint ai4bharat/IndicBERTv2-MLM-only
python -m pytest tests/test_indian_transformer.py -q
```
Settings (identical for both models, all stored in the scores JSON): learning rate 2e-5, batch size 32, 3 epochs, linear warm-up over 10%, weight decay 0.01, max length 96, raw text, mixed precision on GPU, seeds via `src.seed.set_seed`. The best epoch is picked by development weighted F1, which uses the dev set for selection (allowed by the project rules; the test set is refused by the loader). A subset run (`--max-train-samples`) prints results but writes no files, so a smoke test cannot be mistaken for a real result. The files are picked up by `python -m src.error_analysis_models` and `python -m src.cmi_evaluation` (see `docs/error_analysis.md` and `docs/cmi_analysis.md`).

## Relation to issue #5
Issue #7 says to reuse the shared training pipeline of issue #5. That pipeline (`src/train_transformer.py`, `src/transformer_utils.py`, `configs/transformer_config.json`) is now on `main`, but `src/indian_transformer.py` was written before it and was **not** switched to it: it is a small self-contained loop that reuses the shared project parts (`src.data`, `src.preprocess`, `src.metrics`, `src.seed`, the `uid,label,pred` format) and keeps all training logic in one function, `fine_tune`, shared by MuRIL and IndicBERT. Because of this, the comparison with mBERT and XLM-R is not perfectly controlled: mBERT and XLM-R were tuned over several learning rates and trained for up to 6 epochs (best epoch 3 to 6, see `docs/transformer_results.md`), while MuRIL and IndicBERT v2 use one fixed setting (learning rate 2e-5, 3 epochs). XLM-R in particular chose a later best epoch in some runs, so it may be favoured. Moving `fine_tune` onto the shared pipeline and giving MuRIL the same learning-rate search would remove this difference.

## Limitations
- IndicBERT v2 has a single seed; MuRIL has three. A fair comparison needs the same number of seeds for both. The differences between MuRIL, IndicBERT v2 and XLM-R are within the seed spread.
- The original gated IndicBERT was not run; IndicBERT v2 is a different, larger model.
- MuRIL and IndicBERT v2 were not tuned (no learning-rate search), and the epoch count is fixed at 3; the comparison with the tuned mBERT and XLM-R is therefore not fully controlled (see "Relation to issue #5").
- Dev-based epoch selection makes the dev score slightly optimistic. All numbers here are dev scores; the test set was not used for these models.
- The MuRIL seed spread (0.010) is similar in size to the differences between models, so small differences must not be reported as wins.
