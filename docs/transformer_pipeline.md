# Shared transformer training pipeline (issue #5)

One script fine-tunes any Hugging Face encoder on the three-class SentiMix Hinglish task. The model is a setting, not a code path. mBERT, XLM-R, MuRIL and IndicBERT all go through the same tokenisation, length limit, training loop, metrics, prediction format and checkpoint code.

| File | Role |
|---|---|
| `src/transformer_utils.py` | Config (`TransformerConfig`), model aliases, model and tokenizer loading, subword-length analysis, `max_length` derivation, dataset with dynamic padding, prediction, checkpoint save and load |
| `src/train_transformer.py` | Training and dev evaluation for one or more seeds, output protection, logging, CLI |
| `configs/transformer_config.json` | Shared default settings (the same values as the `TransformerConfig` defaults) |
| `tests/test_transformer_pipeline.py` | Offline tests with a tiny 1-layer BERT built in a temp folder (no download) |
| `results/transformer_token_lengths.json` | Subword-length statistics per tokenizer (train and dev) |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate                  # Windows; use source .venv/bin/activate on Linux/macOS
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build; plain `pip install torch` for CPU
pip install -r requirements.txt
```

The first run downloads the checkpoint from the Hugging Face Hub (about 700 MB for mBERT and 1.1 GB for XLM-R) into the local Hugging Face cache.

## Usage

```bash
# sequence lengths per tokenizer (train + dev only), writes results/transformer_token_lengths.json
python -m src.transformer_utils --models mbert xlmr muril --mode clean

# quick check on 200 tweets, 1 epoch: nothing is saved or logged
python -m src.train_transformer --model mbert --limit 200 --set epochs=1

# real runs (train split -> best epoch on dev), three seeds
python -m src.train_transformer --model mbert --seeds 42 43 44
python -m src.train_transformer --model xlmr  --seeds 42 43 44

# learning-rate search on dev: one run name per value
python -m src.train_transformer --model mbert --set lr=3e-5 --run-name mbert_clean_lr3e-5
```

For the full learning-rate search with three seeds, use `src/run_transformer_experiments.py`. Then build the comparison with `src/transformer_comparison.py`. The protocol and results are in `docs/transformer_results.md` (issue #6).

`--model` accepts the aliases `mbert` (`bert-base-multilingual-cased`), `xlmr` (`xlm-roberta-base`), `xlmr-large`, `muril` (`google/muril-base-cased`) and `indicbert` (`ai4bharat/indic-bert`). It also accepts any Hub id or local folder. Change any setting with `--set key=value` or a JSON file passed to `--config`.

## Outputs

These are the same conventions as the BiLSTM (`src/train_bilstm.py`). By default they go under `results/`:

| Path | Content |
|---|---|
| `predictions/transformer_<run>_seed<N>_dev.csv` | `uid,label,pred`, the common prediction format, which works with `src/cmi_evaluation.py` and the error analysis |
| `transformer_<run>_dev_scores.json` | Config, max-length details, per-seed metrics and history per epoch, confusion matrix, mean and sd over seeds |
| `experiment_log.csv` | One appended row per seed (weighted F1 in `dev_score`; macro F1, accuracy, best epoch and max length in `notes`) |
| `checkpoints/transformer_<run>_seed<N>/` | Best-dev-epoch model and tokenizer (`save_pretrained`) plus `run_info.json`. Git-ignored. Reload with `src.transformer_utils.load_checkpoint_dir` |

`<run>` defaults to `<model>_<mode>`, for example `mbert_clean`. Earlier results are never overwritten by accident:
- The run stops before training if any of its outputs already exist.
- Outputs are staged and published only after all seeds finish.
- `--overwrite` moves old files to `archive/<timestamp>/`.
- The work of a failed run is kept in `failed_runs/`.

## Rules for fair comparison

- **Splits.** Train on the official train split. Select the best epoch on the official dev split. The training script never loads the test split. The final test run is a separate, single step (issue #10).
- **Selection.** The best epoch is chosen by dev weighted F1. Early stopping uses `patience`. Because selection happens on dev, the dev score is slightly optimistic.
- **Metrics.** All metrics come from `src.metrics.compute_metrics`. Weighted F1 is the official score.
- **Same settings for every model.** Use the same `configs/transformer_config.json`, the same text `mode` and the same seeds. Only `model_name` (and a tuned `lr`) should differ.

## Settings and reproducibility

| Setting | Default | Why |
|---|---|---|
| `mode` | `clean` | URLs and mentions become the plain words `URL` and `USER`, hashtags lose `#`, and contractions are rejoined. Spellings are left alone because subword tokenizers handle variants. `normalized` is available for the ablation. |
| `max_length` | `null` (derived) | The smallest multiple of 8 that is at least the longest train tweet in subwords, so no train tweet is truncated. It is derived from train only and per tokenizer. |
| `batch_size` | 32 | Fits a 6 GB GPU with fp16 for base models. Use `grad_accum_steps` for larger effective batches. |
| `lr` | 2e-5 | The usual range for base encoders is 1e-5 to 5e-5. Search this range on dev. |
| `epochs` / `patience` | 4 / 2 | Base encoders usually peak within 2 to 4 epochs on about 14k examples. In practice every 4-epoch run here peaked in its last epoch, so issue #6 used `epochs=6` (see `docs/transformer_results.md`). |
| `warmup_ratio` | 0.1 | Linear warm-up, then linear decay (`get_linear_schedule_with_warmup`). |
| `weight_decay` | 0.01 | AdamW. Bias and LayerNorm weights are excluded. |
| `grad_clip` | 1.0 | |
| `fp16` | true | Mixed precision on CUDA only. It is ignored on CPU. |
| `device` | `auto` | CUDA when available, otherwise CPU. CPU training of a base model is slow (hours per epoch) and is meant only for `--limit` debug runs. |

How each source of randomness is controlled:
- `src.seed.set_seed(seed)` fixes Python, NumPy and PyTorch.
- The data shuffling uses a seeded `torch.Generator`.
- The classifier head is initialised right after the seed is set.

On GPU, repeated runs with the same seed are very close but not always bit-identical, because some CUDA kernels are non-deterministic. Report the mean and sd over seeds (42, 43, 44), as for the BiLSTM.

## Smoke test (22 Nov 2026)

This test checks that the pipeline runs end to end on the real data. These numbers are not results.
- **Setup.** One epoch on the full train split, scored on dev, seed 42, default config, `--limit 20000`, so nothing was saved.
- **Hardware.** RTX 4050 Laptop GPU (6 GB), fp16, torch 2.14.1+cu126, transformers 5.24.0.

| Model | Time per epoch | Dev weighted F1 after 1 epoch |
|---|---|---|
| mBERT | about 230 s | 0.581 |
| XLM-R | about 380 s | 0.575 |

- **Why the scores are low.** With `epochs=1` the learning rate warms up and decays to zero within that single epoch, so these scores sit below the TF-IDF baseline (0.611).
- **Real runs.** The real runs for issue #6 use the default 4 epochs and a learning-rate search.
- **Time for a full run.** A full 3-seed run takes roughly 45 minutes for mBERT and 75 minutes for XLM-R on this GPU.

## Maximum sequence length

Measured with `python -m src.transformer_utils --models mbert xlmr muril --mode clean`. Lengths include the special tokens. The full numbers are in `results/transformer_token_lengths.json`.

| Tokenizer | Train mean | p95 | p99 | p99.9 | Train max | Dev max | `max_length` |
|---|---|---|---|---|---|---|---|
| mBERT (`bert-base-multilingual-cased`) | 33.1 | 46 | 53 | 66 | 79 | 69 | **80** |
| XLM-R (`xlm-roberta-base`) | 30.6 | 42 | 47 | 61 | 90 | 90 | **96** |
| MuRIL (`google/muril-base-cased`) | 29.8 | 42 | 51 | 64 | 76 | 64 | **80** |

- **Words versus subwords.** The longest tweet has 56 words (EDA part 1) but up to 90 subwords. This is why the limit is measured per tokenizer and not taken from word counts.
- **No truncation.** With these limits no train or dev tweet is truncated. Every limit is far below the 512-position cap.
- **Padding cost.** Padding is dynamic (to the longest tweet in each batch), so a generous `max_length` costs almost nothing. Most batches are padded to about 50 to 60 subwords.
- **How the limit is derived.** The training script derives `max_length` itself on every run, from train only. Set `max_length` in the config to fix it by hand. Each run records the value and the number of truncated train and dev tweets in its scores JSON and log row.
