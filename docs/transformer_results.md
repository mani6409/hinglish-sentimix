# mBERT and XLM-R fine-tuning results (issue #6)

Both models were fine-tuned with the shared pipeline from issue #5 (`docs/transformer_pipeline.md`), with a learning-rate search on dev and three seeds each. All numbers are on the official **dev** split. The test split was not used.

**Summary.** On every seed, XLM-R (`xlm-roberta-base`) scores higher than mBERT (`bert-base-multilingual-cased`).

| Model | Dev weighted F1 (mean ± sd, 3 seeds) | Compared with the baselines |
|---|---|---|
| XLM-R | **0.627 ± 0.003** | About +0.015 above TF-IDF (0.613) and the BiLSTM (0.612) |
| mBERT | 0.610 ± 0.005 | No better than the baselines |

On the same dev tweets, the paired bootstrap interval for the mBERT − XLM-R difference just includes zero. So the ranking holds for every training seed, but the gap (about 0.017) is about the size of the dev-set sampling error.

## Protocol

1. **Settings.** Everything uses the shared `configs/transformer_config.json` except `epochs=6` (see the next section).
   - Text mode is `clean`.
   - Batch size is 32, with fp16.
   - Warm-up is 10%, followed by linear decay.
   - AdamW uses weight decay 0.01.
   - Early stopping has patience 2.
   - `max_length` is derived from train with no truncation: 80 for mBERT and 96 for XLM-R.
2. **Learning-rate search.** Each model was trained at lr 1e-5, 2e-5 and 3e-5, with one seed (42) per lr. The best epoch of each run is chosen by dev weighted F1.
3. **Selection.** Per model, the lr with the highest dev weighted F1 is kept (ties go to the smaller lr). See `results/transformer_lr_search_ep6.csv`.
4. **Final seeds.** The selected lr is run with seeds 43 and 44. Together with the seed-42 search run, that gives three seeds.
5. **Seed 42 is slightly optimistic.** Seed 42 helped choose the lr, so its score is a little high. The table therefore also gives the mean over seeds 43 and 44 alone.
6. **Comparison.** `src/transformer_comparison.py` recomputes every metric from the prediction files with `src.metrics.compute_metrics` and checks it against the score the training run reported. It also checks that all files cover the same dev uids. It then adds the paired bootstrap (1000 resamples).

Commands:

```bash
python -m src.run_transformer_experiments --suffix _ep6 --set epochs=6 --lrs 1e-5 2e-5 3e-5 --prune-checkpoints
python -m src.transformer_comparison --suffix _ep6
```

`--suffix _ep6` keeps these run names (`mbert_clean_ep6_lr2e-5`, ...) apart from the earlier 4-epoch runs. `--set` applies a config override to every run.

## Why 6 epochs and not the default 4

A first search used the default 4 epochs. **Every** one of those runs had its best dev score in the last epoch, which suggests the models were still improving when training stopped:

| Run (4 epochs, seed 42) | Dev weighted F1 | Best epoch |
|---|---|---|
| mBERT lr 1e-5 | 0.6113 | 4 of 4 |
| mBERT lr 2e-5 | 0.6054 | 4 of 4 |
| mBERT lr 3e-5 | 0.6051 | 4 of 4 |
| mBERT lr 5e-5 | 0.6062 | 4 of 4 |
| XLM-R lr 1e-5 | 0.6255 | 4 of 4 |

- **Fresh runs.** The warm-up and decay schedule depends on the total number of epochs, so the runs could not simply be continued. All runs were redone with 6 epochs.
- **What was kept.** The 4-epoch files (`transformer_*_clean_lr*`) are kept as the record of this check. The XLM-R 4-epoch search was stopped after its first lr.
- **The results below are the 6-epoch runs.**

With 6 epochs, most runs peaked before the end and early stopping ended several of them, so 6 epochs gives enough room. The selected mBERT runs peaked at epoch 3 on all seeds. The selected XLM-R runs peaked at epochs 6, 3 and 5. Seed 42 still peaked in its last epoch, but the gain from epoch 4 to 6 was only 0.002, and the other two seeds peaked earlier.

## Learning-rate search (6 epochs, seed 42)

| Model | lr | Dev weighted F1 | Macro F1 | Accuracy | Best epoch / run | Selected |
|---|---|---|---|---|---|---|
| mBERT | 1e-5 | 0.6088 | 0.6130 | 0.6103 | 6 / 6 | |
| mBERT | 2e-5 | **0.6129** | 0.6172 | 0.6137 | 3 / 5 | yes |
| mBERT | 3e-5 | 0.6063 | 0.6107 | 0.6070 | 3 / 5 | |
| XLM-R | 1e-5 | **0.6301** | 0.6357 | 0.6350 | 6 / 6 | yes |
| XLM-R | 2e-5 | 0.6300 | 0.6345 | 0.6300 | 4 / 6 | |
| XLM-R | 3e-5 | 0.6177 | 0.6241 | 0.6223 | 5 / 6 | |

- **mBERT.** The three learning rates are within 0.007 of each other.
- **XLM-R, 1e-5 against 2e-5.** These two are tied: 0.6301 against 0.6300 is noise. 1e-5 was selected by a margin of 0.0001, so 2e-5 would have been an equally fair choice.
- **XLM-R, 3e-5.** This lr is clearly worse. Higher learning rates fit the training data faster: the training loss in the last epoch was 0.66 at 1e-5 and 0.51 at 3e-5. They also overfit sooner.

## Final results (dev, 3 seeds at the selected lr)

Full table: `results/transformer_comparison.csv`. It has a `mean` row per model, one `seed` row per run, and reference rows for the baselines.

| Model | lr | Weighted F1 | Macro F1 | Accuracy | Macro P | Macro R | Weighted P | Weighted R | wF1, seeds 43+44 only |
|---|---|---|---|---|---|---|---|---|---|
| **XLM-R** | 1e-5 | **0.6269 ± 0.0029** | 0.6323 ± 0.0029 | 0.6309 ± 0.0036 | 0.6301 | 0.6392 | 0.6278 | 0.6309 | 0.6254 |
| mBERT | 2e-5 | 0.6099 ± 0.0046 | 0.6138 ± 0.0046 | 0.6097 ± 0.0048 | 0.6147 | 0.6132 | 0.6105 | 0.6097 | 0.6084 |
| TF-IDF char 2-5 + LR (reference) | | 0.6127 | 0.6164 | 0.6123 | 0.6170 | 0.6162 | 0.6135 | 0.6123 | |
| BiLSTM word+char, clean (reference) | | 0.6118 ± 0.0015 | 0.6172 ± 0.0022 | 0.6133 ± 0.0028 | 0.6160 | 0.6199 | 0.6119 | 0.6133 | |

- **What ± means.** Mean ± sample sd over seeds 42, 43 and 44. Precision and recall columns are means.
- **Weighted recall.** Weighted recall equals accuracy by definition.

Per seed (weighted F1):

| Seed | mBERT | XLM-R | mBERT − XLM-R |
|---|---|---|---|
| 42 (used to choose lr) | 0.6129 | 0.6301 | −0.0172 |
| 43 | 0.6122 | 0.6265 | −0.0143 |
| 44 | 0.6047 | 0.6243 | −0.0196 |

### Is XLM-R really better?

- **Across training seeds: yes, consistently.**
  - XLM-R wins on all three seeds, by 0.014 to 0.020.
  - The seed-to-seed sd is about 0.003 for XLM-R and 0.005 for mBERT.
- **Across samples of tweets: not established.**
  - The paired bootstrap resamples the same dev tweets for both models (representative runs, seed 43 for each, 1000 resamples). The mBERT − XLM-R difference is **−0.014, with a 95% interval from −0.030 to +0.003**.
  - The interval just includes zero, so on 3,000 dev tweets we cannot rule out that the two models would tie on another sample of tweets.
  - The file is `results/transformer_pairwise.csv`.
- **Agreement.** The two representative runs agree on 76% of dev tweets.

Fair wording: *"XLM-R scored higher than mBERT on every training seed (mean dev weighted F1 0.627 against 0.610). On the dev set the gap is close to the sampling error (bootstrap 95% interval −0.030 to +0.003)."*

**Against the baselines:**
- **XLM-R** is about 0.014 above TF-IDF and 0.015 above the BiLSTM. That gap is about the same size as the mBERT − XLM-R gap, so the same caution applies.
- **mBERT** is level with both baselines, slightly below on the mean.

### Per class (representative runs, seed 43)

| Class | mBERT P / R / F1 | XLM-R P / R / F1 |
|---|---|---|
| negative (890) | 0.640 / 0.612 / 0.626 | 0.650 / 0.645 / **0.648** |
| neutral (1128) | 0.534 / 0.570 / **0.552** | 0.570 / 0.522 / 0.545 |
| positive (982) | 0.683 / 0.657 / 0.669 | 0.669 / 0.737 / **0.701** |

- **Same pattern as the baselines.** Neutral is the hardest class for both models.
- **Where XLM-R gains.** Its gain comes from negative and especially positive: positive recall rises from 0.66 to 0.74. It predicts neutral less often, so neutral recall drops and neutral precision rises.
- **Errors still involve neutral.** Most errors in both models still involve the neutral class. The clearest case is neutral tweets predicted as positive: 255 for mBERT and 298 for XLM-R.

## Deliverables

| File | Content |
|---|---|
| `results/mbert_dev_predictions.csv` | Representative mBERT run (seed 43, the median dev weighted F1 of the three), `uid,label,pred` |
| `results/xlmr_dev_predictions.csv` | Representative XLM-R run (seed 43, median), `uid,label,pred` |
| `results/transformer_comparison.csv` | Mean and sd per model, per-seed rows, baseline reference rows; weighted F1, macro F1, accuracy, macro and weighted precision and recall |
| `results/transformer_pairwise.csv` | mBERT − XLM-R per seed, plus the paired-bootstrap interval |
| `results/transformer_lr_search_ep6.csv` | The 6-epoch search and the selected lr |
| `results/predictions/transformer_*_dev.csv`, `results/transformer_*_dev_scores.json` | Every run, including the 4-epoch check: config, per-epoch history and confusion matrix |
| `src/run_transformer_experiments.py` | Search, select and final-seeds driver (restartable: finished runs are skipped) |
| `src/transformer_comparison.py` | Builds the comparison files above |

**Checkpoints.** The checkpoints of the selected runs are in `results/checkpoints/` (git-ignored): mBERT lr 2e-5 seeds 42, 43, 44 and XLM-R lr 1e-5 seeds 42, 43, 44. The test run in issue #10 can reload them with `src.transformer_utils.load_checkpoint_dir`. The checkpoints of the unselected search runs were deleted.

## Limitations

- **Small search.** The 6-epoch search covered lr 1e-5 to 3e-5 with a single seed per lr.
  - 5e-5 was tried only for mBERT and only at 4 epochs, where it scored 0.6062.
  - XLM-R scores fell from 2e-5 to 3e-5, so higher rates are unlikely to help.
  - Batch size, warm-up and weight decay were not tuned.
- **Selection on dev.** Both the lr and the best epoch were chosen on dev, so the dev scores are somewhat optimistic. Seeds 43 and 44 alone give 0.625 for XLM-R and 0.608 for mBERT. Only the single test run (issue #10) gives an unbiased number.
- **GPU non-determinism.** Repeating a seed on GPU gives very close but not always bit-identical results.
- **One text mode.** Only the `clean` text mode was run. Raw and normalized modes for transformers remain open.
- **Hardware and timing.** RTX 4050 Laptop GPU (6 GB), fp16, torch 2.14.1+cu126, transformers 5.24.0.
  - mBERT took about 80 s per epoch, about 7 min per run.
  - XLM-R took about 195 s per epoch, about 16–20 min per run.
  - XLM-R uses nearly all of the 6 GB. With browsers and other GPU apps open, its epochs slowed to 12–21 min, probably because GPU memory overflowed into system RAM. Close other GPU apps before training XLM-R on a 6 GB card.
  - One early XLM-R run crashed with an access violation (0xC0000005) while a CPU test suite ran at the same time. Run alone, it succeeded.

## Dependency

This work builds on the shared pipeline in PR #19 (issue #5): `src/transformer_utils.py`, `src/train_transformer.py` and `configs/transformer_config.json`. The branch is stacked on it, so PR #19 should be merged first.
