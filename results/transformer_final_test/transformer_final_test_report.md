# Transformer final test-set evaluation (single, frozen-model run)

*Evaluates the six frozen checkpoints (mBERT and XLM-R x seeds 42, 43, 44) on the official SentiMix Hinglish test set (3,000 tweets). Protocol: `docs/transformer_final_evaluation_protocol.md`. Produced by `python -m src.evaluate_transformer_test --confirm-final-test` on 2026-11-22T10:27:54+00:00.*

**Nothing was selected, tuned or changed based on these results.** Checkpoints, learning rates, text preprocessing, inference settings and metrics were frozen before the test labels were read. No model is declared the winner by this evaluation.

## 1. Primary results: mean ± sample standard deviation over seeds 42, 43, 44

| Model | Checkpoint | lr | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| mBERT | `bert-base-multilingual-cased` | 2e-05 | **0.6893 ± 0.0076** | **0.6866 ± 0.0081** |
| XLM-R | `xlm-roberta-base` | 1e-05 | **0.7034 ± 0.0041** | **0.7006 ± 0.0041** |

Per seed (seed 42 / 43 / 44):

| Model | Macro F1 | Weighted F1 |
|---|---|---|
| mBERT | 0.6805 / 0.6930 / 0.6943 | 0.6773 / 0.6908 / 0.6918 |
| XLM-R | 0.7077 / 0.7031 / 0.6994 | 0.7044 / 0.7011 / 0.6963 |

Pre-specified descriptive comparison, mBERT minus XLM-R per seed (same seed, same test tweets):

| Seed | Macro F1 difference | Weighted F1 difference |
|---|---|---|
| 42 | -0.0272 | -0.0271 |
| 43 | -0.0101 | -0.0103 |
| 44 | -0.0051 | -0.0045 |

Standard deviations are sample standard deviations (n = 3, ddof = 1) and are rough with three seeds. The test set is one sample of 3,000 tweets, so F1 carries a sampling error of roughly ±0.01 that is not estimated here.

## 2. Secondary metrics (mean ± sample std over seeds)

| Model | Accuracy | Macro precision | Macro recall | Weighted precision | Weighted recall |
|---|---|---|---|---|---|
| mBERT | 0.6848 ± 0.0074 | 0.6958 ± 0.0115 | 0.6856 ± 0.0069 | 0.6913 ± 0.0112 | 0.6848 ± 0.0074 |
| XLM-R | 0.7003 ± 0.0043 | 0.7050 ± 0.0047 | 0.7029 ± 0.0048 | 0.7018 ± 0.0039 | 0.7003 ± 0.0043 |

Weighted recall equals accuracy by definition.

## 3. Class-wise precision, recall and F1 (mean ± sample std over seeds)

| Model | Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|---|
| mBERT | negative | 900 | 0.7255 ± 0.0175 | 0.6778 ± 0.0135 | 0.7006 ± 0.0030 |
| mBERT | neutral | 1,100 | 0.5915 ± 0.0113 | 0.6536 ± 0.0405 | 0.6206 ± 0.0177 |
| mBERT | positive | 1,000 | 0.7703 ± 0.0234 | 0.7253 ± 0.0247 | 0.7467 ± 0.0088 |
| XLM-R | negative | 900 | 0.7134 ± 0.0259 | 0.7067 ± 0.0443 | 0.7090 ± 0.0149 |
| XLM-R | neutral | 1,100 | 0.6193 ± 0.0102 | 0.6297 ± 0.0249 | 0.6242 ± 0.0084 |
| XLM-R | positive | 1,000 | 0.7822 ± 0.0096 | 0.7723 ± 0.0162 | 0.7771 ± 0.0049 |

## 4. Every run

| Run | Accuracy | Macro P | Macro R | Macro F1 | Weighted P | Weighted R | Weighted F1 |
|---|---|---|---|---|---|---|---|
| mBERT seed 42 | 0.6763 | 0.6835 | 0.6783 | 0.6805 | 0.6790 | 0.6763 | 0.6773 |
| mBERT seed 43 | 0.6877 | 0.7062 | 0.6866 | 0.6930 | 0.7008 | 0.6877 | 0.6908 |
| mBERT seed 44 | 0.6903 | 0.6977 | 0.6919 | 0.6943 | 0.6942 | 0.6903 | 0.6918 |
| XLM-R seed 42 | 0.7047 | 0.7071 | 0.7084 | 0.7077 | 0.7041 | 0.7047 | 0.7044 |
| XLM-R seed 43 | 0.7003 | 0.7083 | 0.7002 | 0.7031 | 0.7040 | 0.7003 | 0.7011 |
| XLM-R seed 44 | 0.6960 | 0.6996 | 0.7001 | 0.6994 | 0.6974 | 0.6960 | 0.6963 |

## 5. Confusion matrices

Rows = true class, columns = predicted class. Each matrix is the **sum over the three seeds** (9,000 predictions per model); percentages are row-normalised. Per-run matrices are in `confusion_matrices.json`.

**mBERT**

| | pred negative | pred neutral | pred positive | Row total |
|---|---|---|---|---|
| true negative | 1,830 (67.8%) | 767 (28.4%) | 103 (3.8%) | 2,700 |
| true neutral | 594 (18.0%) | 2,157 (65.4%) | 549 (16.6%) | 3,300 |
| true positive | 100 (3.3%) | 724 (24.1%) | 2,176 (72.5%) | 3,000 |

**XLM-R**

| | pred negative | pred neutral | pred positive | Row total |
|---|---|---|---|---|
| true negative | 1,908 (70.7%) | 691 (25.6%) | 101 (3.7%) | 2,700 |
| true neutral | 677 (20.5%) | 2,078 (63.0%) | 545 (16.5%) | 3,300 |
| true positive | 95 (3.2%) | 588 (19.6%) | 2,317 (77.2%) | 3,000 |

Figure: `figures/transformer_final_test_confusion.png`.

## 6. Integrity and procedure

- Dev reproduction before the test files were touched (agreement with the frozen dev predictions; weighted F1 recomputed vs. reported in training):

| Run | Agreement | Changed predictions | Dev wF1 now | Dev wF1 in training |
|---|---|---|---|---|
| mbert/seed42 | 1.0000 | 0 | 0.6129 | 0.6129 |
| mbert/seed43 | 1.0000 | 0 | 0.6122 | 0.6122 |
| mbert/seed44 | 1.0000 | 0 | 0.6047 | 0.6047 |
| xlmr/seed42 | 1.0000 | 0 | 0.6301 | 0.6301 |
| xlmr/seed43 | 1.0000 | 0 | 0.6265 | 0.6265 |
| xlmr/seed44 | 1.0000 | 0 | 0.6243 | 0.6243 |

- All checkpoint files, the frozen dev predictions and the unlabeled test file were MD5-verified **before** inference. The labels file was verified and read for the **first time only after** all six test prediction files had been written and validated. Every hashed input was verified **again after** evaluation and was unchanged.
- Predictions were joined to the gold labels by **UID**, never by row order. Metrics were computed from the saved prediction files.
- Hashes, the time of every stage, the environment and the output hashes are in `integrity_manifest.json`.

## 7. Limitations

- Three seeds per model: the standard deviation is rough.
- One test sample of 3,000 tweets; differences of about 0.01 F1 are within sampling error.
- The learning rate and the best epoch were chosen on dev; seed 42 also took part in choosing the learning rate. The test scores are the unbiased estimate for these frozen models.
- The team's EDA computed descriptive statistics on the test split earlier (label counts, overlaps, CMI); no transformer decision used them (see the protocol).

