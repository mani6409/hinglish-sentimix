# BiLSTM final test-set evaluation (single, frozen-model run)

*Evaluates the nine frozen checkpoints (raw / clean / normalized x seeds 42, 43, 44) on the official SentiMix Hinglish test set (3,000 tweets). Protocol: `docs/bilstm_final_evaluation_protocol.md`. Produced by `python -m src.evaluate_bilstm_test --confirm-final-test` on 2026-11-17T02:29:46+00:00.*

**Nothing was selected, tuned or changed based on these results.** The checkpoints, preprocessing and metrics were frozen before the test labels were read. This report states measurements only; no preprocessing mode is declared the winner, and any difference smaller than the seed-to-seed spread must not be interpreted as a real difference.

## 1. Primary results: mean ± sample standard deviation over seeds 42, 43, 44

| Mode | Macro F1 | Weighted F1 |
|---|---|---|
| raw | **0.6615 ± 0.0077** | **0.6603 ± 0.0066** |
| clean | **0.6716 ± 0.0030** | **0.6681 ± 0.0023** |
| normalized | **0.6589 ± 0.0061** | **0.6566 ± 0.0059** |

Per-seed values (seed 42 / 43 / 44):

| Mode | Macro F1 | Weighted F1 |
|---|---|---|
| raw | 0.6532 / 0.6684 / 0.6631 | 0.6533 / 0.6665 / 0.6609 |
| clean | 0.6718 / 0.6685 / 0.6745 | 0.6679 / 0.6660 / 0.6705 |
| normalized | 0.6571 / 0.6539 / 0.6657 | 0.6551 / 0.6516 / 0.6631 |

Standard deviations are sample standard deviations (n = 3, ddof = 1) and are themselves rough with only three seeds.

## 2. Secondary metrics (mean ± sample std over seeds)

| Mode | Accuracy | Macro precision | Macro recall | Weighted precision | Weighted recall |
|---|---|---|---|---|---|
| raw | 0.6576 ± 0.0053 | 0.6863 ± 0.0093 | 0.6538 ± 0.0083 | 0.6797 ± 0.0076 | 0.6576 ± 0.0053 |
| clean | 0.6668 ± 0.0037 | 0.6767 ± 0.0034 | 0.6688 ± 0.0063 | 0.6718 ± 0.0024 | 0.6668 ± 0.0037 |
| normalized | 0.6533 ± 0.0065 | 0.6801 ± 0.0063 | 0.6515 ± 0.0066 | 0.6737 ± 0.0068 | 0.6533 ± 0.0065 |

Weighted recall equals accuracy by definition.

## 3. Class-wise precision, recall and F1 (mean ± sample std over seeds)

| Mode | Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|---|
| raw | negative | 900 | 0.7441 ± 0.0487 | 0.5830 ± 0.0728 | 0.6499 ± 0.0311 |
| raw | neutral | 1,100 | 0.5468 ± 0.0107 | 0.6942 ± 0.0350 | 0.6113 ± 0.0100 |
| raw | positive | 1,000 | 0.7679 ± 0.0134 | 0.6843 ± 0.0196 | 0.7235 ± 0.0058 |
| clean | negative | 900 | 0.7208 ± 0.0203 | 0.6756 ± 0.0470 | 0.6963 ± 0.0173 |
| clean | neutral | 1,100 | 0.5728 ± 0.0150 | 0.6161 ± 0.0341 | 0.5930 ± 0.0078 |
| clean | positive | 1,000 | 0.7365 ± 0.0056 | 0.7147 ± 0.0160 | 0.7254 ± 0.0088 |
| normalized | negative | 900 | 0.7352 ± 0.0333 | 0.6141 ± 0.0350 | 0.6681 ± 0.0105 |
| normalized | neutral | 1,100 | 0.5431 ± 0.0131 | 0.6676 ± 0.0285 | 0.5985 ± 0.0033 |
| normalized | positive | 1,000 | 0.7620 ± 0.0624 | 0.6730 ± 0.0762 | 0.7102 ± 0.0195 |

## 4. Every run

| Run | Accuracy | Macro P | Macro R | Macro F1 | Weighted P | Weighted R | Weighted F1 |
|---|---|---|---|---|---|---|---|
| raw seed 42 | 0.6523 | 0.6954 | 0.6446 | 0.6532 | 0.6866 | 0.6523 | 0.6533 |
| raw seed 43 | 0.6630 | 0.6865 | 0.6607 | 0.6684 | 0.6808 | 0.6630 | 0.6665 |
| raw seed 44 | 0.6573 | 0.6768 | 0.6562 | 0.6631 | 0.6716 | 0.6573 | 0.6609 |
| clean seed 42 | 0.6663 | 0.6755 | 0.6690 | 0.6718 | 0.6705 | 0.6663 | 0.6679 |
| clean seed 43 | 0.6633 | 0.6806 | 0.6624 | 0.6685 | 0.6745 | 0.6633 | 0.6660 |
| clean seed 44 | 0.6707 | 0.6741 | 0.6749 | 0.6745 | 0.6703 | 0.6707 | 0.6705 |
| normalized seed 42 | 0.6537 | 0.6731 | 0.6515 | 0.6571 | 0.6663 | 0.6537 | 0.6551 |
| normalized seed 43 | 0.6467 | 0.6852 | 0.6450 | 0.6539 | 0.6796 | 0.6467 | 0.6516 |
| normalized seed 44 | 0.6597 | 0.6820 | 0.6581 | 0.6657 | 0.6751 | 0.6597 | 0.6631 |

## 5. Confusion matrices

Rows = true class, columns = predicted class, order negative / neutral / positive. Each matrix is the **sum of the three seeds' matrices** (9,000 predictions per mode); percentages are row-normalised. Per-run matrices are in `confusion_matrices.json`.

**raw**

| | pred negative | pred neutral | pred positive | Row total |
|---|---|---|---|---|
| true negative | 1,574 (58.3%) | 1,027 (38.0%) | 99 (3.7%) | 2,700 |
| true neutral | 486 (14.7%) | 2,291 (69.4%) | 523 (15.8%) | 3,300 |
| true positive | 72 (2.4%) | 875 (29.2%) | 2,053 (68.4%) | 3,000 |

**clean**

| | pred negative | pred neutral | pred positive | Row total |
|---|---|---|---|---|
| true negative | 1,824 (67.6%) | 759 (28.1%) | 117 (4.3%) | 2,700 |
| true neutral | 617 (18.7%) | 2,033 (61.6%) | 650 (19.7%) | 3,300 |
| true positive | 94 (3.1%) | 762 (25.4%) | 2,144 (71.5%) | 3,000 |

**normalized**

| | pred negative | pred neutral | pred positive | Row total |
|---|---|---|---|---|
| true negative | 1,658 (61.4%) | 941 (34.9%) | 101 (3.7%) | 2,700 |
| true neutral | 540 (16.4%) | 2,203 (66.8%) | 557 (16.9%) | 3,300 |
| true positive | 64 (2.1%) | 917 (30.6%) | 2,019 (67.3%) | 3,000 |

Figures: `figures/bilstm_final_test_confusion_raw.png`, `figures/bilstm_final_test_confusion_clean.png`, `figures/bilstm_final_test_confusion_normalized.png`, `figures/bilstm_final_test_confusion_all_modes.png`.

## 6. Integrity and procedure

- All hashes below are MD5. The nine checkpoints and the unlabeled test file were verified **before** inference; the labels file was verified and read for the **first time only after** all nine prediction files had been written and validated; every hashed input was verified **again after** evaluation and was unchanged.
- Predictions were joined to the gold labels by **UID**, never by row order. Metrics were computed from the saved prediction files.
- Full hashes, timestamps of each stage, the environment and output hashes are in `integrity_manifest.json`.

| Item | MD5 (before = after) |
|---|---|
| checkpoint raw seed 42 | `45f166e48d73532f89c3e044c54c80fc` |
| checkpoint raw seed 43 | `540987f2af85d646c368572f0b07fd61` |
| checkpoint raw seed 44 | `a8f54c8cd8836155c8d7ea35b6806895` |
| checkpoint clean seed 42 | `190bc0c38b5da17fd4515439f30f0d32` |
| checkpoint clean seed 43 | `a481a1a2eb7159ad8ee7bfbe996dd6ad` |
| checkpoint clean seed 44 | `65d71b5961eeaff0e07141b1e2023a67` |
| checkpoint normalized seed 42 | `afc64d40cec38a52d40672a23c3894fa` |
| checkpoint normalized seed 43 | `d1b162e07e1d08fbddbd6ad619b5affb` |
| checkpoint normalized seed 44 | `d56fd14f34725e74d9dc32e403a4f912` |
| test file (unlabeled) | `4706975d6fa025634c176111d3271487` |
| test file (labels) | `60199636a5da7e8195f0db359332163e` |

## 7. Limitations

- Three seeds per mode: the standard deviation is rough, and differences between modes cannot be judged from the means alone.
- The test set is one fixed sample of 3,000 tweets; scores carry sampling error of about ±0.01 on F1, which this report does not estimate.
- The team's EDA computed descriptive statistics on the test split earlier (label counts, overlap with train and dev, CMI). No model decision used them (see the protocol).
- Dev scores used for early stopping are optimistic; test scores are the unbiased estimate for these frozen models.

