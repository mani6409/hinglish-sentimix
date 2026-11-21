# BiLSTM final analysis (descriptive, exploratory)

Everything below is computed from the **saved** predictions of the nine frozen runs (3 preprocessing modes x seeds 42, 43, 44) and the official gold labels. Nothing was retrained, no model was run on the test set again, and no result here selects a model, a seed or a setting. The single pre-specified test evaluation is in `results/bilstm_final_test/`; this analysis is **exploratory** and was written after those results were known. Earlier descriptive test access by the team's EDA scripts is disclosed in section 8 of `docs/bilstm_final_evaluation_protocol.md`.

## 1. Final results (mean ± sample std over seeds 42, 43, 44)

| mode | test macro F1 | test weighted F1 | test accuracy | dev macro F1 | dev weighted F1 | dev accuracy |
|---|---|---|---|---|---|---|
| raw | 0.6615 ± 0.0077 | 0.6603 ± 0.0066 | 0.6576 ± 0.0053 | 0.6143 ± 0.0056 | 0.6105 ± 0.0041 | 0.6096 ± 0.0048 |
| clean | 0.6716 ± 0.0030 | 0.6681 ± 0.0023 | 0.6668 ± 0.0037 | 0.6172 ± 0.0022 | 0.6118 ± 0.0015 | 0.6133 ± 0.0028 |
| normalized | 0.6589 ± 0.0061 | 0.6566 ± 0.0059 | 0.6533 ± 0.0065 | 0.6105 ± 0.0004 | 0.6071 ± 0.0007 | 0.6058 ± 0.0008 |

Dev numbers are the best epoch per run (early stopping on dev weighted F1), so they are optimistically selected; test numbers are the unbiased estimate for these frozen models. The official metric of the task is weighted F1.

## 2. Statistical analysis

**Two kinds of uncertainty are reported separately.**

* **Seed variation** (training randomness): the sample standard deviation across the three seeds of one mode, and the range. No resampling is involved.
* **Test-sampling uncertainty** (the test set is one fixed sample of 3,000 tweets): a **paired bootstrap** over tweets, 10,000 resamples, fixed seed 20261116. In every resample the *same* multiset of tweet uids is used for all nine runs, so each comparison is paired. For each resample the macro F1 of every run is computed, averaged over the three seeds of a mode, and the modes are compared. The interval therefore describes the uncertainty from the test sample **for these three trained seeds**; it does not include the uncertainty from training new seeds (that is what the seed variation shows, roughly).

### 2.1 Seed variation

| split | mode | metric | s42 | s43 | s44 | mean | sd (ddof=1) | range |
|---|---|---|---|---|---|---|---|---|
| dev | raw | macro_f1 | 0.607817 | 0.617598 | 0.617590 | 0.614335 | 0.005645 | 0.009781 |
| dev | raw | weighted_f1 | 0.605844 | 0.613052 | 0.612722 | 0.610539 | 0.004069 | 0.007208 |
| dev | raw | accuracy | 0.604000 | 0.612333 | 0.612333 | 0.609556 | 0.004811 | 0.008333 |
| dev | clean | macro_f1 | 0.619498 | 0.615216 | 0.616755 | 0.617156 | 0.002169 | 0.004283 |
| dev | clean | weighted_f1 | 0.613591 | 0.610929 | 0.610922 | 0.611814 | 0.001539 | 0.002669 |
| dev | clean | accuracy | 0.616000 | 0.610333 | 0.613667 | 0.613333 | 0.002848 | 0.005667 |
| dev | normalized | macro_f1 | 0.610971 | 0.610387 | 0.610125 | 0.610494 | 0.000433 | 0.000846 |
| dev | normalized | weighted_f1 | 0.607406 | 0.607516 | 0.606303 | 0.607075 | 0.000671 | 0.001213 |
| dev | normalized | accuracy | 0.606667 | 0.605000 | 0.605667 | 0.605778 | 0.000839 | 0.001667 |
| test | raw | macro_f1 | 0.653204 | 0.668369 | 0.663068 | 0.661547 | 0.007696 | 0.015166 |
| test | raw | weighted_f1 | 0.653311 | 0.666546 | 0.660929 | 0.660262 | 0.006642 | 0.013235 |
| test | raw | accuracy | 0.652333 | 0.663000 | 0.657333 | 0.657556 | 0.005337 | 0.010667 |
| test | clean | macro_f1 | 0.671753 | 0.668484 | 0.674492 | 0.671576 | 0.003008 | 0.006008 |
| test | clean | weighted_f1 | 0.667941 | 0.665970 | 0.670484 | 0.668132 | 0.002263 | 0.004514 |
| test | clean | accuracy | 0.666333 | 0.663333 | 0.670667 | 0.666778 | 0.003687 | 0.007333 |
| test | normalized | macro_f1 | 0.657131 | 0.653917 | 0.665734 | 0.658927 | 0.006110 | 0.011818 |
| test | normalized | weighted_f1 | 0.655107 | 0.651581 | 0.663134 | 0.656607 | 0.005921 | 0.011553 |
| test | normalized | accuracy | 0.653667 | 0.646667 | 0.659667 | 0.653333 | 0.006506 | 0.013000 |

### 2.2 Bootstrap intervals for the test set

**Macro F1 per mode (mean over seeds, 95% percentile interval from the test-sampling bootstrap)**

| mode | mean | 95% interval |
|---|---|---|
| raw | 0.6615 | [0.6467, 0.6760] |
| clean | 0.6716 | [0.6573, 0.6855] |
| normalized | 0.6589 | [0.6447, 0.6729] |

**Macro F1: pairwise differences a − b (mean over seeds)**

| comparison | difference | 95% interval | Bonferroni 98.3% interval (3 comparisons) | 95% interval excludes 0 | same-seed differences (s42, s43, s44) |
|---|---|---|---|---|---|
| clean − raw | +0.0100 | [+0.0003, +0.0194] | [-0.0017, +0.0211] | yes | +0.0185, +0.0001, +0.0114 |
| normalized − raw | -0.0026 | [-0.0116, +0.0063] | [-0.0136, +0.0085] | no | +0.0039, -0.0145, +0.0027 |
| normalized − clean | -0.0126 | [-0.0212, -0.0040] | [-0.0230, -0.0018] | yes | -0.0146, -0.0146, -0.0088 |

**Weighted F1 per mode (mean over seeds, 95% percentile interval from the test-sampling bootstrap)**

| mode | mean | 95% interval |
|---|---|---|
| raw | 0.6603 | [0.6457, 0.6748] |
| clean | 0.6681 | [0.6536, 0.6824] |
| normalized | 0.6566 | [0.6425, 0.6708] |

**Weighted F1: pairwise differences a − b (mean over seeds)**

| comparison | difference | 95% interval | Bonferroni 98.3% interval (3 comparisons) | 95% interval excludes 0 | same-seed differences (s42, s43, s44) |
|---|---|---|---|---|---|
| clean − raw | +0.0079 | [-0.0019, +0.0174] | [-0.0042, +0.0193] | no | +0.0146, -0.0006, +0.0096 |
| normalized − raw | -0.0037 | [-0.0127, +0.0053] | [-0.0146, +0.0074] | no | +0.0018, -0.0150, +0.0022 |
| normalized − clean | -0.0115 | [-0.0202, -0.0027] | [-0.0221, -0.0005] | yes | -0.0128, -0.0144, -0.0074 |

### 2.3 The same intervals on the development set (for reference)

| macro F1 comparison | difference | 95% interval | Bonferroni 98.3% interval |
|---|---|---|---|
| clean − raw | +0.0028 | [-0.0066, +0.0121] | [-0.0087, +0.0143] |
| normalized − raw | -0.0038 | [-0.0130, +0.0054] | [-0.0149, +0.0074] |
| normalized − clean | -0.0067 | [-0.0155, +0.0022] | [-0.0173, +0.0042] |

### 2.4 How to read these intervals

* A 95% interval is the range of differences that the resampling procedure produced for 95% of resamples. It is a statement about the test sample, conditional on the three trained models per mode.
* There are three pairwise comparisons on the same data. The unadjusted intervals are the main presentation; the Bonferroni intervals show how much wider they become when the three comparisons are treated as one family. Neither is a hypothesis test and **no statistical significance is claimed**.
* An interval that contains zero means the data do not separate the modes; it is **not** evidence that they are equivalent. An interval that excludes zero says the difference is unlikely to be only test-sampling noise for these three seeds; it does not show that a new seed would reproduce it, because seed variation is separate (section 2.1).
* The comparisons were chosen after seeing the dev results; the test intervals do not select anything.

**Test macro F1, in numbers:**

* **clean − raw** (macro F1): difference +0.0100, 95% interval [+0.0003, +0.0194] — excludes zero; seed-to-seed range within the two modes: 0.0060 / 0.0152.
* **normalized − raw** (macro F1): difference -0.0026, 95% interval [-0.0116, +0.0063] — contains zero; seed-to-seed range within the two modes: 0.0118 / 0.0152.
* **normalized − clean** (macro F1): difference -0.0126, 95% interval [-0.0212, -0.0040] — excludes zero; seed-to-seed range within the two modes: 0.0118 / 0.0060.

**Evidence per comparison (test, both primary metrics):**

* **clean − raw**: macro F1 interval excludes 0 (Bonferroni: contains 0); weighted F1 interval contains 0 (Bonferroni: contains 0); the three same-seed differences in macro F1 all have the same sign (range +0.0001 to +0.0185).
* **normalized − raw**: macro F1 interval contains 0 (Bonferroni: contains 0); weighted F1 interval contains 0 (Bonferroni: contains 0); the three same-seed differences in macro F1 do not all have the same sign (range -0.0145 to +0.0039).
* **normalized − clean**: macro F1 interval excludes 0 (Bonferroni: excludes 0); weighted F1 interval excludes 0 (Bonferroni: excludes 0); the three same-seed differences in macro F1 all have the same sign (range -0.0146 to -0.0088).

Comparisons for which all of the following hold: the unadjusted and Bonferroni intervals exclude 0 on **both** macro and weighted F1, and the three same-seed macro-F1 differences share a sign: normalized − clean. Comparisons not listed are not separated by these data. This describes the strength of evidence in this sample; it is not a significance test and does not establish that a different set of seeds or test tweets would give the same ordering.

## 3. Error analysis (computed counts only)

Class order: negative, neutral, positive. Rows = true class, columns = predicted class. Matrices are summed over the three seeds of a mode (3 x 3,000 predictions).

### 3.1 Confusion matrices (test)

**raw**

| true \ pred | negative | neutral | positive |
|---|---|---|---|
| negative | 1574 | 1027 | 99 |
| neutral | 486 | 2291 | 523 |
| positive | 72 | 875 | 2053 |

**clean**

| true \ pred | negative | neutral | positive |
|---|---|---|---|
| negative | 1824 | 759 | 117 |
| neutral | 617 | 2033 | 650 |
| positive | 94 | 762 | 2144 |

**normalized**

| true \ pred | negative | neutral | positive |
|---|---|---|---|
| negative | 1658 | 941 | 101 |
| neutral | 540 | 2203 | 557 |
| positive | 64 | 917 | 2019 |

### 3.2 Confusion matrices (dev)

**raw**

| true \ pred | negative | neutral | positive |
|---|---|---|---|
| negative | 1650 | 877 | 143 |
| neutral | 674 | 1946 | 764 |
| positive | 186 | 870 | 1890 |

**clean**

| true \ pred | negative | neutral | positive |
|---|---|---|---|
| negative | 1780 | 721 | 169 |
| neutral | 795 | 1743 | 846 |
| positive | 214 | 735 | 1997 |

**normalized**

| true \ pred | negative | neutral | positive |
|---|---|---|---|
| negative | 1617 | 890 | 163 |
| neutral | 664 | 1963 | 757 |
| positive | 196 | 878 | 1872 |

### 3.3 Per-class precision, recall and F1 (mean ± sd over seeds)

**test**

| mode | class | support | precision | recall | F1 |
|---|---|---|---|---|---|
| raw | negative | 900 | 0.7441 ± 0.0487 | 0.5830 ± 0.0728 | 0.6499 ± 0.0311 |
| raw | neutral | 1100 | 0.5468 ± 0.0107 | 0.6942 ± 0.0350 | 0.6113 ± 0.0100 |
| raw | positive | 1000 | 0.7679 ± 0.0134 | 0.6843 ± 0.0196 | 0.7235 ± 0.0058 |
| clean | negative | 900 | 0.7208 ± 0.0203 | 0.6756 ± 0.0470 | 0.6963 ± 0.0173 |
| clean | neutral | 1100 | 0.5728 ± 0.0150 | 0.6161 ± 0.0341 | 0.5930 ± 0.0078 |
| clean | positive | 1000 | 0.7365 ± 0.0056 | 0.7147 ± 0.0160 | 0.7254 ± 0.0088 |
| normalized | negative | 900 | 0.7352 ± 0.0333 | 0.6141 ± 0.0350 | 0.6681 ± 0.0105 |
| normalized | neutral | 1100 | 0.5431 ± 0.0131 | 0.6676 ± 0.0285 | 0.5985 ± 0.0033 |
| normalized | positive | 1000 | 0.7620 ± 0.0624 | 0.6730 ± 0.0762 | 0.7102 ± 0.0195 |

**dev**

| mode | class | support | precision | recall | F1 |
|---|---|---|---|---|---|
| raw | negative | 890 | 0.6615 ± 0.0349 | 0.6180 ± 0.0818 | 0.6353 ± 0.0309 |
| raw | neutral | 1128 | 0.5280 ± 0.0148 | 0.5751 ± 0.0434 | 0.5495 ± 0.0111 |
| raw | positive | 982 | 0.6758 ± 0.0052 | 0.6415 ± 0.0098 | 0.6582 ± 0.0029 |
| clean | negative | 890 | 0.6394 ± 0.0210 | 0.6667 ± 0.0406 | 0.6519 ± 0.0125 |
| clean | neutral | 1128 | 0.5455 ± 0.0112 | 0.5151 ± 0.0319 | 0.5292 ± 0.0112 |
| clean | positive | 982 | 0.6631 ± 0.0099 | 0.6779 ± 0.0082 | 0.6704 ± 0.0057 |
| normalized | negative | 890 | 0.6535 ± 0.0200 | 0.6056 ± 0.0198 | 0.6283 ± 0.0063 |
| normalized | neutral | 1128 | 0.5265 ± 0.0081 | 0.5801 ± 0.0374 | 0.5515 ± 0.0136 |
| normalized | positive | 982 | 0.6745 ± 0.0415 | 0.6354 ± 0.0619 | 0.6517 ± 0.0191 |

### 3.4 Where the errors are (neutral-class analysis)

Definitions, all per run, then mean ± sd over seeds: an *error* is a tweet whose predicted label differs from the gold label. **Neutral-involved error** = gold is neutral **or** predicted is neutral (a negative/positive confusion is the only kind that never involves neutral). **Gold-neutral error** = gold neutral, predicted negative/positive. **Predicted-neutral error** = predicted neutral, gold negative/positive. **Negative↔positive flip** = gold and predicted are the two non-neutral classes, in either order. The three kinds are disjoint and add up to 100%; neutral-involved = gold-neutral + predicted-neutral.

**test** (3,000 tweets)

| mode | errors per run | neutral-involved | gold-neutral | predicted-neutral | negative↔positive flips |
|---|---|---|---|---|---|
| raw | 1027.3 ± 16.0 | 94.4 ± 0.9% | 32.8 ± 4.0% | 61.7 ± 4.7% | 5.6 ± 0.9% |
| clean | 999.7 ± 11.1 | 93.0 ± 1.1% | 42.3 ± 4.2% | 50.7 ± 5.4% | 7.0 ± 1.1% |
| normalized | 1040.0 ± 19.5 | 94.7 ± 1.8% | 35.2 ± 3.5% | 59.5 ± 5.0% | 5.3 ± 1.8% |

**dev** (3,000 tweets)

| mode | errors per run | neutral-involved | gold-neutral | predicted-neutral | negative↔positive flips |
|---|---|---|---|---|---|
| raw | 1171.3 ± 14.4 | 90.6 ± 1.6% | 41.0 ± 4.7% | 49.7 ± 6.2% | 9.4 ± 1.6% |
| clean | 1160.0 ± 8.5 | 89.0 ± 0.9% | 47.2 ± 3.4% | 41.8 ± 4.3% | 11.0 ± 0.9% |
| normalized | 1182.7 ± 2.5 | 89.9 ± 1.2% | 40.1 ± 3.6% | 49.8 ± 4.5% | 10.1 ± 1.2% |

### 3.5 Misclassification patterns by tweet property

Accuracy and macro F1 (mean over seeds) inside code-mixing-index buckets (CMI terciles of the **train** split, edges 23.81 and 34.783; low <= 23.81, high > 34.783) and tweet-length buckets (word-count terciles of the train split, edges 25.0 and 30.0). These are descriptions of where the models do better or worse; they are not causal explanations.

**test, by cmi**

| mode | bucket | n | accuracy | macro F1 |
|---|---|---|---|---|
| raw | low | 1065 | 0.6404 ± 0.0062 | 0.6459 ± 0.0065 |
| raw | medium | 932 | 0.6617 ± 0.0076 | 0.6688 ± 0.0091 |
| raw | high | 1003 | 0.6720 ± 0.0085 | 0.6654 ± 0.0103 |
| clean | low | 1065 | 0.6482 ± 0.0044 | 0.6530 ± 0.0048 |
| clean | medium | 932 | 0.6638 ± 0.0070 | 0.6696 ± 0.0059 |
| clean | high | 1003 | 0.6893 ± 0.0087 | 0.6870 ± 0.0101 |
| normalized | low | 1065 | 0.6438 ± 0.0053 | 0.6501 ± 0.0038 |
| normalized | medium | 932 | 0.6416 ± 0.0060 | 0.6491 ± 0.0069 |
| normalized | high | 1003 | 0.6743 ± 0.0190 | 0.6734 ± 0.0175 |

**dev, by cmi**

| mode | bucket | n | accuracy | macro F1 |
|---|---|---|---|---|
| raw | low | 1018 | 0.6081 ± 0.0112 | 0.6111 ± 0.0083 |
| raw | medium | 981 | 0.6028 ± 0.0039 | 0.6079 ± 0.0040 |
| raw | high | 1001 | 0.6177 ± 0.0071 | 0.6142 ± 0.0111 |
| clean | low | 1018 | 0.6146 ± 0.0099 | 0.6153 ± 0.0079 |
| clean | medium | 981 | 0.6137 ± 0.0054 | 0.6171 ± 0.0070 |
| clean | high | 1001 | 0.6117 ± 0.0038 | 0.6099 ± 0.0047 |
| normalized | low | 1018 | 0.5940 ± 0.0133 | 0.5975 ± 0.0137 |
| normalized | medium | 981 | 0.6069 ± 0.0060 | 0.6122 ± 0.0062 |
| normalized | high | 1001 | 0.6167 ± 0.0065 | 0.6131 ± 0.0071 |

**test, by length**

| mode | bucket | n | accuracy | macro F1 |
|---|---|---|---|---|
| raw | short | 1119 | 0.6729 ± 0.0071 | 0.6728 ± 0.0116 |
| raw | medium | 1061 | 0.6616 ± 0.0090 | 0.6655 ± 0.0098 |
| raw | long | 820 | 0.6313 ± 0.0025 | 0.6360 ± 0.0041 |
| clean | short | 1119 | 0.6795 ± 0.0027 | 0.6824 ± 0.0021 |
| clean | medium | 1061 | 0.6861 ± 0.0057 | 0.6883 ± 0.0043 |
| clean | long | 820 | 0.6244 ± 0.0042 | 0.6307 ± 0.0036 |
| normalized | short | 1119 | 0.6634 ± 0.0121 | 0.6671 ± 0.0126 |
| normalized | medium | 1061 | 0.6648 ± 0.0054 | 0.6693 ± 0.0035 |
| normalized | long | 820 | 0.6248 ± 0.0087 | 0.6299 ± 0.0103 |

**dev, by length**

| mode | bucket | n | accuracy | macro F1 |
|---|---|---|---|---|
| raw | short | 1103 | 0.6132 ± 0.0084 | 0.6120 ± 0.0099 |
| raw | medium | 1034 | 0.6035 ± 0.0079 | 0.6091 ± 0.0079 |
| raw | long | 863 | 0.6122 ± 0.0162 | 0.6132 ± 0.0163 |
| clean | short | 1103 | 0.6129 ± 0.0009 | 0.6115 ± 0.0015 |
| clean | medium | 1034 | 0.6122 ± 0.0054 | 0.6163 ± 0.0063 |
| clean | long | 863 | 0.6153 ± 0.0141 | 0.6139 ± 0.0119 |
| normalized | short | 1103 | 0.6120 ± 0.0107 | 0.6097 ± 0.0101 |
| normalized | medium | 1034 | 0.6025 ± 0.0070 | 0.6078 ± 0.0061 |
| normalized | long | 863 | 0.6018 ± 0.0082 | 0.6032 ± 0.0109 |

Highest- and lowest-accuracy bucket per mode (mean over seeds): **test, cmi**: raw: highest high (0.672), lowest low (0.640); clean: highest high (0.689), lowest low (0.648); normalized: highest high (0.674), lowest medium (0.642) | **dev, cmi**: raw: highest high (0.618), lowest medium (0.603); clean: highest low (0.615), lowest high (0.612); normalized: highest high (0.617), lowest low (0.594) | **test, length**: raw: highest short (0.673), lowest long (0.631); clean: highest medium (0.686), lowest long (0.624); normalized: highest medium (0.665), lowest long (0.625) | **dev, length**: raw: highest short (0.613), lowest medium (0.603); clean: highest long (0.615), lowest medium (0.612); normalized: highest short (0.612), lowest long (0.602).

### 3.6 Examples

383 test tweets are misclassified by **all nine** runs. 10 of them were drawn at random (fixed seed 20261116; no hand selection). The tweets come from the SemEval-2020 Task 9 data, released under CC BY 4.0 (Patwa et al., 2020); @mentions are replaced by USER and links by URL, and no ids are shown. No reason for the error is given, because none was analysed.

| # | gold | predicted (all nine) | error kind | CMI bucket | tweet (masked) |
|---|---|---|---|---|---|
| 1 | negative | neutral | predicted-neutral error | high | what's wrong with skinny guys who wear hulk size t shirts ? phir kehte hein humein kuch kaho na . bhai tum chota size â€¦ URL |
| 2 | negative | neutral/positive | negative/positive flip | high | USER Everytime HASHTAG acts up this Bengali song Bhoot amaar Poot Petni Amaar Jhi Ram Lakkhan Sathe Ache Korbe ama â€¦ |
| 3 | positive | neutral | predicted-neutral error | medium | USER tute nii koi rishte purane kisi naye ke aajane se Jata hu mai mujhe tu jane de Kyo pareshan h me â€¦ URL |
| 4 | negative | neutral/positive | negative/positive flip | low | USER This kind of secculurism will nvr give any attention to our ðŸ‡®ðŸ‡³ indians we love USER thougtpro â€¦ URL |
| 5 | neutral | positive | gold-neutral error | medium | USER I wish my dad still alive miss him a lot and I love ur dad's response Khich k rekh |
| 6 | negative | neutral | predicted-neutral error | medium | USER patchi tum kabhi kamiyab nahi hogay barray matches mai hamesha zalil hogay dusro ka haaq markr jo aat â€¦ URL |
| 7 | neutral | positive | gold-neutral error | high | USER Aaj ki HASHTAG thank you HASHTAG lovely meeting you all HASHTAG HASHTAG HASHTAG HASHTAG HASHTAG â€¦ |
| 8 | negative | neutral | predicted-neutral error | low | Teko kya problem hai bhai hijab say yahi to problem hi bjp ki bhakto ki hamare mazhab may neeji cheezo may tang ada â€¦ URL |
| 9 | negative | neutral | predicted-neutral error | low | USER We need strictly good people in HR not the jhoote ma â€¦ URL |
| 10 | negative | neutral | predicted-neutral error | low | USER Bibi kabhi ao na water canon key samney khud bhi jo tumharey bap ki hakumat main har bat sey pehley use h â€¦ URL |

## 4. Investigation of the dev–test gap

Test macro F1 is higher than dev macro F1 for every mode:

| mode | dev macro F1 | test macro F1 | gap | test macro F1 re-weighted to the dev class mix |
|---|---|---|---|---|
| raw | 0.6143 | 0.6615 | +0.0472 | 0.6618 |
| clean | 0.6172 | 0.6716 | +0.0544 | 0.6711 |
| normalized | 0.6105 | 0.6589 | +0.0484 | 0.6590 |

### 4.1 Evaluation and data integrity checks

| check | result |
|---|---|
| dev and test uids unique | True / True |
| uid overlap dev∩test / train∩dev / train∩test | 0 / 0 / 0 |
| test label uids equal test tweet uids | True |
| gold labels identical across the nine runs (per split) | True |
| label order used everywhere | negative, neutral, positive |
| max |recomputed − saved| test metrics (5 metrics x 9 runs) | 0.00e+00 |
| max |recomputed − saved| dev metrics (3 metrics x 9 runs) | 0.00e+00 |
| max |sklearn macro F1 − project macro F1| (both splits, 9 runs) | 0.00e+00 |

Predictions and gold labels were joined by uid (never by row position); the same metric function is used for dev and test, and it reproduces the saved numbers of both. These checks make an implementation, alignment or label-mapping error unlikely as the cause of the gap.

### 4.2 Class distributions

| split | negative | neutral | positive |
|---|---|---|---|
| train | 29.3% | 37.6% | 33.1% |
| dev | 29.7% | 37.6% | 32.7% |
| test | 30.0% | 36.7% | 33.3% |

Share of predicted labels (mean over seeds):

| split | mode | negative | neutral | positive |
|---|---|---|---|---|
| dev | raw | 27.9% | 41.0% | 31.1% |
| dev | clean | 31.0% | 35.5% | 33.5% |
| dev | normalized | 27.5% | 41.5% | 31.0% |
| test | raw | 23.7% | 46.6% | 29.7% |
| test | clean | 28.2% | 39.5% | 32.3% |
| test | normalized | 25.1% | 45.1% | 29.7% |

The last column of the first table above re-weights each test tweet so that the gold-class mix equals the dev mix. The difference between that column and the plain test macro F1 is the part of the gap attributable to the class prior alone.

### 4.3 Tweet length, code mixing, and out-of-vocabulary words

| split | words mean | words median | words 90th pct | chars mean | chars median | CMI mean | CMI low/medium/high share |
|---|---|---|---|---|---|---|---|
| dev | 26.23 | 28.0 | 33.0 | 127.7 | 142.0 | 29.96 | 33.9% / 32.7% / 33.4% |
| test | 26.12 | 28.0 | 33.0 | 129.9 | 144.0 | 29.71 | 35.5% / 31.1% / 33.4% |

Out-of-vocabulary (OOV) rates against each mode's **train-only** word vocabulary (the vocabulary stored in the checkpoints; the same for all three seeds of a mode):

| mode | split | vocabulary size | OOV token rate | OOV type rate | tweets with ≥1 OOV word | OOV words per tweet |
|---|---|---|---|---|---|---|
| raw | dev | 15748 | 13.00% | 56.02% | 97.0% | 3.410 |
| raw | test | 15748 | 15.72% | 56.06% | 97.4% | 4.108 |
| clean | dev | 12984 | 10.86% | 47.20% | 81.6% | 2.112 |
| clean | test | 12984 | 14.14% | 47.44% | 92.2% | 2.849 |
| normalized | dev | 12914 | 10.76% | 47.12% | 81.4% | 2.093 |
| normalized | test | 12914 | 14.08% | 47.45% | 92.1% | 2.836 |

**Measured differences between dev and test** (computed from the tables above):

* Errors per run fall from 1171 to 1027 (raw), 1160 to 1000 (clean), 1183 to 1040 (normalized). Of that reduction, negative↔positive flips account for 53 of 144 (raw), 57 of 160 (clean), 65 of 143 (normalized); the remainder is fewer errors that involve neutral.
* F1 gain from dev to test by class (negative / neutral / positive): raw: +0.015 / +0.062 / +0.065; clean: +0.044 / +0.064 / +0.055; normalized: +0.040 / +0.047 / +0.058.
* OOV token rate is +2.7 percentage points (raw), +3.3 percentage points (clean), +3.3 percentage points (normalized) on test relative to dev. More unseen words would be expected to make test harder, not easier, so OOV does not account for the higher test scores.
* Mean tweet length is 26.2 words on dev and 26.1 on test; mean CMI is 30.0 and 29.7. These two properties are nearly the same in both splits.
* The models predict neutral more often on test than on dev (41.0% to 46.6% (raw), 35.5% to 39.5% (clean), 41.5% to 45.1% (normalized)) although neutral is a smaller share of the gold labels on test (37.6% on dev, 36.7% on test).

### 4.4 What this does and does not explain

* Re-weighting the test set to the dev class mix changes test macro F1 by -0.0005 to +0.0002, against gaps of +0.0472 to +0.0544. So the class prior accounts for little of the gap (see the table).
* Length and code mixing are nearly the same in both splits and OOV is higher on test (section 4.3); none of these accounts for the higher test scores. The differences listed above describe where the gain appears (fewer polarity flips and fewer neutral-related errors) but not why.
* **Not tested, and therefore unexplained:** whether the test tweets are intrinsically easier (annotation consistency, topic or time of collection), and whether the gap is specific to this model family. Checking the second would need test predictions from another model, which this analysis does not create.
* Nothing in this section was used to change, select or re-run any model.

## 5. Limitations

* Three seeds per mode: standard deviations are rough, and the bootstrap does not include training-seed uncertainty.
* One fixed test sample of 3,000 tweets; intervals are conditional on these nine trained models.
* Three pairwise comparisons were made on one data set; only the Bonferroni intervals address multiplicity, and no hypothesis test is performed.
* Dev scores are optimistically selected (best epoch by dev weighted F1); test scores are the unbiased estimates.
* The test set was described by the team's EDA before the evaluation (protocol section 8). This analysis is exploratory and post hoc.
* The dev–test gap remains partly unexplained (section 4.4).
* Results were verified on CPU only; a GPU can differ in the last digits.
* No claim is made that one preprocessing mode is better; the differences should be read against both the seed variation and the intervals above.

## 6. Reproducibility

* **Environment actually used for the evaluation, this analysis and the tests:** Python 3.12.5, torch 2.15.0+cpu, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, matplotlib 3.11.2, pytest 9.2.0 (listed in `requirements-bilstm.txt`; other combinations were not tested). The checkpoints record torch 2.15.0+cpu for training; the other library versions of the training runs were not recorded.
* **Hardware:** CPU only. Reproducibility was checked on CPU; a GPU can differ in the last digits.
* **Frozen artifacts:** the nine checkpoints (git-ignored, MD5 in `configs/bilstm_frozen_checkpoints.json`), the saved test predictions and the final-evaluation outputs. This analysis verifies all of them against their recorded hashes before it starts and again when it ends, and stops if one differs.
* **Commands** (repository root, in the environment above; the official data in `data/raw/`):

```
python -m pytest tests/test_analyze_bilstm_test.py tests/test_evaluate_bilstm_test.py
python -m src.analyze_bilstm_test --data-root data/raw      # needs the checkpoints; refuses to overwrite an existing results/bilstm_test_analysis
```

* The analysis uses the saved predictions, the official gold labels, the checkpoints' vocabularies and `results/cmi_per_tweet.csv`. It reads, and never changes, the frozen files, and trains and runs no model.
* Bootstrap: 10,000 resamples, seed 20261116; identical inputs give identical numbers.

## 7. Submission checklist

- [ ] Nine checkpoints unchanged (hash check passes in `src.analyze_bilstm_test` and `src.evaluate_bilstm_test --validate-only`)
- [ ] `results/bilstm_final_test/` and `configs/bilstm_frozen_checkpoints.json` unmodified since the evaluation
- [ ] Test split used once for the pre-specified evaluation; this analysis is labelled exploratory
- [ ] Test-set access by the EDA scripts disclosed (protocol section 8)
- [ ] Results reported as mean ± sample sd over three seeds, with the bootstrap intervals; no winner claimed beyond what section 2 supports
- [ ] Dev–test gap stated as partly unexplained (section 4.4)
- [ ] README: data location, install steps and commands work from a clean clone (README is updated after the team's pending PRs merge)
- [ ] No keys, tokens or personal data committed; checkpoints and raw data not committed unless the course requires it
- [ ] One GitHub link submitted for the team, pointing to the default branch with all parts merged
