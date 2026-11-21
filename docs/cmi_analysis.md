# CMI bucket analysis (issue #3)

Reproduce with `python -m src.cmi_evaluation` (about 4 seconds). Numbers: `results/cmi_bucket_scores.csv` (scores per bucket) and `results/cmi_bucket_pairwise.csv` (differences between buckets). Tests: `tests/test_cmi_evaluation.py`, `tests/test_uid_checks.py`.

## Question
Does the amount of code-mixing in a tweet change how well a model classifies its sentiment?

## Method
1. **Buckets.** CMI per tweet comes from `results/cmi_per_tweet.csv` (standard formula, `src/cmi.py`). The two cut points are the 1/3 and 2/3 quantiles of CMI on the **training** set: 23.81 and 34.78. This reproduces the terciles documented in `docs/eda_part2.md`. Dev tweets are placed into the buckets with these train cut points, so no dev information is used to define them. The CMI table is read with its test rows dropped chunk by chunk, so the test rows are never used. Bucket sizes on dev: low 1,018, medium 981, high 1,001.
2. **Join.** Each file `results/predictions/*_dev.csv` is joined to the CMI table by `uid`. Before joining, `src/uid_checks.py` verifies that the prediction file and the CMI dev rows contain exactly the same uids, each exactly once; any duplicate, missing or unexpected uid stops the run with an error. The CMI file itself is also checked for duplicate uids. The join then keeps all 3,000 rows (asserted).
3. **Scores.** Accuracy, precision and recall per class, macro F1 and weighted F1 (official metric) for all dev tweets and for each bucket.
4. **Bootstrap.** Each resample draws, with replacement, as many tweets as the bucket holds, and recomputes weighted F1 (1,000 resamples; each prediction file has its own random generator seeded from 42 and the file name, so adding a file never changes the intervals of the others). The weighted F1 used inside the bootstrap is a fast exact version; a test checks that it equals `compute_metrics` on random data, including a class that is never predicted. Tests also check that a resample has the bucket's size and draws with replacement.
5. **Pairwise differences.** For low - medium, medium - high and low - high we take the difference of the two buckets' resampled scores and report its 95% interval. The two buckets are different tweets, so they are resampled independently. An interval that contains 0 means the data do not show a difference.

All intervals are uncorrected for the number of comparisons: 12 prediction files x 3 comparisons = 36 intervals.

## Results (dev set, weighted F1)

**TF-IDF baselines** (bucket scores, and pairwise differences with 95% interval)

| Model | Low | Medium | High | Low - Medium | Medium - High | Low - High |
|---|---|---|---|---|---|---|
| Word 1-2 | 0.602 | 0.609 | 0.621 | -0.007 (-0.051, +0.035) | -0.011 (-0.059, +0.031) | -0.018 (-0.064, +0.023) |
| Char 2-5 | 0.611 | 0.598 | 0.625 | +0.013 (-0.030, +0.058) | -0.027 (-0.073, +0.018) | -0.015 (-0.058, +0.027) |
| Word + char | 0.596 | 0.605 | 0.630 | -0.009 (-0.053, +0.036) | -0.025 (-0.068, +0.018) | -0.034 (-0.078, +0.009) |

**BiLSTM** (word + character; mean ± sample standard deviation over seeds 42, 43, 44, from `results/cmi_bucket_group_summary.csv`; the three seeds use the same tweets, so they are not independent evidence)

| Preprocessing | Low | Medium | High | Low - High: mean (range over seeds) |
|---|---|---|---|---|
| Raw | 0.611 ± 0.008 | 0.602 ± 0.005 | 0.617 ± 0.008 | -0.007 (-0.010 to -0.002) |
| Clean | 0.615 ± 0.008 | 0.609 ± 0.009 | 0.611 ± 0.004 | +0.004 (+0.001 to +0.010) |
| Normalized | 0.597 ± 0.013 | 0.607 ± 0.007 | 0.616 ± 0.008 | -0.019 (-0.044 to -0.004) |

Per-seed numbers and their intervals are in the CSV files.

## What this shows
- **No clear evidence of a relationship between CMI bucket and performance.** Of the 36 pairwise intervals, 35 contain 0. The one that does not (BiLSTM, normalized text, seed 43, low - high = -0.044, interval -0.085 to -0.0005, only just excluding 0) is about what chance gives: with 36 intervals at 95% we expect about 2 to exclude 0 even if no difference exists. The other two normalized seeds point the same way (-0.010 and -0.004) but their intervals contain 0.
- **Direction.** High CMI is nominally the best bucket for all three TF-IDF models and for BiLSTM with raw and normalized text, but the gaps (about 0.01 to 0.03) are smaller than the sampling error of a 1,000-tweet bucket (about 0.03 either side).
- **Absence of evidence is not evidence of absence.** With 1,000 tweets per bucket we could only detect a difference of roughly 0.04 to 0.05 or more. A smaller real effect cannot be ruled out.
- **Why this may be so.** Every tweet in this data set is mixed (see `docs/eda_part2.md`), so "low" means moderately mixed, not single-language. This is a possible reason, not a tested one.
- **BiLSTM compared with TF-IDF.** The two model families show the same picture (no consistent trend), and the BiLSTM does not outperform TF-IDF on dev.

## Limits
- Dev set only; the test set is not used here.
- Buckets are equal-sized thirds of a distribution in which every tweet is mixed.
- Intervals are not corrected for multiple comparisons.
- Our CMI averages (29.8) differ from the task paper's (25.3). The ranking of tweets is what the buckets use, so the conclusion does not depend on the absolute values.
- Transformer and LLM predictions are not available yet.

## Next
Rerun `python -m src.cmi_evaluation` when `results/predictions/<model>_dev.csv` files (columns `uid,label,pred`) exist for the transformer models, to compare the effect of CMI across model families.
