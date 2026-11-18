# CMI bucket analysis (issue #3)

Reproduce with `python -m src.cmi_evaluation` (takes a few minutes because of the bootstrap). Numbers: `results/cmi_bucket_scores.csv`. Tests: `tests/test_cmi_evaluation.py`.

## Question
Does the amount of code-mixing in a tweet change how well a model classifies its sentiment?

## Method
1. **Buckets.** CMI per tweet comes from `results/cmi_per_tweet.csv` (standard formula, `src/cmi.py`). The two cut points are the 1/3 and 2/3 quantiles of CMI on the **training** set: 23.81 and 34.78. This reproduces the terciles documented in `docs/eda_part2.md` (low 0 to 23.8, medium 24.0 to 34.8, high 35.0 to 50). Dev tweets are placed into the buckets with these train cut points, so no dev information is used to define them.
2. **Join.** Each file `results/predictions/*_dev.csv` is joined to the CMI table by `uid`. The code stops with an error if any uid has no CMI value; none was missing.
3. **Scores.** Accuracy, precision and recall per class, macro F1 and weighted F1 (official metric) for all dev tweets and for each bucket.
4. **Uncertainty.** A bucket holds about 1,000 tweets. We resample the bucket's tweets 1,000 times (seed 42) and report a 95% interval for weighted F1.

## Results (dev set, TF-IDF baselines, weighted F1 with 95% interval)
| Model | Low CMI (1,018 tweets) | Medium CMI (981) | High CMI (1,001) | All (3,000) |
|---|---|---|---|---|
| Word 1-2 | 0.602 (0.567 to 0.632) | 0.609 (0.579 to 0.641) | 0.621 (0.592 to 0.649) | 0.611 |
| Char 2-5 | 0.611 (0.580 to 0.640) | 0.598 (0.568 to 0.629) | 0.625 (0.595 to 0.654) | 0.613 |
| Word + char | 0.596 (0.567 to 0.627) | 0.605 (0.575 to 0.635) | 0.630 (0.598 to 0.659) | 0.612 |

## What this shows
- **No measurable effect of CMI on TF-IDF performance.** Scores differ by at most about 0.03 between buckets, and every bucket's 95% interval is about 0.06 wide and overlaps the others. A difference of this size is what chance alone produces with 1,000 tweets.
- **The direction is not consistent.** High CMI is nominally best for all three models, but the medium bucket is lowest for one model, and the low bucket is lowest for the other two. We do not read a trend into this.
- **Why this may be so.** TF-IDF counts words and ignores order. Whether a tweet mixes the two languages heavily or lightly may matter little to a model that only looks at which words occur. Transformers read context, so the question should be asked again for them. This is a reason, not a tested explanation.

## Limits
- Dev set only; the test set is not used here.
- Buckets are equal-sized thirds of a distribution in which every tweet is mixed. "Low" means moderately mixed, not single-language (see `docs/eda_part2.md`).
- The 95% intervals are for each bucket separately. A formal test of the difference between two buckets (a paired bootstrap) was not done; the heavy overlap already says there is no clear difference.
- Our CMI averages (29.8) differ from the task paper's (25.3). The ranking of tweets is what the buckets use, so the conclusion does not depend on the absolute values.

## Next
The same script scores any new `results/predictions/*_dev.csv` file (BiLSTM, transformers), so the comparison across model types needs only a rerun once those files exist.
