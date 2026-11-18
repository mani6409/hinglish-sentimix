# Baseline evaluation and error analysis (issue #4)

Reproduce with `python -m src.error_analysis`. Dev set only; the test set is not used. Reuses `src/metrics.py` and the existing TF-IDF predictions (`results/predictions/tfidf_*_dev.csv`); no baseline is retrained.

Files: `results/baseline_scores.csv`, `results/baseline_class_scores.csv`, `results/confusion_matrix_<model>.png`, `results/error_analysis.csv` (all 1,163 dev errors of the best baseline, with flags), `results/error_flag_rates.csv`, `results/error_categories_manual.csv` (hand categories). Tests: `tests/test_error_analysis.py`.

## 1. Scores (dev, 3,000 tweets)
| Model | Weighted F1 (official) | Macro F1 | Accuracy |
|---|---|---|---|
| TF-IDF char 2-5 (best) | 0.6127 | 0.6164 | 0.6123 |
| TF-IDF word 1-2 | 0.6113 | 0.6161 | 0.6120 |
| TF-IDF word + char | 0.6121 | 0.6154 | 0.6107 |
| Always "neutral" (floor) | 0.205 | n/a | 0.376 |

The three models are within 0.002 of each other, far inside the typical error of 0.009, so none is better than the others. We call char 2-5 the "best" only to choose one model for the error analysis. Comparison target for new models: **0.61 weighted F1** on dev (task winner: 0.750).

## 2. Per class
| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| char 2-5 | negative | 0.621 | 0.642 | 0.631 |
| char 2-5 | neutral | 0.547 | 0.552 | 0.550 |
| char 2-5 | positive | 0.683 | 0.655 | 0.669 |
| word 1-2 | negative | 0.622 | 0.666 | 0.644 |
| word 1-2 | neutral | 0.545 | 0.524 | 0.534 |
| word 1-2 | positive | 0.678 | 0.664 | 0.671 |
| word + char | negative | 0.633 | 0.619 | 0.626 |
| word + char | neutral | 0.534 | 0.572 | 0.552 |
| word + char | positive | 0.690 | 0.648 | 0.668 |

Neutral is the weakest class in every model, in both precision and recall. Negative and positive are about equally good.

## 3. Confusion matrices
`results/confusion_matrix_tfidf_char_2-5.png` (and the two others). Rows are the true label.

| Best model | Predicted negative | Predicted neutral | Predicted positive |
|---|---|---|---|
| True negative (890) | 571 (64%) | 265 (30%) | 54 (6%) |
| True neutral (1,128) | 261 (23%) | 623 (55%) | 244 (22%) |
| True positive (982) | 88 (9%) | 251 (26%) | 643 (65%) |

Of the 1,163 errors, 88% involve neutral (as the true or the predicted label). Only 142 (12%) swap negative and positive. The model rarely confuses opposite feelings; its trouble is the boundary with neutral.

## 4. What goes with errors? (automatic flags, all 3,000 dev tweets)
| Property | % of errors | % of correct tweets | p-value |
|---|---|---|---|
| Contains a negation word (nahi, nhi, not, don't ...) | 37.5 | 30.0 | < 0.001 |
| Contains an emoji | 16.6 | 19.9 | 0.028 |
| Contains a spelling variant from Ekta's dictionary | 13.8 | 12.0 | 0.16 |
| Cut off at the end ("...") | 9.3 | 9.5 | 0.88 |
| Contains a URL fragment | 56.4 | 56.9 | 0.83 |
| Has repeated letters (sooo) | 4.2 | 5.2 | 0.24 |

p-values are from a chi-square test of errors against correct tweets. With six tests, a p-value must be below about 0.008 to count; only negation does. The emoji difference (p = 0.028) is not strong enough to claim.

- **Negation is linked to errors.** The 988 dev tweets that contain a negation word are misclassified 44.1% of the time, against 36.1% for the 2,012 without one (8 points more). A bag-of-words model cannot flip the meaning of "acha nahi" (not good). This is the clearest, measured weakness, and the sequence models should be checked on it.
- **Spelling variants are not clearly linked to errors.** 13.8% of errors against 12.0% of correct tweets is not a reliable difference. The flag only detects the 57 variants in Ekta's dictionary, so it covers a small part of all spelling variation; the result is weak evidence either way for the transliteration part of our research question.
- Truncated tweets, URLs and repeated letters show no link to errors.

## 5. Categories read by hand (first pass)
A random sample of 60 errors from the best baseline (seed 42) was read and each given one main category. This is one reader's judgement and has not been checked by a second person. With 60 tweets, each percentage is uncertain by about ±10 points.

| Category | Tweets | Share | What it means |
|---|---|---|---|
| Label looks questionable | 14 | 23% | Gold label (mostly neutral) does not fit the text: greetings, praise or clear complaints labelled neutral |
| Context-dependent | 13 | 22% | No explicit sentiment words; meaning depends on political or social context, or an unseen follow-up |
| Sarcasm or irony | 9 | 15% | Positive-looking words used mockingly, or the reverse |
| Negation | 8 | 13% | The sentiment sits in a negated phrase ("itani asha nahi thi") |
| Slang and informal language | 7 | 12% | Abuse, internet slang, short forms (yu, jhand, BC) |
| Transliteration variants | 5 | 8% | Unusual spellings of Hindi/Urdu words (bohat, khubsurt, ummide) |
| Clear sentiment missed | 3 | 5% | Plainly positive or negative English the model still missed |
| Code-mixing challenge | 1 | 2% | Mixed script (Devanagari with Roman) |

Reading:
- About a quarter of errors may be a labelling problem rather than a model problem. This agrees with the earlier note on confident mistakes (`docs/baselines_tfidf.md`) and with the paper's low annotator agreement (55%). It is not yet a measured noise rate; a second reader should re-label the same 60 tweets and we should report how often we agree.
- Another fifth is context a word-count model cannot use. Models that read word order and context may help here, but cannot fix tweets with no sentiment words.
- Only 2 of the 60 sit in the code-mixing category. In this sample, errors are not mainly caused by the mixing itself, which fits the flat CMI result in `docs/cmi_analysis.md`.

## 6. For the team
- **Reviewer (Ekta or Monis):** read `results/error_categories_manual.csv` next to the tweets in `results/error_analysis.csv` (filter `manual_category`) and note the categories you disagree with. Report the share of agreement.
- **Everyone:** when your model's dev predictions are ready, save them as `results/predictions/<model>_dev.csv` (columns `uid,label,pred`). `python -m src.cmi_evaluation` then scores them by CMI, and this script can be pointed at them.
- **Experiment log:** the log has one score column (weighted F1). Macro F1 and accuracy per model are in `results/baseline_scores.csv`. The log's columns will be extended after PR #12 (which appends rows to it) is merged, to avoid a merge conflict.

## Limits
- Dev set only; one model family; one reader for the hand categories.
- The automatic flags are word lists, not understanding; "no" or "na" can be ordinary words.
- Flag differences show association, not cause.
