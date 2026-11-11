# Baselines: TF-IDF + logistic regression (dev set)

Reproduce with `python -m src.baselines_tfidf` (about 35 seconds). Full numbers: `results/tfidf_dev_scores.json`. Dev predictions: `results/predictions/tfidf_*_dev.csv`. All runs are in `results/experiment_log.csv`.

## What was done
- Text: the tweets as loaded (tokens joined by spaces), no extra cleaning. This is the "raw" data version; Ekta's normalised version will be tested later.
- Features: TF-IDF with sublinear term frequency and `min_df=2`. Vocabulary and weights are fitted on the **training set only**.
  - Word: single words and word pairs (1 to 2).
  - Character: letter sequences of length 2 to 5 inside words.
  - Word + character combined.
- Classifier: logistic regression. The regularisation strength C was chosen from 0.1, 1 and 10 using the **dev set**. The test set was not used.
- Metric: weighted F1 (official), plus accuracy.

## Results on dev (3,000 tweets)
| Model | Features | Best C | Weighted F1 | Accuracy |
|---|---|---|---|---|
| Always predict "neutral" | 0 | n/a | 0.205 | 0.376 |
| TF-IDF word 1-2 | 47,284 | 1 | 0.611 | 0.612 |
| TF-IDF char 2-5 | 122,301 | 1 | 0.613 | 0.612 |
| TF-IDF word + char | 169,585 | 0.1 | 0.612 | 0.611 |

For comparison, the best SemEval Hinglish system reached 0.750 weighted F1 (paper [1]).

## What we learned
1. **A simple model already scores about 0.61.** It is far above the always-neutral floor but well below the best competition result of 0.75, so there is room for the BiLSTM and transformers.
2. **Weighted F1 punishes lazy models.** Always guessing neutral gets 37.6% accuracy but only 0.205 weighted F1.
3. **Character n-grams did not help.** We expected them to cope better with the many spellings of Hindi words, but the three versions are within 0.0014 of each other. On 3,000 dev tweets the typical error in an F1 score is about 0.009, so these differences are noise. The hypothesis is not supported for this model; it may still hold for others.
4. **Neutral is the hardest class.**

   | Class | F1 (char model) |
   |---|---|
   | Negative | 0.63 |
   | Neutral | 0.55 |
   | Positive | 0.67 |

   Confusion matrix (rows true, columns predicted):

   | | Negative | Neutral | Positive |
   |---|---|---|---|
   | Negative | 571 | 265 | 54 |
   | Neutral | 261 | 623 | 244 |
   | Positive | 88 | 251 | 643 |

   Mistakes between negative and positive are rare (54 and 88). Almost all errors involve neutral, the middle class. A likely reason is that the boundary between neutral and the other two is fuzzy, and overall annotator agreement was only 55% (paper [1]). We have not checked agreement per class, so this is a hypothesis for the error analysis.
5. **Settings matter a little.** C = 10 lowered the score by about 0.02 for every version, which suggests overfitting at high C.

## Caveats
- These are single runs. Logistic regression is deterministic with a fixed seed, so repeating with other seeds would give the same numbers. The uncertainty comes from the finite dev set, not from the seed.
- C was selected on dev, so these dev scores are slightly optimistic. Final numbers come from the test set, run once at the end.
- The text still contains URL fragments, `@` mentions and `RT`. Cleaning these might change the scores; we test that in the normalisation ablation.

## Next
Day 6: finish this section of the report, review Ekta's preprocessing pull request, and start the dataset section.
