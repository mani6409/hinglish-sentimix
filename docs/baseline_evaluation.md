# Baseline evaluation and error analysis (issue #4)

Reproduce with `python -m src.error_analysis`. Dev set only; the test set is not used. Reuses `src/metrics.py` and the existing TF-IDF predictions (`results/predictions/tfidf_*_dev.csv`); no baseline is retrained.

Files: `results/baseline_scores.csv`, `results/baseline_class_scores.csv`, `results/confusion_matrix_<model>.png`, `results/error_analysis.csv` (all 1,163 dev errors of the best baseline, with flags), `results/error_flag_rates.csv`, `results/error_summary.json` (error counts checked two ways), `results/error_categories_manual.csv` (first reader's hand categories, fixed), `results/error_categories_reviewer2_template.csv` (blank form for a second reader). Second-reader agreement: `python -m src.label_agreement`. Every merge is guarded by `src/uid_checks.py`: prediction files, the CMI table and the hand labels must contain exactly the expected uids (no missing, duplicate or unexpected uid), so no row can be lost silently. Tests: `tests/test_error_analysis.py`, `tests/test_label_agreement.py`, `tests/test_uid_checks.py`.

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

Of the 1,163 errors, 1,021 (87.8%) involve neutral (as the true or the predicted label): 505 are true-neutral tweets predicted as something else (261 + 244) and 516 are negative or positive tweets predicted neutral (265 + 251). Only 142 (12.2%) swap negative and positive (54 + 88). The count is computed twice, from the confusion matrix and from the prediction rows, and the script stops if they differ (`results/error_summary.json`). The model rarely confuses opposite feelings; its trouble is the boundary with neutral.

## 4. What goes with errors? (automatic flags, all 3,000 dev tweets)
| Property | % of errors | % of correct tweets | p-value |
|---|---|---|---|
| Contains a negation word (nahi, nhi, not, don't ...) | 37.5 | 30.0 | < 0.001 |
| Contains an emoji (heuristic) | 16.6 | 19.9 | 0.028 |
| Contains a spelling variant from Ekta's dictionary | 13.8 | 12.0 | 0.16 |
| Cut off at the end ("...") | 9.3 | 9.5 | 0.88 |
| Contains a URL fragment | 56.4 | 56.9 | 0.83 |
| Has repeated letters (sooo) | 4.2 | 5.2 | 0.24 |

p-values are from a chi-square test of errors against correct tweets. Six flags are tested, so the Bonferroni threshold is 0.05 / 6 = 0.0083 (also stored in `results/error_flag_rates.csv`, with a column saying whether each flag passes). Only negation passes. The emoji difference (p = 0.028) does not, so we do not claim it.

How the flags are detected (all are simple rules, not understanding):
- **URL:** matches both the tokenised form in this data (`https // t . co / x`, where the colon was lost when the text was split) and ordinary URLs (`https://example.com`, `http://example.com`, `www.example.com`). An earlier version only matched the tokenised form. Every URL in this data set is tokenised, so the correction changes none of the numbers here (56.4% of errors against 56.9% of correct tweets, as before); it matters if the script is used on other text.
- **Emoji:** a heuristic, not complete emoji detection. It matches only the Unicode blocks U+1F300 to U+1FAFF and U+2600 to U+27BF. Emoji outside them (for example regional-indicator flags) are missed, and a multi-codepoint sequence is flagged if any of its symbols falls in range. The emoji row is therefore approximate.
- **Negation:** a keyword heuristic: tweets are flagged if they contain any word of a fixed list (nahi, nhi, mat, na, no, not, never, ...). Words such as "no" or "na" can be ordinary words. The result does not depend on those ambiguous words: Ekta repeated the test without "na", "no" and "mat" and still found errors on 32.2% of flagged tweets against 25.5% (p < 0.001).

- **Negation is linked to errors.** The 988 dev tweets that contain a negation word are misclassified 44.1% of the time, against 36.1% for the 2,012 without one (8 points more). A bag-of-words model cannot flip the meaning of "acha nahi" (not good). This is the clearest, measured weakness, and the sequence models should be checked on it.
- **Spelling variants are not clearly linked to errors.** 13.8% of errors against 12.0% of correct tweets is not a reliable difference. The flag only detects the 57 variants in Ekta's dictionary, so it covers a small part of all spelling variation; the result is weak evidence either way for the transliteration part of our research question.
- Truncated tweets, URLs and repeated letters show no link to errors.

## 5. Hand-checked categories
A random sample of 60 errors from the best baseline (seed 42) was given one main category each. One reader (Mani) then went through all 60 tweets and kept every category unchanged. No second reader used this scheme, so there is no agreement figure. With 60 tweets, each percentage is uncertain by about ±10 points.

**How the 60 tweets were chosen.** `reading_sample_uids` in `src/error_analysis.py` draws 60 of the best baseline's dev errors (rows in dev-file order, `random_state=42`); `error_analysis` stops if `error_categories_manual.csv` is not exactly that sample, and a test checks the same.

**Original labels are kept fixed.** `results/error_categories_manual.csv` holds the categories exactly as assigned; because the check changed nothing, it is also the accepted version (MD5 `7126ecf962eff729071dc410b7310dc3`); it must not be edited. A second reader can work independently on `results/error_categories_reviewer2_template.csv` (the same 60 tweets, with an empty `category` column) using the same eight categories, and saves `uid,category` as a separate file. Then `python -m src.label_agreement <their file>` prints raw agreement, Cohen's kappa and the list of tweets where the two readers differ. The script refuses a file whose uids differ from the original 60. Kappa corrects raw agreement for the agreement expected by chance; with eight categories of very different size it is the fairer number to report next to raw agreement.

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
- Only 1 of the 60 sits in the code-mixing category. In this sample, errors are not mainly caused by the mixing itself, which fits the flat CMI result in `docs/cmi_analysis.md`.

## 6. For the team
- **Second reader:** Ekta labelled the same 60 tweets with a different scheme (13 tone labels, `results/error_categories_reviewer2_ekta.csv`), so `python -m src.label_agreement` cannot compare the two sets. If a reader relabels them with the eight categories, run `python -m src.label_agreement <file>` and report raw agreement and Cohen's kappa.
- **Everyone:** when your model's dev predictions are ready, save them as `results/predictions/<model>_dev.csv` (columns `uid,label,pred`). `python -m src.cmi_evaluation` then scores them by CMI. `python -m src.error_analysis` currently reads only the three `tfidf_*_dev.csv` files and analyses one of them (`BEST`); analysing another model needs a small change to that script (not yet generic).
- **Experiment log:** `results/experiment_log.csv` now has `dev_weighted_f1`, `dev_macro_f1` and `dev_accuracy` columns for all 16 runs (majority floor, three TF-IDF models, nine BiLSTM word+char runs, three BiLSTM word-only runs). They are recomputed from the dev prediction files by `python -m src.log_metrics`, which also checks them against the existing `dev_score` column (weighted F1) and against the macro F1 and accuracy written in the BiLSTM rows' notes; all 16 agree. The original columns are unchanged. After logging a new experiment, rerun the script to fill the new columns for that row.

## Limits
- Dev set only; one model family; hand categories checked by one reader, with no agreement figure.
- The automatic flags are word lists, not understanding; "no" or "na" can be ordinary words.
- Flag differences show association, not cause.
- The negation flag matches whole words only, so it misses split contractions such as `don ’ t` (tokenised) and negation expressed without a listed word.
- The tweet texts in `results/error_analysis.csv` and the reviewer template are **unmasked** (they contain `@mentions` and URL fragments). The data set is public (CC BY 4.0). Masked examples are used only where tweets are quoted in other documents.
