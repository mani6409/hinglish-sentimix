# Categorized error analysis (issue #9)

Reproduce (about one minute, no training): `python -m src.error_analysis_models`
Tests: `python -m pytest tests/test_error_analysis_models.py -q`

All numbers below come from the CSV files in `results/` written by that command. Dev set only (3,000 tweets); the test set is not used.

## What was analysed
Six model settings, from the existing dev predictions in `results/predictions/` (nothing was retrained):
TF-IDF word 1-2, char 2-5 and word+char (1 run each), and the word+character BiLSTM on raw, clean and normalised text (3 seeds each, 42/43/44).
Runs that differ only in the seed are merged into one *setting* by majority vote (ties go to the lowest seed). This is an ensemble of the seeds, so its weighted F1 can differ from the mean single-seed score; both are in `results/error_model_summary.csv`.
Predictions of MuRIL/IndicBERT (issue #7) and the zero-shot LLM (issue #8) are picked up automatically if their files exist; they did not exist when this report was produced, so **they are not included**.

Tweets that at least one setting got wrong: **1,720 of 3,000**. Wrong in all six settings: **664**. Distribution of "number of settings wrong" (1 to 6): 275, 206, 211, 185, 179, 664. Many errors are therefore shared by every model.

## Error categories (definitions)
Defined in `src/error_analysis_models.py` (`CATEGORIES`). The automatic part only produces **candidate flags** from keyword and pattern rules. A flag says "this tweet has the surface feature", not "this is the cause of the error".

| Category | Definition | How it is found |
|---|---|---|
| Sarcasm / irony | Positive-looking words or laughing emoji while the gold label is negative | automatic cue (`sarcasm_cue`) + hand label |
| Slang / informal | Slang, abuse words, ALL-CAPS words, stretched letters | automatic (`slang_informal`) |
| Code-mixing | High-CMI group (CMI above the training 2/3 quantile, 34.78) | automatic (`high_cmi`, buckets from `src/cmi_evaluation.py`) |
| Negation | Contains nahi, nhi, na, mat, not, don't, ... | automatic (`negation`) |
| Transliteration / spelling | `src.preprocess.normalize_hinglish` rewrites something | automatic (`spelling_variant`) |
| Context-dependent | Needs outside knowledge (event, person, news) | **hand label only** |
| Annotation error candidate | At least 75% of at least 4 settings give the same wrong label | automatic (`label_noise_candidate`), then hand check |
| Very short | 15 tokens or fewer (about 8% of tweets; the shortest dev tweet has 8) | automatic (`very_short`) |

The negation and slang word lists are small and hand-made; they miss many cases and flag some wrong ones (for example "na" is also a question particle). Treat the rates below as rough.

## Findings

### 1. Neutral is the main confusion (all six settings)
Between 88% and 91% of each setting's errors involve the neutral class (`share_errors_involving_neutral`). Only a small share swap negative and positive directly. Counting each tweet once by its most common wrong prediction: negative→neutral 419, neutral→positive 377, neutral→negative 389, positive→neutral 408, negative→positive 57, positive→negative 70.

### 2. The settings make mostly the same mistakes
Overlap of wrong tweets between two settings (Jaccard, `results/error_agreement.csv`) lies between 0.59 and 0.76. The two TF-IDF variants char 2-5 and word+char overlap most (0.76); BiLSTM clean vs TF-IDF word+char overlap least (0.59). The BiLSTM seeds-ensembles overlap 0.69 to 0.71 with each other. So switching model family or preprocessing changes a minority of the errors; most errors are shared, which points to the data (ambiguous or noisy labels, missing context) more than to one model's weakness.

### 3. Model-specific differences (dev recall, from `error_model_summary.csv`)
| Setting | recall negative | recall neutral | recall positive |
|---|---|---|---|
| BiLSTM clean | 0.674 | 0.521 | 0.689 |
| BiLSTM normalized | 0.608 | 0.593 | 0.651 |
| BiLSTM raw | 0.626 | 0.576 | 0.647 |
| TF-IDF char 2-5 | 0.642 | 0.552 | 0.655 |
| TF-IDF word+char | 0.619 | 0.572 | 0.648 |
| TF-IDF word 1-2 | 0.666 | 0.524 | 0.664 |

The settings trade neutral recall against negative/positive recall: BiLSTM-clean and TF-IDF word 1-2 find the polar classes more often but recognise neutral least often (0.52); BiLSTM-normalized has the best neutral recall (0.59) at the cost of negative recall (0.61). Weighted F1 of all six is within 0.009 of each other (0.611 to 0.620), so these are *different error profiles at about the same score*, not clear winners. Only one seed-ensemble per BiLSTM setting was analysed, so seed noise could account for part of the differences in this table.

### 4. Which surface features go with more errors
Error rate over all (setting, tweet) pairs, with vs without the feature (`results/error_flag_rates_models.csv`). Chi-square p-values treat each pair as independent, but the same tweet appears once per setting, so they overstate certainty; the Bonferroni threshold is 0.05/5 = 0.0071.

| Feature | Error rate with | without | p | Below Bonferroni threshold |
|---|---|---|---|---|
| Negation word | 0.440 | 0.358 | 1.6e-26 | yes |
| Spelling variant (normaliser changes text) | 0.421 | 0.380 | 2.0e-4 | yes |
| Slang / informal | 0.371 | 0.396 | 6.1e-4 | yes, but errors are *lower* with the flag |
| High CMI | 0.375 | 0.391 | 0.038 | no |
| Very short (≤15 tokens) | 0.363 | 0.388 | 0.063 | no |

- **Negation** is the clearest weakness: about 8 points more errors with a negation word. Flag rule is crude, so the true effect size is uncertain.
- **Spelling variants** go with about 4 points more errors. This agrees with the motivation for normalisation, but the normalised BiLSTM did not score better overall, so this is an association, not proof of a cause.
- **Slang** did not go with more errors; our rule found slightly fewer. The word list is probably too crude to capture the hard cases, so no conclusion about slang should be drawn from it.

### 5. Code-mixing (CMI) and errors
Error rate per CMI bucket (`results/error_cmi_rates.csv`, buckets from the training terciles: low, medium, high) is between 0.370 and 0.403 for every setting, and the high bucket has the lowest error rate for all six settings (0.370 to 0.379, against 0.382 to 0.403 for low/medium). The intervals in the file overlap, so this does not show that more code-mixing helps or hurts; it matches the earlier CMI analysis (`docs/cmi_analysis.md`). Among the 664 tweets wrong in all six settings, the buckets are nearly equal (high 219, low 227, medium 218), so consistently-wrong tweets are not concentrated at high CMI either.

### 6. Possible annotation errors, and a warning about the flag
751 of the 1,720 error tweets are flagged `label_noise_candidate` (at least 75% of the settings agree on one wrong label). That is too many to all be mislabelled: the flag also catches tweets that are simply hard or lack context. It narrows the list for a human to check; it does **not** measure label noise. The earlier baseline analysis estimated a share of questionable labels from a hand-labelled sample; use that or the hand labels below for a number. The SemEval paper reports 55% annotator agreement (see `docs/problem_statement.md`), which is consistent with a high ceiling on accuracy.

### 7. Representative examples
Usernames are replaced by `@USER`. Chosen at random (seed fixed) among tweets wrong in all six settings that carry the feature, not picked for being funny. The label is the gold label; "predicted" is the most common wrong prediction. Whether the feature is really the *reason* is my reading, not something the automatic flag proves.

| Feature | uid | gold → predicted | Tweet (shortened as in the data) | Comment |
|---|---|---|---|---|
| Negation | 29086 | positive → neutral | "@USER @USER Opposition can never die nhi toh taanashah sarkaar ho jyegi and that is not in our constitution" | Two negations and a political stance; sentiment depends on the author's view. |
| Negation | 26812 | positive → neutral | "Is zamane ne gham ke alawa kuch dekha nahi . Aajkal ummeed duniya se naya rishta jodne ki taak mein hai ..." | Poetic; "nahi" is part of a sentence about sorrow, label is positive. Hard even for a human. |
| Sarcasm cue | 36134 | negative → positive | "@USER Sbse Badi bt terrorist sadhvi pragya jeet gyi hament karkare ko shrap dene wali wah desh such trakki krra hh" | "wah" (praise) used bitterly; needs the news context as well. |
| Sarcasm cue | 15606 | negative → positive | "RT @USER Give me an Emoji .. ❤ I Love You 😂 ... I hate You 🙂 Nice person Aby band kro ye chutiyapa 😡😪…" | Mixed positive and negative words; the closing abuse makes it negative. |
| Spelling variant | 4534 | negative → neutral | "@USER May i ask what happened wit you logo ko hnsanay wali aaj khud udas hai kyu jnab" | Many non-standard spellings ("hnsanay", "wit", "jnab"); sadness word "udas" is rare. |
| Spelling variant | 24804 | neutral → negative | "@USER Bus dalit muslim politics krte rahoge toh kuch haasil nahi aaplog ye jaanne k liye tyar hi nahi ground ..." | Complaint tone, but gold label neutral: could be an annotation disagreement. |
| Very short | 5395 | neutral → positive | "Godbless bro always remember so you may never forget ! @USER" | Religious blessing; labelled neutral, models read it as positive. |
| Unflagged | 42802 | neutral → positive | "RT @USER Thanks to PM Shri @USER Ji and @USER president Shri @USER Ji for giving me opportunity to serve my count ..." | "Thanks" reads positive; the label neutral looks questionable. |
| Unflagged | 23450 | negative → neutral | "@USER Modi ji Desh Ke corrupt officer on ko hatao ..." | Criticism without a strong sentiment word. |

## Recurring weaknesses
1. **Neutral boundary:** nearly 9 in 10 errors involve neutral; none of the six settings separates it well (recall 0.52 to 0.59).
2. **Negation** and **non-standard spelling** go with more errors; the models do not seem to handle polarity flips.
3. **Context and stance:** political and news tweets whose polarity depends on knowing the event or the author's side are repeatedly wrong in all models.
4. **Shared errors:** 664 tweets are wrong in all six settings, which a different classifier alone is unlikely to fix.

## Suggested future work
- Transformer models with multilingual pretraining (issues #5 to #7) may cope better with unseen spellings and negation; compare their wrong-tweet overlap with the BiLSTM using `results/error_agreement.csv` (rerun the command).
- Add a lexicon-based or learned negation scope feature.
- Have a second reader check the hand sample and report agreement (`src/label_agreement.py` exists on the baseline-evaluation branch, PR #14).
- Check the flagged "annotation error candidates" by hand and drop or relabel those that are clearly wrong for a cleaner dev estimate.
- Collect context (thread, news) for stance-dependent tweets, or treat them as an irreducible error floor.

## Limitations of this analysis
- Dev set only; one split. Flags are heuristics (word lists), not human categories. The pooled chi-square tests are not independent.
- Seed-ensembles were analysed, not single runs.
- **The hand-labelled category counts are not part of this report yet.** `results/error_manual_sample.csv` holds 60 stratified errors of the best-scoring setting (BiLSTM clean) with an empty `manual_category` column. It has to be filled by a person; only then can a hand-category table be reported. Nothing in this document is claimed from hand labels.
- Predictions from MuRIL, IndicBERT and the zero-shot LLM are not included (not produced yet).

## Files
- Code: `src/error_analysis_models.py`; tests: `tests/test_error_analysis_models.py`.
- Output: `results/error_analysis_models.csv` (misclassified tweets with flags), `results/error_model_summary.csv`, `results/error_flag_rates_models.csv`, `results/error_agreement.csv`, `results/error_cmi_rates.csv`, `results/error_manual_sample.csv`.
- The file names differ from the issue's `src/error_analysis.py` / `results/error_analysis.csv` on purpose: the open PR #14 (branch `feat/baseline-evaluation`) already uses those names, and the two would collide when merged.
