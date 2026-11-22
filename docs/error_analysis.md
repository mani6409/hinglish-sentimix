# Categorized error analysis (issue #9)

Reproduce (about one minute, no training): `python -m src.error_analysis_models`
Tests: `python -m pytest tests/test_error_analysis_models.py -q`

All numbers below come from the CSV files in `results/` written by that command. Dev set only (3,000 tweets); the test set is not used.

## What was analysed
27 model settings, taken from every dev prediction file in `results/`. Nothing was retrained for this report.
- TF-IDF word 1-2, char 2-5 and word+char (1 run each), and the word+character BiLSTM on raw, clean and normalised text (3 seeds each, 42/43/44), plus three single-seed word-only BiLSTM runs.
- mBERT and XLM-R runs from the learning-rate search and the final seeds (issues #5 and #6): 16 runs, each counted as its own setting.
- **MuRIL** (3 seeds, issue #7), **IndicBERT v2** (`ai4bharat/IndicBERTv2-MLM-only`, 1 seed, issue #7) and the **zero-shot LLM** (Qwen2.5-0.5B-Instruct, issue #8, only the 200 dev tweets it was run on).

Runs that differ only in the seed are merged into one *setting* by majority vote (ties go to the lowest seed). This is an ensemble of the seeds, so its weighted F1 can differ from the mean single-seed score; both are in `results/error_model_summary.csv`. For MuRIL the ensemble scores 0.6337 and the mean single-seed score is 0.6271.

**Caveat on the pooled numbers.** The mBERT and XLM-R learning-rate-search runs are near-duplicates of each other (same model, different learning rate), so they weigh heavily in every pooled count and test below. The pooled numbers describe "all runs in the repository", not "27 independent models".

Tweets that at least one setting got wrong: **2,284 of 3,000**. Wrong in every setting that covers them: **312** (10.4% of the dev set). Many errors are therefore shared by all models.

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
| Annotation error candidate | At least 75% of the settings that cover the tweet give the same wrong label | automatic (`label_noise_candidate`), then hand check |
| Very short | 15 tokens or fewer (about 8% of tweets; the shortest dev tweet has 8) | automatic (`very_short`) |

The negation and slang word lists are small and hand-made; they miss many cases and flag some wrong ones (for example "na" is also a question particle). Treat the rates below as rough.

## Findings

### 1. Neutral is the main confusion (all 26 full-dev settings)
Between 86% and 91% of each setting's errors involve the neutral class (`share_errors_involving_neutral`). Only a small share swap negative and positive directly. Counting each wrong tweet once by its most common wrong prediction: negative→neutral 585, neutral→negative 554, positive→neutral 537, neutral→positive 459, positive→negative 84, negative→positive 65.

### 2. Different models make mostly the same mistakes
Overlap of wrong tweets between two settings (Jaccard, `results/error_agreement.csv`), for the main settings:

| Pair | Jaccard |
|---|---|
| MuRIL vs IndicBERT | 0.623 |
| IndicBERT vs XLM-R (lr 1e-5, seed 42) | 0.592 |
| MuRIL vs XLM-R (lr 1e-5, seed 42) | 0.578 |
| MuRIL vs TF-IDF char 2-5 | 0.530 |
| MuRIL vs BiLSTM clean | 0.522 |
| IndicBERT vs BiLSTM clean | 0.522 |
| XLM-R vs BiLSTM clean | 0.525 |
| BiLSTM clean vs TF-IDF char 2-5 | 0.613 |

The three transformers overlap more with each other (0.58 to 0.62) than with the BiLSTM or TF-IDF (about 0.52 to 0.53), so pretrained transformers share more of their errors with one another. Even so, no pair in the table exceeds a Jaccard overlap of 0.62, so the sets of wrong tweets are far from identical. Switching model family changes a minority of the errors; most errors are shared, which points to the data (ambiguous or noisy labels, missing context) more than to one model's weakness.

### 3. Model-specific differences (dev recall, from `error_model_summary.csv`)
| Setting | runs | weighted F1 | recall negative | recall neutral | recall positive |
|---|---|---|---|---|---|
| MuRIL (3-seed ensemble) | 3 | 0.634 | 0.670 | 0.532 | 0.723 |
| IndicBERT v2 | 1 | 0.631 | 0.737 | 0.495 | 0.703 |
| XLM-R (lr 1e-5, seed 42) | 1 | 0.630 | 0.703 | 0.491 | 0.738 |
| BiLSTM clean | 3 | 0.620 | 0.674 | 0.521 | 0.689 |
| BiLSTM normalized | 3 | 0.618 | 0.608 | 0.593 | 0.651 |
| BiLSTM raw | 3 | 0.616 | 0.626 | 0.576 | 0.647 |
| TF-IDF char 2-5 | 1 | 0.613 | 0.642 | 0.552 | 0.655 |
| TF-IDF word+char | 1 | 0.612 | 0.619 | 0.572 | 0.648 |
| TF-IDF word 1-2 | 1 | 0.611 | 0.666 | 0.524 | 0.664 |
| Zero-shot Qwen2.5-0.5B (200 tweets only, not comparable) | 1 | 0.477 | 0.200 | 0.726 | 0.424 |

- The transformers find positive tweets best (0.70 to 0.74) but recognise neutral least often (0.49 to 0.53). IndicBERT has the highest negative recall (0.737) of all settings in this table, with neutral recall 0.495 (XLM-R is lowest at 0.491).
- BiLSTM-normalized has the best neutral recall (0.593) at the cost of negative recall (0.608).
- The top of the ranking is very close: MuRIL, IndicBERT and XLM-R are within 0.004 weighted F1, and the MuRIL seed spread alone is about 0.005. This is a tie, not a ranking. IndicBERT and XLM-R were analysed with one seed each, so part of the difference in recall could be seed noise.
- The zero-shot model (a very small one, see `docs/zero_shot_results.md`) is far below all trained models. It mostly answers neutral and rarely says negative (recall 0.20); 101 of its 200 answers are wrong, and its most common confusion is negative→neutral (40).

### 4. Which surface features go with more errors
Error rate over all (setting, tweet) pairs, with vs without the feature (`results/error_flag_rates_models.csv`). Chi-square p-values treat each pair as independent, but the same tweet appears once per setting (and the learning-rate-search runs are near-duplicates), so they overstate certainty a lot; the Bonferroni threshold is 0.05/7 = 0.0071.

| Feature | Error rate with | without | p | Below Bonferroni threshold |
|---|---|---|---|---|
| Negation word | 0.434 | 0.358 | 1.2e-95 | yes |
| Spelling variant (normaliser changes text) | 0.426 | 0.378 | 4.9e-20 | yes |
| Slang / informal | 0.372 | 0.392 | 9.6e-9 | yes, but errors are *lower* with the flag |
| Very short (≤15 tokens) | 0.365 | 0.386 | 8.8e-4 | yes, but errors are *lower* with the flag |
| High CMI | 0.376 | 0.388 | 2.1e-3 | yes, but errors are *lower* with the flag; difference about 1 point |

- **Negation** is the clearest weakness: about 7.5 points more errors with a negation word. The flag rule is crude, so the true effect size is uncertain.
- **Spelling variants** go with about 5 points more errors. This agrees with the motivation for normalisation, but the normalised BiLSTM did not score better overall, so this is an association, not proof of a cause.
- **Slang, very short tweets and high CMI** go with slightly *fewer* errors. With so many pooled, non-independent pairs even a difference of one or two points becomes "significant", so no conclusion about these three features should be drawn from the p-values.

### 5. Code-mixing (CMI) and errors
Error rate per CMI bucket (`results/error_cmi_rates.csv`, buckets from the training terciles: low, medium, high) is between 0.355 and 0.413 for the 26 full-dev settings. The high bucket has the lowest error rate in 20 of the 26 settings (high 0.355 to 0.395, low 0.361 to 0.413, medium 0.367 to 0.403). For MuRIL the rates are 0.371 (low), 0.368 (medium) and 0.355 (high); for IndicBERT 0.367, 0.374 and 0.355. The intervals in the file overlap, so this does not show that more code-mixing helps or hurts; it fits the CMI analysis: re-running `python -m src.cmi_evaluation` over all settings gives 96 pairwise bucket differences (`results/cmi_bucket_pairwise.csv`), of which only 1 has an interval that excludes zero, about what chance gives. (`docs/cmi_analysis.md` still describes the earlier run with 36 intervals.) Among the 312 tweets wrong in every setting the buckets are nearly equal (high 107, low 111, medium 94), so consistently wrong tweets are not concentrated at high CMI either.

### 6. Possible annotation errors, and a warning about the flag
546 of the 2,284 error tweets are flagged `label_noise_candidate` (at least 75% of the settings agree on one wrong label). That is too many to all be mislabelled: the flag also catches tweets that are simply hard or lack context. It narrows the list for a human to check; it does **not** measure label noise. The earlier baseline analysis estimated a share of questionable labels from a hand-labelled sample; use that or the hand labels below for a number. The SemEval paper reports 55% annotator agreement (see `docs/problem_statement.md`), which is consistent with a high ceiling on accuracy.

### 7. Representative examples
Usernames are replaced by `@USER`. These were chosen at random (seed fixed) among tweets that were wrong in all six baseline settings and carry the feature, not picked for being funny. After adding the transformer and LLM runs I re-checked each one: the last column shows in how many settings it is wrong now. The label is the gold label; "predicted" is the most common wrong prediction. Whether the feature is really the *reason* is my reading, not something the automatic flag proves.

| Feature | uid | gold → predicted | Wrong in | Tweet (shortened as in the data) | Comment |
|---|---|---|---|---|---|
| Negation | 29086 | positive → negative | 26 of 26 | "@USER @USER Opposition can never die nhi toh taanashah sarkaar ho jyegi and that is not in our constitution" | Two negations and a political stance; sentiment depends on the author's view. |
| Negation | 26812 | positive → neutral | 26 of 26 | "Is zamane ne gham ke alawa kuch dekha nahi . Aajkal ummeed duniya se naya rishta jodne ki taak mein hai ..." | Poetic; "nahi" is part of a sentence about sorrow, label is positive. Hard even for a human. |
| Sarcasm cue | 36134 | negative → positive | 14 of 26 | "@USER Sbse Badi bt terrorist sadhvi pragya jeet gyi hament karkare ko shrap dene wali wah desh such trakki krra hh" | "wah" (praise) used bitterly; needs the news context as well. The only example that some models (12 of 26) get right. |
| Sarcasm cue | 15606 | negative → positive | 26 of 26 | "RT @USER Give me an Emoji .. ❤ I Love You 😂 ... I hate You 🙂 Nice person Aby band kro ye chutiyapa 😡😪…" | Mixed positive and negative words; the closing abuse makes it negative. |
| Spelling variant | 4534 | negative → neutral | 24 of 27 | "@USER May i ask what happened wit you logo ko hnsanay wali aaj khud udas hai kyu jnab" | Many non-standard spellings ("hnsanay", "wit", "jnab"); sadness word "udas" is rare. |
| Spelling variant | 24804 | neutral → negative | 26 of 26 | "@USER Bus dalit muslim politics krte rahoge toh kuch haasil nahi aaplog ye jaanne k liye tyar hi nahi ground ..." | Complaint tone, but gold label neutral: could be an annotation disagreement. |
| Very short | 5395 | neutral → positive | 25 of 26 | "Godbless bro always remember so you may never forget ! @USER" | Religious blessing; labelled neutral, models read it as positive. |
| Unflagged | 42802 | neutral → positive | 27 of 27 | "RT @USER Thanks to PM Shri @USER Ji and @USER president Shri @USER Ji for giving me opportunity to serve my count ..." | "Thanks" reads positive; the label neutral looks questionable. Wrong even for the zero-shot model. |
| Unflagged | 23450 | negative → neutral | 25 of 27 | "@USER Modi ji Desh Ke corrupt officer on ko hatao ..." | Criticism without a strong sentiment word. |

### 8. Category counts in the 60-error sample (checked by one reader)
`results/error_manual_sample.csv` holds 60 stratified errors of the best-scoring setting (MuRIL). Each row has one category from `MANUAL_CATEGORIES`. Every row was read and checked by a person (Akanksha); the `notes` column records this. There was a single reader, so no agreement between readers is reported.

| Category | Tweets | Share of 60 |
|---|---|---|
| Context-dependent (needs event, person or news) | 19 | 32% |
| Annotation error candidate (gold label looks questionable) | 13 | 22% |
| Slang / informal / abuse | 7 | 12% |
| Sarcasm / irony | 7 | 12% |
| Negation | 6 | 10% |
| Transliteration / spelling | 4 | 7% |
| Other (not Hinglish: Turkish, Indonesian, Punjabi) | 3 | 5% |
| Code-mixing | 1 | 2% |

Caveats: one label per tweet is forced although many tweets fit two categories; the sample is a stratified sample of MuRIL's errors, so the shares do not describe all errors; with 60 tweets and one reader the shares are rough (a few points either way mean nothing). The pattern is that many errors need outside context or have a label that looks doubtful, which agrees with the shared-error finding above. A second reader would be needed to report how far the labels agree (`src/label_agreement.py`).

## Recurring weaknesses
1. **Neutral boundary:** about 9 in 10 errors involve neutral; none of the 26 settings separates it well (neutral recall 0.48 to 0.62). The three best transformers do not fix this (neutral recall 0.49 to 0.53).
2. **Negation** and **non-standard spelling** go with more errors; the models do not seem to handle polarity flips.
3. **Context and stance:** political and news tweets whose polarity depends on knowing the event or the author's side are repeatedly wrong in all models.
4. **Shared errors:** 312 tweets are wrong in every setting, which a different classifier alone is unlikely to fix.
5. **Small zero-shot model:** answers neutral by default and misses negative tweets (recall 0.20); this says nothing about stronger LLMs.

## Suggested future work
- Add a lexicon-based or learned negation scope feature.
- Run IndicBERT and XLM-R with three seeds, so their recall profiles can be compared with MuRIL's on equal terms.
- Have a second reader check the hand sample and report agreement (`src/label_agreement.py`).
- Check the flagged "annotation error candidates" by hand and drop or relabel those that are clearly wrong for a cleaner dev estimate.
- Collect context (thread, news) for stance-dependent tweets, or treat them as an irreducible error floor.
- Repeat the zero-shot analysis with a stronger LLM; the 0.5B model is too weak to say much about prompting on Hinglish.

## Limitations of this analysis
- Dev set only; one split. Flags are heuristics (word lists), not human categories. The pooled chi-square tests are not independent, and the learning-rate-search runs inflate the pooled counts.
- Seed-ensembles were analysed for MuRIL and the BiLSTM, single runs for IndicBERT, XLM-R, mBERT and TF-IDF.
- The zero-shot predictions cover only 200 tweets, so its row is not comparable with the others.
- **The category labels of the 60-error sample come from one reader.** No second reader labelled the same tweets, so the labels have no agreement score, and one category per tweet is forced. Nothing else in this document depends on these labels.

## Files
- Code: `src/error_analysis_models.py`; tests: `tests/test_error_analysis_models.py`.
- Output: `results/error_analysis_models.csv` (misclassified tweets with flags), `results/error_model_summary.csv`, `results/error_flag_rates_models.csv`, `results/error_agreement.csv`, `results/error_cmi_rates.csv`, `results/error_manual_sample.csv`.
- `src/error_analysis.py` and `results/error_analysis.csv` on `main` belong to the baseline error analysis (PR #14). The model-level analysis uses different file names on purpose, so the two do not collide.
