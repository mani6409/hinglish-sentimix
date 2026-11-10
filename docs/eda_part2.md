# EDA part 2: code-mixing index (CMI), CMI vs label, what the text looks like

Reproduce with `python -m src.eda_part2`. Per-tweet values: `results/cmi_per_tweet.csv`. Numbers: `results/eda_part2_stats.json`. Figures: `results/figures/cmi_distribution.png`, `cmi_by_label.png`.

## 1. How CMI is computed
`CMI = 100 * (1 - max(w) / (n - u))`, implemented in `src/cmi.py`.
- `n` = tokens in the tweet; `u` = language-independent tokens (tags `O` and `EMT`); `max(w)` = tokens of the most frequent language (Hin or Eng).
- 0 = one language; because there are only two languages, the maximum possible value is 50 (an even split).
- Everyone must use `src.cmi.cmi` so all CMI numbers in the project agree.

## 2. Our CMI does not match the paper's averages
| Split | Paper [1] average CMI | Ours (standard formula) |
|---|---|---|
| Train | 25.32 | 29.77 |
| Dev | 25.53 | 29.96 |
| Test | 25.13 | 29.71 |

- The gap is about 4.5 points, the same in every split, so the code is consistent and the difference comes from the definition.
- I also tried counting `O` only as independent, counting nothing as independent, and several formulas that add language switching. None matched; the closest (the average of the standard value and a switch-point value) gave about 24.1, still about 1.2 off.
- The exact formula the paper used is not given in the text. We use the standard formula throughout and say in the report that our values differ from the paper's average and why we do not know the exact definition. For our ablation only the ranking of tweets by CMI matters, not matching the paper.

## 3. CMI distribution (train)
- Mean 29.8, median 28.6, 10th percentile 16.7, 90th percentile 45.0, maximum 50.
- Only one tweet has CMI 0, and almost none are below 10. Every tweet contains both languages to some degree.
- Terciles (equal-sized groups): low 0 to 23.8, mid 24.0 to 34.8, high 35.0 to 50. These are candidate buckets for the Day 7 ablation.

## 4. CMI and sentiment
| Label | Mean CMI | Median | Std |
|---|---|---|---|
| Negative | 28.9 | 27.3 | 9.8 |
| Neutral | 30.4 | 29.4 | 10.4 |
| Positive | 29.9 | 29.2 | 10.7 |

- Differences between labels (about 1.5 points) are small compared with the spread (about 10), so CMI alone says very little about sentiment.
- Label shares by tercile: negative tweets are 31.7% of the low group and 25.0% of the high group; neutral is 34.9% and 40.1%. This is descriptive only. We have not tested whether it is more than chance.

## 5. What the text looks like (train, 14,000 tweets)
| Property | Tweets |
|---|---|
| Contain a URL fragment (`https // t . co ...`) | 7,805 (56%) |
| Contain a mention (`@ name`) | 11,282 (81%) |
| Start with `RT` | 2,218 (16%) |
| End with `…` (the tweet was cut off) | 1,527 (11%) |
| Contain Devanagari script | 59 |

- Text has already been split into tokens: punctuation and URLs are separated by spaces (for example `https // t . co / xyz`, `didn ’ t`). The model sees these fragments unless we clean them. Ekta's preprocessing should replace URLs and mentions with placeholders.
- Many tweets are cut off at the end, so some sentiment may be missing from the text.
- Almost all text is Latin script; only 59 tweets contain Devanagari. Transliteration normalisation therefore means making Romanised spelling consistent, not converting scripts.
- Content is real social media, including abusive language and political arguments. Be aware of this when reading the samples.

## 6. The 11 near-monolingual tweets
See `docs/near_monolingual_tweets.md`. Ten are plain English with no Hindi at all, and one has the text "tweet" only (uid 18288). So the training data contains a few tweets that are not Hinglish and one that is junk. This is tiny (11 of 14,000), but worth a sentence in the report on data quality.

## 7. For the team
- **Reading task (everyone, 20 minutes):** open `docs/reading_sample.md` (50 tweets per class) and write notes in the last column. Note slang, repeated letters, sarcasm, negation and labels you disagree with. These notes feed the error analysis.
- **Ekta:** see section 5 for preprocessing needs; also list 30 spelling variants of common Hindi words from the sample.
- **Monis:** check how mBERT and XLM-R tokenise the URL fragments and mentions, since they add many tokens.
- **Mani, Day 7:** define CMI buckets; the terciles above are the starting point.
