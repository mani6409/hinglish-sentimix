# EDA part 1: labels, length, language shares

Reproduce with `python -m src.eda_part1`. Numbers are in `results/eda_part1_stats.json`; figures are in `results/figures/`.

## 1. Label distribution
| Split | Tweets | Negative | Neutral | Positive |
|---|---|---|---|---|
| Train | 14,000 | 29.3% | 37.6% | 33.1% |
| Dev | 3,000 | 29.7% | 37.6% | 32.7% |
| Test | 3,000 | 30.0% | 36.7% | 33.3% |

- The classes are close to balanced; neutral is the largest. The three splits look alike.
- Always predicting "neutral" would give about 37.6% accuracy on train and dev. That is the floor any model must clearly beat.
- The official metric is weighted F1, which suits classes of unequal size.

## 2. Tweet length
| | Mean | Median | 95th percentile | Max |
|---|---|---|---|---|
| Tokens (train) | 26.1 | 28 | 34 | 56 |
| Characters (train) | 127 | 142 | 151 | 179 |

- Tweets are short and uniform: 95% have 34 tokens or fewer.
- Negative tweets are slightly longer on average (27.0 tokens vs 25.8 for neutral and 25.7 for positive). The gap is small, so do not read much into it yet.
- These are word-level tokens. Transformers split words into subword pieces, so their lengths will be larger. The maximum input length for transformers must be chosen after measuring real subword lengths with each model's tokenizer (part of Monis's training-script work).

## 3. Language shares (token tags)
| Split | Hindi | English | Other (`O`) | Emoticon |
|---|---|---|---|---|
| Train | 46.5% | 33.2% | 20.2% | 0.1% |
| Dev | 47.0% | 32.6% | 20.3% | 0.1% |
| Test | 46.1% | 33.6% | 20.1% | 0.1% |

- Hindi tokens are the most common, followed by English. About a fifth of tokens are "other" (mentions, URLs, punctuation, symbols).
- Almost every tweet is genuinely mixed. Taking the share of the more frequent language among Hindi and English tokens, the average is 0.70 and it almost never exceeds 0.9. Only 11 of 14,000 training tweets have 90% or more of one language.
- No tweet is without Hindi and English tokens.
- Consequence for our CMI research question: the data has little single-language text, so the CMI buckets will compare moderately mixed tweets with heavily mixed ones. We should not expect a "low mixing" group of tweets in one language.
- These language tags are noisy (see `data_format.md`), so everything here is approximate.

## 4. Safety checks
| Check | Result |
|---|---|
| Empty tweets | 0 in every split |
| Duplicate texts inside a split | 1 in train, 0 in dev and test |
| Same text in train and dev | 0 |
| Same text in train and test | 1 |
| Same text in dev and test | 0 |
| Train texts with conflicting labels | 0 |

There is practically no overlap between splits, so scores will not be inflated by repeated tweets.

## What this means for the next steps
1. The metric choice is confirmed: classes are balanced enough, and weighted F1 is the official score.
2. Model input length for BiLSTM can be about 64 words (longest tweet is 56). For transformers, wait for the tokenizer check.
3. Day 4: compute the code-mixing index per tweet. The paper reports average CMI of 25.32 (train), 25.53 (dev) and 25.13 (test), so our numbers should land near those.
4. Day 4: read 50 tweets per class by hand and note what makes them hard.

## Open questions for the team
- What do the 11 near-monolingual tweets look like? (Mani, Day 4)
- Is the language tagging good enough to trust for CMI? (Ekta and Mani, while reading tweets)
