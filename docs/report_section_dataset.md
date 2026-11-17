# Report draft: Dataset and exploratory analysis

Draft by Mani (Day 6). Numbers come from `docs/eda_part1.md`, `docs/eda_part2.md` and `docs/baselines_tfidf.md`; rerun the scripts if anything changes. Text in [BRACKETS] is still open. Paste into the NLE template once the layout is confirmed.

## 3. Dataset and analysis

### 3.1 Data
We use the Hinglish subset of SemEval-2020 Task 9 (SentiMix) [1]. It contains 14,000 training, 3,000 development and 3,000 test tweets in Hindi-English code-mixed text. Each tweet has one sentiment label (negative, neutral or positive) and every token has a language tag (Hindi, English, other, or emoticon). The tweets are tokenised and written mostly in Roman script; only 59 of the 14,000 training tweets contain Devanagari characters. Test labels come in a separate file and are used only once, for the final evaluation.

### 3.2 Labels and metric
The three classes are close to balanced (train: 29.3% negative, 37.6% neutral, 33.1% positive) and the three splits have almost the same distribution. A classifier that always predicts neutral reaches 37.6% accuracy but only 0.205 weighted F1 on the development set, which is the floor for any model. We report weighted F1, the official metric of the task [1], and also accuracy and per-class F1. The best SemEval system for Hinglish reached 0.750 weighted F1 [1], and annotators agreed on only 55% of the tweets [1], so a high ceiling should not be assumed.

### 3.3 Text properties
Tweets are short: the mean is 26 word tokens (95th percentile 34, maximum 56). Hindi tokens make up 46.5% of all tokens, English 33.2%, and "other" (mentions, URLs, punctuation, symbols) 20.2%. The text is raw social media: 81% of tweets contain a mention, 56% a URL fragment, 16% begin with "RT", and 11% are cut off at the end. URLs and punctuation are already split into separate tokens (for example "https // t . co"). The content includes abuse and political argument. Two data-quality issues appear: 11 training tweets are almost entirely in one language (ten are plain English) and one is the single word "tweet". Duplicate and overlap checks found at most one repeated text per split pair, so data leakage between splits is negligible.

### 3.4 Code-mixing index
We measure the amount of mixing per tweet with the Code-Mixing Index, CMI = 100 × (1 − max(w) / (n − u)), where n is the number of tokens, u the number of language-independent tokens (tags "other" and emoticon) and max(w) the count of the most frequent language [2]. With two languages the maximum is 50. The mean CMI is 29.8 on train, higher than the 25.3 average reported by the task organisers [1]. We tried several variants of the formula and none reproduced their value, so the exact definition they used is unknown; we use the standard formula consistently and compare tweets only by their rank. Almost every tweet is mixed: only one training tweet has CMI 0, the 10th percentile is 16.7 and the 90th is 45.0. Consequently there is no group of truly monolingual tweets, and our "low", "medium" and "high" groups (terciles: 0 to 23.8, 24 to 34.8, 35 to 50) compare moderately mixed with heavily mixed text.

CMI is only weakly related to the label: mean CMI is 28.9 for negative, 30.4 for neutral and 29.9 for positive tweets, a difference of about 1.5 against a spread of about 10. Negative tweets are 31.7% of the low-CMI group and 25.0% of the high-CMI group, but we have not tested whether this exceeds chance. [Day 7: test this and report the per-bucket scores.]

### 3.5 Baseline results and first error analysis
A TF-IDF plus logistic regression baseline fitted on the training set reaches 0.611 (word 1-2 grams), 0.613 (character 2-5 grams) and 0.612 (both) weighted F1 on the development set. The three differ by less than the typical error of about 0.009 on 3,000 tweets, so character n-grams gave no measurable benefit here. Neutral is the hardest class (F1 0.55 against 0.63 negative and 0.67 positive), and almost all errors involve neutral. The model relies on emotional words (insults for negative, greetings and praise for positive), and 26 of its 30 most confident errors are tweets labelled neutral that read as clearly positive or negative. This suggests that some neutral labels are questionable; we treat this as a hypothesis until the team has read the examples [and counted: X of 30 judged label-wrong].

### References for this section
[1] Patwa, P., Aguilar, G., Kar, S., Pandey, S., PYKL, S., Gambäck, B., Chakraborty, T., Solorio, T., Das, A. (2020). SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets. [verify the venue and pages against the paper PDF before submission]
[2] Gambäck, B., Das, A. (2014). On Measuring the Complexity of Code-Mixing. [verify the citation; CMI formula source]

## Open items before this section is final
1. Confirm the reference details above (cannot be checked offline here).
2. Add a figure: label distribution and CMI histogram (`results/figures/`).
3. Replace [X of 30] after the team reads `docs/tfidf_confident_errors.md`.
4. Check the section against the page limit and required sections once Chhavi replies.
