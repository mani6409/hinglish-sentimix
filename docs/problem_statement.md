# Problem statement (DRAFT: team edits this after the Day 2 call)

## Task
Classify the sentiment of Hindi-English (Hinglish) code-mixed tweets as **positive**, **negative** or **neutral**. This is SemEval-2020 Task 9 (SentiMix), Hinglish track.

## Why it is hard
- Hindi is written in Latin letters with no standard spelling, so one word can appear in several forms.
- Tweets switch between Hindi and English inside one sentence (average code-mixing index about 25 in the official data).
- Annotators agreed only 55% of the time (paper [1]), so some labels are noisy.

## Data
Official SentiMix Hinglish data: 14,000 training, 3,000 development and 3,000 test tweets, with word-level language tags (Hin, Eng, O, EMT).

## Evaluation
Weighted F1 over the three classes (paper [1], Section 3.1). The best system in the competition reached 75.0% F1.

## Our approach
1. Baselines: TF-IDF with logistic regression; BiLSTM with word and character embeddings.
2. Fine-tuned transformers: mBERT, XLM-R, MuRIL/IndicBERT.
3. Zero-shot LLM prompting.

## Research questions
1. Does normalising transliteration (making spelling variants consistent) improve accuracy?
2. How does accuracy change as the code-mixing index increases?

## To decide on the call (fill in)
- Final wording of the problem statement: ______
- Which normalisation method we try first: ______
- Which LLM we use: ______
- Anything we are unsure about after reading the papers: ______
