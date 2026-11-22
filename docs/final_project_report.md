# Final project report (summary)

This is the Markdown summary asked for in issue #10. The full report, with related work, references and every
analysis, is [`reports/report_body.tex`](../reports/report_body.tex). Every number below comes from
[`results/final_model_comparison.csv`](../results/final_model_comparison.csv) (all models, dev and test) or
[`results/final_test_results.csv`](../results/final_test_results.csv) (the test rows of the same table).

## Task and data

Sentiment classification (negative, neutral, positive) of Hindi-English code-mixed tweets from SemEval-2020 Task 9
(SentiMix, Hinglish): 14,000 train, 3,000 dev and 3,000 test tweets, official splits unchanged. The official metric is
weighted F1, computed with `src/metrics.py`; macro F1, precision, recall and accuracy are also reported.

Research questions: (1) does normalising Romanised spelling help? (2) does the amount of code-mixing (CMI) change accuracy?

## Protocol

- Models are trained on train; dev is used for early stopping and for choosing settings.
- The test set was used once, under protocols written in advance
  ([BiLSTM](bilstm_final_evaluation_protocol.md), [transformers](transformer_final_evaluation_protocol.md)).
  Test predictions were saved before test labels were read, and no test result selected a model, seed or setting.
- BiLSTM, mBERT and XLM-R: seeds 42, 43, 44, reported as mean ± sample SD.

## Development results (3,000 tweets)

| Model | Weighted F1 | Macro F1 | Accuracy |
|---|---|---|---|
| Always "neutral" | 0.2055 | | 0.376 |
| TF-IDF word 1-2 + LR | 0.611 | 0.616 | 0.612 |
| TF-IDF char 2-5 + LR | 0.613 | 0.616 | 0.612 |
| TF-IDF word+char + LR | 0.612 | 0.615 | 0.611 |
| BiLSTM raw (3 seeds) | 0.611 ± 0.004 | 0.614 ± 0.006 | 0.610 ± 0.005 |
| BiLSTM clean (3 seeds) | 0.612 ± 0.002 | 0.617 ± 0.002 | 0.613 ± 0.003 |
| BiLSTM normalized (3 seeds) | 0.607 ± 0.001 | 0.611 ± 0.000 | 0.606 ± 0.001 |
| mBERT, clean, lr 2e-5 (3 seeds) | 0.610 ± 0.005 | 0.614 ± 0.005 | 0.610 ± 0.005 |
| XLM-R, clean, lr 1e-5 (3 seeds) | 0.627 ± 0.003 | 0.632 ± 0.003 | 0.631 ± 0.004 |
| MuRIL, raw, untuned (3 seeds) | 0.627 ± 0.005 | 0.632 ± 0.005 | 0.629 ± 0.006 |
| IndicBERT v2, raw, untuned (1 seed) | 0.6305 | 0.6360 | 0.6347 |

BiLSTM and transformer dev scores are the best epoch per run, so they are optimistic. TF-IDF, BiLSTM and mBERT are
within the sampling error of 3,000 tweets (about 0.009). XLM-R, MuRIL and IndicBERT are within each other's seed spread.

## Zero-shot reference (200-tweet dev subset only)

Qwen2.5-0.5B-Instruct, prompt v2, on a fixed random sample of 200 dev tweets (seed 42): weighted F1 0.477, accuracy 0.495.
These scores are **not comparable** with the full-dev or test rows; see [zero_shot_results.md](zero_shot_results.md).

## Official test results (3,000 tweets, evaluated once)

| Model | Weighted F1 | Macro F1 | Accuracy |
|---|---|---|---|
| BiLSTM raw | 0.660 ± 0.007 | 0.662 ± 0.008 | 0.658 ± 0.005 |
| BiLSTM clean | 0.668 ± 0.002 | 0.672 ± 0.003 | 0.667 ± 0.004 |
| BiLSTM normalized | 0.657 ± 0.006 | 0.659 ± 0.006 | 0.653 ± 0.007 |
| mBERT (lr 2e-5) | 0.687 ± 0.008 | 0.689 ± 0.008 | 0.685 ± 0.007 |
| XLM-R (lr 1e-5) | 0.701 ± 0.004 | 0.703 ± 0.004 | 0.700 ± 0.004 |

TF-IDF, MuRIL, IndicBERT and the zero-shot model were **not** evaluated on the test set. For reference, the best SemEval
Hinglish system reached 0.750 weighted F1. Details: [BiLSTM](../results/bilstm_final_test/bilstm_final_test_report.md),
[transformers](../results/transformer_final_test/transformer_final_test_report.md).

## Findings

- XLM-R is the strongest model on test, about 0.014 weighted F1 above mBERT and 0.03-0.04 above the BiLSTM. The
  XLM-R vs mBERT gap is not clearly beyond test sampling error.
- Spelling normalisation (a 25-word dictionary) did not help; cleaning alone was the best BiLSTM text version.
- CMI: no clear evidence that code-mixing level changes performance ([cmi_analysis.md](cmi_analysis.md)).
- Errors: 86-91% involve the neutral class, most errors are shared across models, and about a quarter of 60
  hand-checked TF-IDF errors have questionable gold labels ([error_analysis.md](error_analysis.md)).
- Test scores are higher than dev scores for every evaluated model; the cause is not identified.

## Limitations

Only the BiLSTM, mBERT and XLM-R have test results. MuRIL and IndicBERT used untuned settings, were still improving at
their last epoch, and IndicBERT has one seed. The zero-shot result is a 0.5B model on 200 dev tweets. Three seeds give
only rough SDs. The hand-checked error categories were checked by one reader, with no agreement figure.

## Reproduction

Setup, data location and commands are in the [README](../README.md). `python -m src.final_comparison` rebuilds
`results/final_model_comparison.csv`; `results/final_test_results.csv` is its rows with `split == test`. Tests:
`python -m pytest -q`.
