# Report draft: open items

`reports/report_body.tex` is the report text, ready to paste into the course Overleaf template (see the comment at the top of the file). Every number in it comes from a file in the repository and was recomputed from the saved prediction or result files.

## Placeholders in the text (search for `PENDING`)
1. Related work: 2 to 3 paragraphs from the team's paper summaries (`docs/paper_summary_template.md` is still empty).
2. Error analysis, hand categories: raw agreement and Cohen's kappa from a second reader's labels of the same 60 tweets (`python -m src.label_agreement <file>`). If no second reader finishes, delete that sentence.

## Where the new numbers come from
- mBERT and XLM-R (development): `results/transformer_comparison.csv`, `results/transformer_pairwise.csv` and `docs/transformer_results.md` from pull request #20. **The report cites them, so #20 must be merged before the report is submitted.**
- IndicBERT (one seed, development): `results/indicbert_dev_scores_seed42.json` and `results/indicbert_dev_predictions.csv` from pull request #26. **#26 must also be merged before the report is submitted.**
- Zero-shot (200 development tweets): `docs/zero_shot_results.md` and `results/zero_shot_dev_predictions.csv` (on `main`).
- Multi-model error analysis: `docs/error_analysis.md` and the `results/error_*.csv` files (on `main`).
- Bootstrap intervals for spelling normalisation: `results/bilstm_test_analysis/bilstm_final_analysis.md` (on `main`).

## What the report says was NOT done (keep it that way unless it changes)
- No test-set evaluation for TF-IDF, mBERT, XLM-R or the zero-shot model; only the BiLSTM has test results.
- MuRIL was not trained (code exists, a GPU is needed). IndicBERT was run once only (seed 42, untuned, still improving at its last epoch).
- The zero-shot run uses a 0.5-billion-parameter model on 200 development tweets only.

## Checks before submission
- Verify every reference (authors, venue, year) against the papers; they were written from memory.
- Upload the figures used in the text to Overleaf: only `results/figures/cmi_distribution.png` is referenced. Candidates to add: `label_distribution.png`, `results/confusion_matrix_tfidf_char_2-5.png`, `results/bilstm_test_analysis/figures/confusion_matrices_dev_test.png`.
- Compile in Overleaf; no LaTeX is installed on the machine used to write the draft, so the file was only checked for balanced environments, labels and citations.
- `docs/bilstm.md` may still carry a stale "no result yet" status line; check before sharing the repository.
