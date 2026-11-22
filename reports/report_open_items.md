# Report draft: open items

`reports/main.tex` is a complete document that follows the NLE style guide (title block with short running headlines, abstract, tables with the caption first and double rules, `\hpt` decimals, figure caption below, author-date citations, bibliography in the journal's format); it loads the NLE style if present and falls back to `article` otherwise. `reports/report_body.tex` holds the text, which can also be pasted between `\maketitle` and `\end{document}` of the course template. Upload `main.tex`, `report_body.tex` and `results/figures/cmi_distribution.png` to the Overleaf project. Every number in it comes from a file in the repository and was recomputed from the saved prediction or result files.

## Placeholders in the text (search for `PENDING`)
1. Related work now has a short factual text; check the statements about pretraining data (marked `% CHECK` in the file). A fuller survey of SentiMix systems was not done.
(The second-reader item is closed: the second reader used a different label scheme, so the report says no agreement figure can be computed. If she relabels the 60 tweets with the 8 categories, run `python -m src.label_agreement <file>` and report agreement and kappa.)

## Where the new numbers come from
- mBERT and XLM-R (development): `results/transformer_comparison.csv`, `results/transformer_pairwise.csv` and `docs/transformer_results.md` (on `main`).
- MuRIL (three seeds) and IndicBERT (one seed), development: `results/muril_*` and `results/indicbert_*` files and `docs/indian_transformer_results.md` (on `main`).
- mBERT and XLM-R (test): `results/transformer_final_test/`, `results/final_model_comparison.csv` (on `main`).
- Zero-shot (200 development tweets): `docs/zero_shot_results.md` and `results/zero_shot_dev_predictions.csv` (on `main`).
- Multi-model error analysis (27 pooled settings) and CMI bucket results over 32 prediction files: `docs/error_analysis.md`, `results/error_*.csv`, `results/cmi_bucket_*.csv` (on `main`, regenerated in #29). Pooled counts include near-duplicate learning-rate search runs.
- Bootstrap intervals for spelling normalisation: `results/bilstm_test_analysis/bilstm_final_analysis.md` (on `main`).

## What the report says was NOT done (keep it that way unless it changes)
- No test-set evaluation for TF-IDF, MuRIL, IndicBERT or the zero-shot model. Test results exist for the BiLSTM (`results/bilstm_final_test/`) and for mBERT and XLM-R (`results/transformer_final_test/`, one-time run, recomputed from the saved predictions).
- MuRIL and IndicBERT were run with untuned settings (3 epochs, best epoch was the last one in the runs inspected); IndicBERT was run once only (seed 42).
- The zero-shot run uses a 0.5-billion-parameter model on 200 development tweets only.

## Checks before submission
- Verify every reference (full author lists, venue, year) against the papers; they were written from memory. Confirm the author names, their order and the affiliation line in `main.tex`.
- Upload the figures used in the text to Overleaf: only `results/figures/cmi_distribution.png` is referenced. Candidates to add: `label_distribution.png`, `results/confusion_matrix_tfidf_char_2-5.png`, `results/bilstm_test_analysis/figures/confusion_matrices_dev_test.png`.
- Compile in Overleaf; no LaTeX is installed on the machine used to write the draft, so the file was only checked for balanced environments, labels and citations.
- `docs/bilstm.md` may still carry a stale "no result yet" status line; check before sharing the repository.
