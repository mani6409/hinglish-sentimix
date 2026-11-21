# Report draft: open items

`reports/report_body.tex` is the report text, ready to paste into the course Overleaf template (see the comment at the top of the file). Every number in it comes from a file in the repository. This list is what is still missing before submission.

## Placeholders in the text (search for `PENDING`)
1. Abstract and introduction: one sentence each on transformer and LLM results, if they exist.
2. Related work: 2 to 3 paragraphs from the team's paper summaries (`docs/paper_summary_template.md` is still empty).
3. Methodology: transformer and LLM description (needs Monis and Akansha).
4. Results: rows for transformers and the LLM in the dev and test tables.
5. Error analysis: raw agreement and Cohen's kappa from the second reader's labels (`python -m src.label_agreement <file>`).
6. Conclusion and limitations: what was and was not done for the transformers and the LLM.

## Checks before submission
- Verify every reference (authors, venue, year) against the papers; they were written from memory.
- The bootstrap intervals in "Spelling normalisation" come from Ekta's `reports/bilstm_final_analysis.md` (PR #16). Merge #16 first, or cite them from her PR.
- Table values: dev rows come from `results/baseline_scores.csv` and the BiLSTM analysis; test rows from `results/bilstm_final_test/`. Recheck them against the files after any rerun.
- Upload the figures used in the text to Overleaf: `results/figures/cmi_distribution.png` is the only one referenced so far. Candidates to add: `label_distribution.png`, `results/confusion_matrix_tfidf_char_2-5.png`, `results/bilstm_test_analysis/figures/confusion_matrices_dev_test.png`.
- Compile in Overleaf early; the page limit is none, but the template's headings should be followed.
- `docs/bilstm.md` still says "No result on the real SentiMix data has been produced yet" (a stale status line); it should be updated before the repository is shared.
- TF-IDF was not evaluated on the test set. If the team agrees, run it once under the same protocol so every model has a test number.
