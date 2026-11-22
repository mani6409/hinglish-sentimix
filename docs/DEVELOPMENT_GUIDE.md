# Development guide

For a new contributor: how to find things, run things, add things and check your work. See `docs/ARCHITECTURE.md` for how the parts fit together.

## 1. Setup and daily commands

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # BiLSTM versions actually used: requirements-bilstm.txt
python -m pytest -q                      # about 100 seconds; about 335 tests on main (the number grows with every merged pull request)
```

- Always run scripts **from the repository root** with `python -m src.<module>` (not `python src/<module>.py`), because modules import each other as `src.<name>`.
- Tests for the BiLSTM and the transformers are skipped when `torch` or `transformers` is not installed (`pytest.importorskip`). A green run without them does not test those parts; install them for the full suite.
- Data is in the repository (`data/raw/zenodo_download/extracted/Semeval_2020_task9_data/Hinglish`); nothing needs to be downloaded. Source and format: `docs/data_format.md`.

## 2. Where do I look?

| I want to... | Look at |
|---|---|
| Load tweets | `src/data.py` (`load_split("train" \| "dev" \| "test")`) |
| Compute a score | `src/metrics.py` (`compute_metrics`; the official score is `weighted_f1`) |
| Make a run repeatable | `src/seed.py` (`set_seed`) |
| Use cleaned or normalised text | `src/preprocess.py`, `docs/preprocessing.md` |
| Compute CMI | `src/cmi.py`; the saved values are in `results/cmi_per_tweet.csv` |
| Safely merge predictions with other tables | `src/uid_checks.py` |
| See how a baseline is built | `src/baselines_tfidf.py`, `docs/baselines_tfidf.md` |
| Train the BiLSTM / a transformer | `src/train_bilstm.py` (`docs/bilstm.md`); `src/train_transformer.py` (`docs/transformer_pipeline.md`) |
| Understand a result | the file in `docs/` named after the analysis (`eda_part1.md`, `eda_part2.md`, `cmi_analysis.md`, `error_analysis.md`, ...) |
| Know what is frozen or protected | `docs/bilstm_final_evaluation_protocol.md` and `configs/bilstm_frozen_checkpoints.json` |

## 3. Rules that are easy to break

1. **Fit on train, choose on dev, touch test once.** Vocabularies, IDF weights, length limits and bucket edges come from the training split only. Do not load the test split (`load_split("test")` attaches the labels) except under a written protocol.
2. **One metric implementation.** Use `compute_metrics`; never write another F1.
3. **Join by uid, never by row position,** and call `check_uid_alignment` before merging predictions with anything.
4. **Do not edit frozen or generated files by hand:** `configs/bilstm_frozen_checkpoints.json`, `results/bilstm_final_test/`, and every file in `results/` that a script writes. Change the script and rerun it.
5. **No force-push and no rewriting of pushed commits.** Merge `main` into your branch, or add a fix commit.
6. **Commit messages** are plain descriptions of the change; do not add co-author or tool trailers.

## 4. How to add a new model

1. Load tweets with `load_split("train")` and `load_split("dev")`; call `set_seed(seed)` first. Use the text version you intend to compare (`raw`, `clean` or `normalized` via `src/preprocess.py`).
2. Fit everything on train; pick the best epoch or setting on dev.
3. Write the dev predictions to `results/predictions/<name>_dev.csv` with exactly the columns `uid,label,pred`, all 3,000 dev uids once, labels spelled `negative`, `neutral`, `positive`. Runs that differ only in seed should end in `_s<seed>_seed<seed>` so `src/cmi_evaluation.py` groups them.
4. Score with `compute_metrics` and append a row to `results/experiment_log.csv` (nine columns: date, who, model, data_version, settings, seed, dev_score, notes, lesson_learned; `dev_score` is weighted F1). The file also has three computed columns (`dev_weighted_f1`, `dev_macro_f1`, `dev_accuracy`); run `python -m src.log_metrics` to fill them for your row.
5. Run `python -m src.cmi_evaluation`: a file in `results/predictions/` is picked up automatically (it stops with a uid error if the file does not cover exactly the dev set). Predictions written outside that folder are not covered.
6. Add tests (synthetic data, no GPU, no downloads) and a document in `docs/` that states settings, results and limits.
7. Do **not** put a file that covers only part of dev (for example a sampled subset) into `results/predictions/`; keep it elsewhere, as the zero-shot file is.

## 5. How to add an analysis script

Follow the existing pattern (see `src/cmi_evaluation.py`): a module header that says purpose, inputs, outputs and callers; a `main()` that reads saved files and writes into `results/`; `if __name__ == "__main__": main()`; helper functions that can be tested without the real data. Say in the header if a rerun overwrites something people edited by hand (for example `docs/tfidf_confident_errors.md`). Add a test file next to the others.

## 6. Documentation conventions

- Every source file starts with a header docstring: purpose, why it exists, place in the pipeline, inputs, outputs, callers, dependencies, design decisions, limits.
- Every function has a Google-style docstring (purpose, how it works, `Args`, `Returns`, `Raises`, edge cases, an example where useful). Examples in docstrings must be true; check them.
- Comments explain *why*, not what. If something cannot be determined from the code, write it as an open question.
- Do not change behaviour while documenting. Verify with this check, which compares the parsed code with comments and docstrings removed:

```python
import ast, subprocess, sys

def strip_docstrings(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
                    and isinstance(first.value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(tree)

ref = "origin/main"
for path in sys.argv[1:]:
    old = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, encoding="utf-8").stdout
    print("SAME   " if strip_docstrings(ast.parse(old)) == strip_docstrings(ast.parse(open(path, encoding="utf-8").read())) else "CHANGED", path)
```

## 7. Working with GitHub

Issue first, then a branch, then a pull request that says `Closes #<issue>`. A teammate reviews before merging; the author does not approve their own pull request. Keep a pull request to one issue. When a pull request is stacked on another branch, change its base to `main` once the first one is merged. Expect the experiment log to conflict when two branches append rows; keep both sets of rows.

## 8. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the repository root with `python -m src.<module>`. |
| `FileNotFoundError` for a data file | The Hinglish folder is missing or moved. It must be at `data/raw/zenodo_download/extracted/Semeval_2020_task9_data/Hinglish` (see `docs/data_format.md`), or pass `root=` to `load_split`. |
| `ValueError: ... duplicate uids / uids ... are missing / ... not in ...` | A prediction file does not match the dev split exactly (a subset, a repeated tweet, or tweets from another split). Fix the file; do not loosen the check. |
| `ValueError: unknown labels ... expected ['negative', 'neutral', 'positive']` | Predictions contain integers, other spellings or missing values. Convert them to the three label strings first. |
| `ValueError: ... bad header line / token line without a tab` | A data file is damaged or a different format was loaded. Compare with `docs/data_format.md`. |
| Many tests are skipped | `torch` or `transformers` is not installed. Install them and rerun. |
| Scores differ between runs | `set_seed` was not called first, a different seed was used, or the run used a GPU (cuDNN settings in `src/seed.py` make it repeatable but slower). `PYTHONHASHSEED` only affects processes started afterwards. |
| `experiment_log.csv` has duplicate rows | Rerunning `src/baselines_tfidf.py` appends four rows each time (no duplicate check). Remove the extra rows by hand and commit. |
| Typed verdicts disappeared from `docs/tfidf_confident_errors.md` | `src/tfidf_error_analysis.py` rewrites that file. Restore it from git and copy the verdicts elsewhere before rerunning. |
| Plotting error about a display | The analysis scripts select the non-interactive `Agg` backend before importing pyplot; keep that line first when editing. |

## 9. Limits of this guide

There is no deployment, API or database in this project, so there is nothing to document for those topics. Modules written by other team members are described only at the level of their own headers; read their files and the documents named in section 2 for details.
