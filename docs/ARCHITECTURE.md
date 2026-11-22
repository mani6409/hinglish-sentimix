# Architecture

How the code of this project fits together. Everything here was checked against the repository on `main`; where something could not be determined from the code it is listed under "Open questions".

## 1. What kind of system this is

An **offline experiment pipeline** for sentiment classification of Hinglish tweets (SemEval-2020 Task 9). It is a set of Python scripts that are run once from the repository root with `python -m src.<module>`. They read the data files in `data/`, train or evaluate models, and write files into `results/` and `docs/`.

There is **no web server, no API, no database, no authentication and no background job** in this project. The only configuration is a handful of JSON files in `configs/`; the only environment variable the code sets is `PYTHONHASHSEED` (in `src/seed.py`). `src/zero_shot_llm.py` can also read an optional API key from an environment variable (never from a file in the repository) for its hosted-API backend, which has not been run. Sections of a typical software architecture document about those topics do not apply here.

## 2. Components

| Layer | Modules | Responsibility |
|---|---|---|
| Data access | `src/data.py` | Parses the official CoNLL files into one DataFrame row per tweet (`load_split`). Every module gets its tweets from here. |
| Shared utilities | `src/metrics.py`, `src/seed.py`, `src/cmi.py`, `src/uid_checks.py` | The official score (weighted F1) and the confusion matrix; one place to fix randomness; the Code-Mixing Index of one tweet; the guard that makes uid merges safe. |
| Preprocessing | `src/preprocess.py`, `configs/hinglish_variants.json` | Three text versions: `raw`, `clean` (URLs/mentions replaced, noise removed) and `normalized` (clean plus repeated letters and a spelling-variant dictionary). |
| Baseline model | `src/baselines_tfidf.py` | TF-IDF features + logistic regression, and the "always neutral" floor. Uses the raw text from `src/data.py`. |
| BiLSTM | `src/bilstm.py`, `src/train_bilstm.py`, `configs/bilstm_default.json` | Word + character BiLSTM: model, text encoding, checkpoints (`bilstm.py`) and training/evaluation/logging (`train_bilstm.py`). |
| Transformers | `src/transformer_utils.py`, `src/train_transformer.py`, `configs/transformer_config.json`, `src/run_transformer_experiments.py`, `src/transformer_comparison.py`, `src/indian_transformer.py`, `src/train_muril.py`, `src/train_indicbert.py` | Fine-tuning of Hugging Face encoders (mBERT, XLM-R, MuRIL, IndicBERT), the learning-rate search driver and the comparison of their dev results. |
| Frozen test evaluation | `src/evaluate_transformer_test.py`, `src/eval_protocol_utils.py`, `src/final_comparison.py` | The one-time transformer test evaluation under a written protocol (`docs/transformer_final_evaluation_protocol.md`), its shared helpers (copied from the removed BiLSTM evaluation script) and the final comparison table. |
| LLM | `src/zero_shot_llm.py` | Zero-shot prompting; predictions are scored on a random subset of the dev set. |
| Analysis | `src/eda_part1.py`, `src/eda_part2.py`, `src/cmi_evaluation.py`, `src/tfidf_error_analysis.py`, `src/error_analysis.py`, `src/error_analysis_models.py`, `src/label_agreement.py`, `src/log_metrics.py` | Descriptive statistics, CMI per tweet, scores per CMI bucket with bootstrap intervals, error analyses (TF-IDF and pooled over all models), agreement between two readers' labels, and the extra score columns of the experiment log. |
| Tests | `tests/` | One test file per module group; synthetic data wherever possible. |

## 3. Module dependencies

Generated from the `import` statements of `src/*.py` (an arrow means "imports"):

```mermaid
flowchart LR
    subgraph shared[Shared utilities]
        data["data.py"]
        metrics["metrics.py"]
        seed["seed.py"]
        cmi["cmi.py"]
        uid_checks["uid_checks.py"]
    end
    baselines_tfidf["baselines_tfidf.py"]
    bilstm["bilstm.py"]
    cmi_evaluation["cmi_evaluation.py"]
    eda_part1["eda_part1.py"]
    eda_part2["eda_part2.py"]
    error_analysis["error_analysis.py"]
    error_analysis_models["error_analysis_models.py"]
    eval_protocol_utils["eval_protocol_utils.py"]
    evaluate_transformer_test["evaluate_transformer_test.py"]
    final_comparison["final_comparison.py"]
    indian_transformer["indian_transformer.py"]
    label_agreement["label_agreement.py"]
    log_metrics["log_metrics.py"]
    preprocess["preprocess.py"]
    run_transformer_experiments["run_transformer_experiments.py"]
    tfidf_error_analysis["tfidf_error_analysis.py"]
    train_bilstm["train_bilstm.py"]
    train_indicbert["train_indicbert.py"]
    train_muril["train_muril.py"]
    train_transformer["train_transformer.py"]
    transformer_comparison["transformer_comparison.py"]
    transformer_utils["transformer_utils.py"]
    zero_shot_llm["zero_shot_llm.py"]
    baselines_tfidf --> data & metrics & seed
    bilstm --> metrics & preprocess
    cmi_evaluation --> metrics & seed & uid_checks
    eda_part1 --> data & metrics
    eda_part2 --> cmi & data & metrics & seed
    error_analysis --> data & metrics & uid_checks
    error_analysis_models --> cmi_evaluation & data & metrics & preprocess & seed & uid_checks
    eval_protocol_utils --> data & metrics
    evaluate_transformer_test --> data & eval_protocol_utils & metrics & preprocess & transformer_utils
    final_comparison --> eval_protocol_utils
    indian_transformer --> metrics & preprocess & seed
    label_agreement --> data & uid_checks
    log_metrics --> data & metrics
    preprocess --> data
    run_transformer_experiments --> train_transformer & transformer_utils
    tfidf_error_analysis --> data & seed
    train_bilstm --> bilstm & metrics & preprocess & seed
    train_indicbert --> indian_transformer
    train_muril --> indian_transformer
    train_transformer --> metrics & preprocess & seed & train_bilstm & transformer_utils
    transformer_comparison --> cmi_evaluation & metrics & run_transformer_experiments & train_transformer & transformer_utils & uid_checks
    transformer_utils --> metrics & preprocess
    zero_shot_llm --> data & metrics & seed
    bilstm -. "lazy import inside two functions" .-> train_bilstm
```

Notes:
- `src/bilstm.py` and `src/train_bilstm.py` need each other. `bilstm.py` imports `train_bilstm` lazily inside two functions (with a comment saying why), so importing either module works.
- `src/train_transformer.py` reuses output-protection helpers from `src/train_bilstm.py` (`LOG_NAME`, `_parse_set`, `_publish`, `_stamp`, `log_has_run`).
- Nothing imports the two EDA scripts; they are run directly.

## 4. Data flow

```mermaid
flowchart TD
    raw["data/raw/.../Hinglish/*.txt<br/>official CoNLL files"] --> load[data.load_split]
    load -->|raw text| tfidf[TF-IDF + logistic regression]
    load --> pre[preprocess: raw / clean / normalized]
    pre --> bil[BiLSTM]
    pre --> tr[Transformers]
    load --> zs[Zero-shot LLM]
    tfidf --> pred[("results/predictions/*_dev.csv<br/>uid, label, pred")]
    bil --> pred
    tr --> pred
    zs -.->|"200-tweet subset, kept outside predictions/"| other["results/zero_shot_dev_predictions.csv"]
    load --> eda2[eda_part2: CMI per tweet]
    eda2 --> cmitab[("results/cmi_per_tweet.csv")]
    pred --> cmie[cmi_evaluation]
    cmitab --> cmie
    cmie --> out1[("results/cmi_bucket_*.csv")]
    pred --> ema[error_analysis_models]
    ema --> out2[("results/error_*.csv")]
    tfidf --> log[("results/experiment_log.csv")]
    bil --> log
    tr --> log
```

Reading the diagram:
1. **Input:** the official files are read only by `src/data.py`. Test labels come from a separate file and are joined by uid.
2. **Text versions:** the TF-IDF baseline uses the raw text; the BiLSTM and the transformers read the version chosen with a `mode` argument through `src/preprocess.py`.
3. **Common output:** dev models write `results/predictions/<name>_dev.csv` with the columns `uid,label,pred`. This file format is the contract between models and analyses. Exceptions: `src/indian_transformer.py` writes MuRIL seed 42 and IndicBERT to `results/<name>_dev_predictions.csv`, outside that folder, and the zero-shot file covers only 200 tweets and is also kept outside it.
4. **Analyses** read those files (never the models), so a new model that writes a full-dev file into `results/predictions/` is picked up by `python -m src.cmi_evaluation` automatically. Files outside that folder (MuRIL seed 42, IndicBERT, zero-shot) are not covered by `cmi_evaluation`; `src/error_analysis_models.py` reads them through its `EXTRA_PREDICTION_FILES` list.
5. **Logging:** training scripts append one row per run to `results/experiment_log.csv` (nine columns written by them: date, who, model, data_version, settings, seed, dev_score, notes, lesson_learned). `src/log_metrics.py` recomputes and adds `dev_weighted_f1`, `dev_macro_f1` and `dev_accuracy`, so the file has 12 columns; rows appended later have empty values there until it is rerun.

## 5. Rules that every part follows

- **Leakage:** vocabularies, IDF weights and length limits are fitted on the training split only; settings are chosen on dev; the test split is used once, at the end, under a written protocol (`docs/bilstm_final_evaluation_protocol.md`). Descriptive access to the test split (counts, CMI) is disclosed there.
- **One metric implementation:** `src/metrics.py`. The official score is **weighted F1**.
- **Joins by uid, never by row position,** and checked with `src/uid_checks.py` where predictions are merged with other tables.
- **Seeds:** `set_seed` at the start of each script; runs that are averaged use seeds 42, 43 and 44.
- **Outputs are never edited by hand.** Result files are produced by scripts; documents in `docs/` that quote them are written by hand and should be re-checked after a rerun.

## 6. Major design decisions

| Decision | Reason | Where |
|---|---|---|
| Data committed to the repository (Hinglish files only) | The course asks for the data in the repository; the licence (CC-BY 4.0) allows it. | `.gitignore`, `docs/data_format.md` |
| Standard CMI formula, used for ranking only | Our mean (29.8) does not match the paper's 25.3 and no variant reproduced it. | `src/cmi.py`, `docs/eda_part2.md` |
| Bucket edges from train quantiles | Dev/test information must not define the groups. | `src/cmi_evaluation.py`, `docs/cmi_analysis.md` |
| Predictions as files in a common format | Analyses stay independent of the models. | Section 4 |
| Bootstrap intervals, one generator per file | Bucket scores rest on about 1,000 tweets; adding a file must not change others' intervals. | `src/cmi_evaluation.py` |
| Three preprocessing versions | Separates the effect of cleaning from the effect of spelling normalisation. | `docs/preprocessing.md` |

## 7. Directory layout

```
data/raw/…            official Hinglish files (committed)
src/                  all code (run with python -m src.<module>)
tests/                pytest tests
configs/              experiment settings and the spelling-variant dictionary
results/              predictions, scores, logs, figures (written by scripts)
docs/                 explanations, protocols, analyses, this file
reports/              the report text
notebooks/            notebooks (BiLSTM summary, Colab runner)
```

## 8. Open questions

- The exact definition of CMI used by the task organisers is unknown (see `docs/eda_part2.md`).
- `dominant_share` is defined in both `src/eda_part1.py` and `src/eda_part2.py`, and `md_escape` in both `src/eda_part2.py` and `src/tfidf_error_analysis.py`; they could be shared.
- `LANGS` in `src/cmi.py` is not used by any code.
- The scripts that ran the frozen BiLSTM test evaluation were removed from the repository (`docs/bilstm_final_evaluation_protocol.md` explains and points to the git history); other code that imported helpers from them needs its own copies.
- The BiLSTM checkpoints are not in git (git-ignored), so the frozen test evaluation can be inspected through its saved results but not rerun from a clean clone.
