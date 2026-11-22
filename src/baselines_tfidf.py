"""TF-IDF + logistic regression baselines (Day 5).

Run from the repo root:  python -m src.baselines_tfidf
- Vocabulary and IDF weights are fitted on TRAIN only.
- Settings (C, n-gram range) are chosen on DEV only. The test set is never used here.
- Writes: results/tfidf_dev_scores.json, results/predictions/tfidf_<name>_dev.csv,
  and appends one row per model to results/experiment_log.csv.

Purpose:
    The first, simplest models of the project: bag-of-words and character n-gram features with a linear
    classifier, plus a "always predict the most common class" floor. They give every later model a score to
    beat and show how much a model that ignores word order can already do.

Why this file exists:
    A deep model is only worth its cost if it beats a cheap baseline. Having these numbers (about 0.61
    weighted F1 on dev, see docs/baselines_tfidf.md) also showed that the BiLSTM does not clearly beat them
    on the development set (docs/bilstm.md).

Place in the pipeline:
    src/data.py (train and dev text)  ->  TF-IDF features  ->  logistic regression  ->  dev predictions
    (results/predictions/*_dev.csv)  ->  read by src/cmi_evaluation.py and the later analysis scripts.

Inputs:
    ``load_split("train")`` and ``load_split("dev")``; the text is the tokens joined by spaces (the project's
    "raw" version, no cleaning).

Outputs:
    * results/predictions/tfidf_<name>_dev.csv with columns uid, label, pred; ``<name>`` is word_1-2,
      char_2-5 or word_char (the ``+`` in "word+char" is replaced by ``_`` in file names).
    * results/tfidf_dev_scores.json: scores for every C, the best C, and per-class numbers.
    * Four new rows (majority class and the three models) appended to results/experiment_log.csv.

Used by:
    Run once from the command line. Its prediction files are read by src/cmi_evaluation.py, and the model
    setup is repeated, for the word 1-2 model with C=1, in src/tfidf_error_analysis.py.

Depends on:
    scikit-learn, scipy, pandas, src/data.py, src/metrics.py, src/seed.py.

Design decisions:
    * Features are fitted on train only and applied to dev, so no information from dev reaches the model.
    * C (inverse regularisation strength) is chosen on dev. Therefore the dev scores are slightly
      optimistic; the project only reports test results for models chosen without looking at test.
    * Ties between C values go to the smaller C, because ``C_GRID`` is ascending and a later value must be
      strictly better to replace the best one.

Warnings for anyone rerunning it:
    * Rerunning APPENDS four more rows to results/experiment_log.csv (there is no duplicate check) and
      overwrites the prediction files and the JSON.
    * results/tfidf_dev_scores.json contains measured seconds per fit, so it differs between runs even though
      the scores do not.
    * The log row has the original nine columns; src/log_metrics.py fills the three later columns.
"""
import csv
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_split
from src.metrics import LABELS, compute_metrics
from src.seed import DEFAULT_SEED, set_seed

ROOT = Path(__file__).resolve().parents[1]
WHO = "Mani"  # written into the "who" column of the experiment log
DATA_VERSION = "raw (tokens joined by spaces)"
C_GRID = [0.1, 1.0, 10.0]  # regularisation strengths tried; smaller C = stronger regularisation

# Feature set name -> function that builds FRESH vectorizers (new objects each time, so nothing fitted
# on one split is reused by mistake). min_df=2 drops terms seen in a single tweet; sublinear_tf uses
# 1 + log(count); "char_wb" builds character n-grams inside word boundaries.
VECTORIZERS = {
    "word_1-2": lambda: [TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                         sublinear_tf=True, lowercase=True, token_pattern=r"\S+")],
    "char_2-5": lambda: [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                         sublinear_tf=True, lowercase=True)],
    "word+char": lambda: [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        lowercase=True, token_pattern=r"\S+"),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True),
    ],
}


def fit_features(make, train_text, dev_text):
    """Build TF-IDF features: fit on the training text, apply to the development text.

    Purpose:
        Keeps the leakage rule in one place: vocabulary and IDF weights come from train only.

    Args:
        make: One of the factories in ``VECTORIZERS``; returns a list of one or two vectorizers.
        train_text: Series of training tweets (strings).
        dev_text: Series of development tweets (strings).

    Returns:
        ``(Xtr, Xdv)``: sparse CSR matrices with one row per tweet. With two vectorizers (word + char) the
        columns of both are placed side by side.
    """
    vecs = make()
    Xtr = hstack([v.fit_transform(train_text) for v in vecs]).tocsr()  # fit on train only
    Xdv = hstack([v.transform(dev_text) for v in vecs]).tocsr()
    return Xtr, Xdv


def log_row(model, settings, dev_score, notes, lesson):
    """Append one result row to results/experiment_log.csv.

    Purpose:
        The project keeps one shared table of every experiment (who ran what, with which settings, which
        seed, and the dev score).

    Args:
        model: Model name.
        settings: Text describing the settings (for example ``C=1.0; min_df=2; ...``).
        dev_score: Weighted F1 on dev (the official metric); written with four decimals.
        notes: Free text.
        lesson: What was learned; empty for the TF-IDF rows.

    Side effects:
        Appends to the file (it is never rewritten). Columns, in order: date, who, model, data_version,
        settings, seed, dev_score, notes, lesson_learned. Line endings are CRLF because that is the csv
        module's default and the existing file uses it.
    """
    path = ROOT / "results" / "experiment_log.csv"
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([date.today().isoformat(), WHO, model, DATA_VERSION, settings,
                                DEFAULT_SEED, f"{dev_score:.4f}", notes, lesson])


def main():
    """Run the majority-class floor and the three TF-IDF models, and write all outputs.

    Stages:
        1. Floor: predict the most common TRAIN label for every dev tweet and score it.
        2. For each feature set in ``VECTORIZERS``: build features (fit on train), train one logistic
           regression per value of ``C_GRID``, score each on dev, and keep the best by weighted F1.
        3. Save the best model's dev predictions, log a row, and print a summary line.
        4. Write results/tfidf_dev_scores.json with everything above.

    Side effects:
        Creates results/predictions/ if missing, writes prediction files and the JSON, appends rows to the
        experiment log, prints. Takes about 35 seconds on a laptop (docs/baselines_tfidf.md).
    """
    set_seed(DEFAULT_SEED)
    # raw text: the TF-IDF baseline does not use src.preprocess
    tr, dv = load_split("train"), load_split("dev")
    (ROOT / "results" / "predictions").mkdir(parents=True, exist_ok=True)
    results = {}

    # floor: always predict the most common training label
    majority = tr["label"].value_counts().idxmax()
    m = compute_metrics(dv["label"], [majority] * len(dv))
    results["majority_baseline"] = {"label": majority, "weighted_f1": m["weighted_f1"], "accuracy": m["accuracy"]}
    log_row("majority-class", f"always '{majority}'", m["weighted_f1"], "floor", "any model must beat this")

    for name, make in VECTORIZERS.items():
        Xtr, Xdv = fit_features(make, tr["text"], dv["text"])
        grid = {}
        best = None
        # C is the inverse regularisation strength: small C = simpler model, large C = fits train more closely
        for C in C_GRID:
            t0 = time.time()
            # max_iter is raised so the solver converges on these large sparse matrices; random_state is
            # fixed so a solver that uses randomness stays repeatable.
            clf = LogisticRegression(C=C, max_iter=3000, random_state=DEFAULT_SEED)
            clf.fit(Xtr, tr["label"])
            pred = clf.predict(Xdv)
            mm = compute_metrics(dv["label"], pred)
            grid[str(C)] = {"weighted_f1": round(mm["weighted_f1"], 4), "accuracy": round(mm["accuracy"], 4),
                            "seconds": round(time.time() - t0, 1)}
            # strictly greater: on a tie the earlier (smaller) C is kept
            if best is None or mm["weighted_f1"] > best[0]:
                best = (mm["weighted_f1"], C, pred, mm)
        f1, C, pred, mm = best
        results[name] = {"n_features": Xtr.shape[1], "grid_dev": grid, "best_C": C,
                         "best_dev": {"weighted_f1": round(f1, 4), "macro_f1": round(mm["macro_f1"], 4),
                                      "accuracy": round(mm["accuracy"], 4), "per_class": mm["per_class"]}}
        pd.DataFrame({"uid": dv["uid"], "label": dv["label"], "pred": pred}).to_csv(
            ROOT / "results" / "predictions" / f"tfidf_{name.replace('+', '_')}_dev.csv", index=False)
        log_row(f"tfidf {name} + logistic regression",
                f"C={C}; min_df=2; sublinear_tf; features={Xtr.shape[1]}", f1,
                f"best of C in {C_GRID} chosen on dev", "")
        print(f"{name:10s} features={Xtr.shape[1]:>7d} best C={C:<5} dev weighted F1={f1:.4f} acc={mm['accuracy']:.4f}")

    (ROOT / "results" / "tfidf_dev_scores.json").write_text(json.dumps(results, indent=2))
    print("majority baseline:", results["majority_baseline"])


if __name__ == "__main__":
    main()
