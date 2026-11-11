"""TF-IDF + logistic regression baselines (Day 5).

Run from the repo root:  python -m src.baselines_tfidf
- Vocabulary and IDF weights are fitted on TRAIN only.
- Settings (C, n-gram range) are chosen on DEV only. The test set is never used here.
- Writes: results/tfidf_dev_scores.json, results/predictions/tfidf_<name>_dev.csv,
  and appends one row per model to results/experiment_log.csv.
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
WHO = "Mani"
DATA_VERSION = "raw (tokens joined by spaces)"
C_GRID = [0.1, 1.0, 10.0]

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
    vecs = make()
    Xtr = hstack([v.fit_transform(train_text) for v in vecs]).tocsr()  # fit on train only
    Xdv = hstack([v.transform(dev_text) for v in vecs]).tocsr()
    return Xtr, Xdv


def log_row(model, settings, dev_score, notes, lesson):
    path = ROOT / "results" / "experiment_log.csv"
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([date.today().isoformat(), WHO, model, DATA_VERSION, settings,
                                DEFAULT_SEED, f"{dev_score:.4f}", notes, lesson])


def main():
    set_seed(DEFAULT_SEED)
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
        for C in C_GRID:
            t0 = time.time()
            clf = LogisticRegression(C=C, max_iter=3000, random_state=DEFAULT_SEED)
            clf.fit(Xtr, tr["label"])
            pred = clf.predict(Xdv)
            mm = compute_metrics(dv["label"], pred)
            grid[str(C)] = {"weighted_f1": round(mm["weighted_f1"], 4), "accuracy": round(mm["accuracy"], 4),
                            "seconds": round(time.time() - t0, 1)}
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
