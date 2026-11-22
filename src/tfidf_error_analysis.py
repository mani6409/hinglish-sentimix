"""Error analysis of the TF-IDF word 1-2 baseline (Day 6).

Run from the repo root:  python -m src.tfidf_error_analysis
- Fits on TRAIN only (C=1, the value chosen on dev in baselines_tfidf), reads DEV only. No test data.
- Writes results/tfidf_top_features.json and docs/tfidf_confident_errors.md.

Purpose:
    Shows what the word 1-2 TF-IDF model has learned (the words with the largest weight per class) and which
    development tweets it gets wrong with the highest confidence.

Why this file exists:
    The most confident mistakes are the best place to look for label noise: if a model is very sure and the
    gold label still disagrees, either the model or the label is wrong. Reading those tweets gave the
    hypothesis, written up in docs/baselines_tfidf.md, that some "neutral" labels are questionable.

Place in the pipeline:
    src/data.py (train, dev)  ->  TF-IDF word 1-2 + logistic regression (same setup as src/baselines_tfidf.py)
    ->  top features JSON + a Markdown table of the 30 most confident dev errors for the team to judge.

Inputs:
    ``load_split("train")`` and ``load_split("dev")``. The model is refitted here (the baseline script does
    not save its model), with C=1.0 written in as the value that script selected for this feature set.

Outputs:
    * results/tfidf_top_features.json: for each class (negative, neutral, positive) the ``TOP_K`` = 25
      features with the largest weight.
    * docs/tfidf_confident_errors.md: the ``N_ERRORS`` = 30 dev errors with the highest predicted
      probability, with an empty "my verdict" column.

Used by:
    Run once from the command line. Its outputs are quoted in docs/baselines_tfidf.md and the report draft.

Depends on:
    scikit-learn, numpy, src/data.py, src/seed.py.

Warnings and notes:
    * Rerunning OVERWRITES docs/tfidf_confident_errors.md; verdicts typed into the empty column would be
      lost, so copy them elsewhere first.
    * A large weight is not the same as a raw association between a word and a class (see
      docs/baselines_tfidf.md); the lists show what this model uses, not causes.
    * ``md_escape`` is also defined in src/eda_part2.py.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_split
from src.seed import DEFAULT_SEED, set_seed

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 25  # words listed per class
N_ERRORS = 30  # confident mistakes listed


def md_escape(s):
    """Make a tweet safe to put into a Markdown table cell.

    Args:
        s: Tweet text.

    Returns:
        The text with ``|`` written as ``\\|`` and newlines replaced by a space.
    """
    return s.replace("|", "\\|").replace("\n", " ")


def main():
    """Fit the model, list the top words per class and the most confident dev errors.

    Stages:
        1. Fit TF-IDF (word 1-2 grams, fitted on train) and logistic regression with C=1.0.
        2. For every class take the ``TOP_K`` features with the largest coefficient. The rows of
           ``coef_`` follow ``clf.classes_``, which is sorted alphabetically (negative, neutral, positive).
        3. Predict probabilities for dev. The predicted class is the one with the highest probability and
           that probability is the "confidence".
        4. Keep the wrong predictions, sort them by confidence (highest first) and take ``N_ERRORS``.
        5. Write the JSON and the Markdown table, and print a short summary.

    Side effects:
        Overwrites results/tfidf_top_features.json and docs/tfidf_confident_errors.md; prints.
    """
    set_seed(DEFAULT_SEED)
    tr, dv = load_split("train"), load_split("dev")
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          lowercase=True, token_pattern=r"\S+")
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=DEFAULT_SEED)
    clf.fit(vec.fit_transform(tr["text"]), tr["label"])

    names = np.array(vec.get_feature_names_out())
    # argsort of the NEGATED weights puts the largest weight first; row i of coef_ belongs to classes_[i]
    top = {c: names[np.argsort(-clf.coef_[i])[:TOP_K]].tolist() for i, c in enumerate(clf.classes_)}
    (ROOT / "results" / "tfidf_top_features.json").write_text(json.dumps(top, indent=2, ensure_ascii=False))

    proba = clf.predict_proba(vec.transform(dv["text"]))
    dv = dv.assign(pred=clf.classes_[proba.argmax(1)], conf=proba.max(1))
    wrong = dv[dv["pred"] != dv["label"]]
    sure = wrong.sort_values("conf", ascending=False).head(N_ERRORS)

    n_wrong = len(wrong)
    true_neutral = int((wrong["label"] == "neutral").sum())
    top_true_neutral = int((sure["label"] == "neutral").sum())
    lines = [
        "# TF-IDF word 1-2: the most confident mistakes on dev",
        "",
        "Reproduce with `python -m src.tfidf_error_analysis`. Model fitted on train only; these are dev tweets.",
        "",
        f"- Dev errors: {n_wrong} of {len(dv)}. Of these, {true_neutral} ({100 * true_neutral / n_wrong:.0f}%) have the true label neutral.",
        f"- Among the {N_ERRORS} errors the model was most sure about, {top_true_neutral} have the true label neutral.",
        "- Read them yourself: do you think the gold label is wrong, or is the model wrong?",
        "",
        "| uid | true | predicted | confidence | tweet | my verdict (label wrong / model wrong / unclear) |",
        "|---|---|---|---|---|---|",
    ]
    for r in sure.itertuples():
        lines.append(f"| {r.uid} | {r.label} | {r.pred} | {r.conf:.2f} | {md_escape(r.text)} | |")
    (ROOT / "docs" / "tfidf_confident_errors.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for c, words in top.items():
        print(c, words[:12])
    print(f"errors {n_wrong}; true neutral {true_neutral}; neutral in top-{N_ERRORS} {top_true_neutral}")


if __name__ == "__main__":
    main()
