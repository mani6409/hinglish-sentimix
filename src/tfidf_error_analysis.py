"""Error analysis of the TF-IDF word 1-2 baseline (Day 6).

Run from the repo root:  python -m src.tfidf_error_analysis
- Fits on TRAIN only (C=1, the value chosen on dev in baselines_tfidf), reads DEV only. No test data.
- Writes results/tfidf_top_features.json and docs/tfidf_confident_errors.md.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_split
from src.seed import DEFAULT_SEED, set_seed

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 25
N_ERRORS = 30


def md_escape(s):
    """Make a tweet safe for a Markdown table (no '|' and no new lines)."""
    return s.replace("|", "\\|").replace("\n", " ")


def main():
    """Train the best TF-IDF word model again, list the strongest words for each class, and save the
    dev tweets it got wrong with the highest confidence, to read by hand."""
    set_seed(DEFAULT_SEED)
    tr, dv = load_split("train"), load_split("dev")
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          lowercase=True, token_pattern=r"\S+")
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=DEFAULT_SEED)
    clf.fit(vec.fit_transform(tr["text"]), tr["label"])

    # the biggest weights show which words point most strongly to each class
    names = np.array(vec.get_feature_names_out())
    top = {c: names[np.argsort(-clf.coef_[i])[:TOP_K]].tolist() for i, c in enumerate(clf.classes_)}
    (ROOT / "results" / "tfidf_top_features.json").write_text(json.dumps(top, indent=2, ensure_ascii=False))

    # confidence = probability of the predicted class
    # wrong answers with high confidence are the ones to read first
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
