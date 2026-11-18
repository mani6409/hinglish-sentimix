"""Baseline evaluation and error analysis for the TF-IDF models (issue #4).

Run from the repo root:  python -m src.error_analysis
Reads results/predictions/tfidf_*_dev.csv (dev set only) and writes:
  results/confusion_matrix_<model>.png      one figure per model (counts and row percentages)
  results/baseline_scores.csv               weighted F1, macro F1, accuracy per model
  results/baseline_class_scores.csv         precision, recall, F1 per class per model
  results/error_analysis.csv                every dev error of the best model, with automatic flags
                                            and, for 60 tweets, a category read by hand
  results/error_flag_rates.csv              how common each automatic flag is among errors vs correct tweets
The hand categories live in results/error_categories_manual.csv (uid, category).
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.data import load_split
from src.metrics import LABELS, compute_metrics, confusion

ROOT = Path(__file__).resolve().parents[1]
BEST = "tfidf_char_2-5"  # highest dev weighted F1 of the three baselines
NEGATION = {"nahi", "nahin", "nhi", "nhin", "nahee", "nai", "na", "mat", "not", "no", "never", "dont", "don't",
            "cannot", "can't", "isn't", "aren't", "wasn't", "didn't", "doesn't", "won't"}


def variant_words():
    cfg = json.loads((ROOT / "configs" / "hinglish_variants.json").read_text())
    return {v for k, vs in cfg.items() if not k.startswith("_") for v in vs}


def flags(df, variants):
    toks = df["text"].str.lower().str.replace("’", "'").str.split()
    out = pd.DataFrame(index=df.index)
    out["negation_word"] = toks.apply(lambda t: any(w in NEGATION for w in t))
    out["spelling_variant"] = toks.apply(lambda t: any(w in variants for w in t))
    out["truncated"] = df["text"].str.rstrip().str.endswith("…")
    out["url"] = df["text"].str.contains(r"https\s*//", regex=True)
    out["repeated_letters"] = df["text"].apply(lambda s: bool(re.search(r"([a-zA-Z])\1{2,}", s)))
    out["has_emoji"] = df["text"].apply(lambda s: bool(re.search(r"[\U0001F300-\U0001FAFF☀-➿]", s)))
    return out


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0f}%", ha="center", va="center",
                    color="white" if pct[i, j] > 55 else "black", fontsize=9)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    dev = load_split("dev")
    paths = sorted((ROOT / "results" / "predictions").glob("tfidf_*_dev.csv"))
    score_rows, class_rows, preds = [], [], {}
    for p in paths:
        name = p.stem.replace("_dev", "")
        d = pd.read_csv(p)
        m = compute_metrics(d["label"], d["pred"])
        preds[name] = d
        score_rows.append({"model": name, "weighted_f1": m["weighted_f1"], "macro_f1": m["macro_f1"],
                           "accuracy": m["accuracy"]})
        for lab in LABELS:
            class_rows.append({"model": name, "class": lab, **m["per_class"][lab]})
        plot_confusion(confusion(d["label"], d["pred"]), f"{name} (dev)", ROOT / "results" / f"confusion_matrix_{name}.png")
    pd.DataFrame(score_rows).round(4).to_csv(ROOT / "results" / "baseline_scores.csv", index=False)
    pd.DataFrame(class_rows).round(4).to_csv(ROOT / "results" / "baseline_class_scores.csv", index=False)

    best = preds[BEST].merge(dev[["uid", "text", "n_tokens"]], on="uid", validate="one_to_one")
    cmi = pd.read_csv(ROOT / "results" / "cmi_per_tweet.csv")
    cmi = cmi[cmi["split"] == "dev"][["uid", "cmi"]]
    best = best.merge(cmi, on="uid", validate="one_to_one")
    f = flags(best, variant_words())
    best = pd.concat([best, f], axis=1)
    best["is_error"] = best["label"] != best["pred"]
    manual = pd.read_csv(ROOT / "results" / "error_categories_manual.csv")
    best = best.merge(manual, on="uid", how="left")
    errors = best[best["is_error"]].drop(columns="is_error")
    errors.rename(columns={"category": "manual_category"}).round(3).to_csv(ROOT / "results" / "error_analysis.csv", index=False)

    rows = []
    for col in f.columns:
        e, c = best.loc[best.is_error, col], best.loc[~best.is_error, col]
        table = [[int(e.sum()), int((~e).sum())], [int(c.sum()), int((~c).sum())]]
        _, pval, _, _ = chi2_contingency(table)
        rows.append({"flag": col, "pct_of_errors": 100 * e.mean(), "pct_of_correct": 100 * c.mean(), "p_value": pval})
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "error_flag_rates.csv", index=False)

    print(pd.DataFrame(score_rows).round(4).to_string(index=False))
    print(pd.DataFrame(class_rows).round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("manual categories:", manual["category"].value_counts().to_dict(), "errors:", len(errors))


if __name__ == "__main__":
    main()
