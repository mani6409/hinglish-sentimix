"""Baseline evaluation and error analysis for the TF-IDF models (issue #4).

Run from the repo root:  python -m src.error_analysis
Reads results/predictions/tfidf_*_dev.csv (dev set only) and writes:
  results/confusion_matrix_<model>.png      one figure per model (counts and row percentages)
  results/baseline_scores.csv               weighted F1, macro F1, accuracy per model
  results/baseline_class_scores.csv         precision, recall, F1 per class per model
  results/error_analysis.csv                every dev error of the best model, with automatic flags
                                            and, for 60 tweets, a category read by hand
  results/error_flag_rates.csv              how common each automatic flag is among errors vs correct tweets,
                                            with the Bonferroni threshold (0.05 / number of flags)
  results/error_summary.json                error counts, checked two ways (from rows and from the confusion matrix)
The hand categories live in results/error_categories_manual.csv (uid, category). That file is the first reader's
original labels and must not be edited; a second reader's labels go in a separate file (src/label_agreement.py).

Every merge is guarded by src/uid_checks.py: prediction files, the CMI table and the hand labels must contain
exactly the expected uids, so no row can be dropped, repeated or invented silently.
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
from src.uid_checks import check_uid_alignment

ROOT = Path(__file__).resolve().parents[1]
BEST = "tfidf_char_2-5"  # highest dev weighted F1 of the three baselines
ALPHA = 0.05
NEGATION = {"nahi", "nahin", "nhi", "nhin", "nahee", "nai", "na", "mat", "not", "no", "never", "dont", "don't",
            "cannot", "can't", "isn't", "aren't", "wasn't", "didn't", "doesn't", "won't"}
# URLs appear in two forms in this data: tokenised ("https // t . co / x", the colon was split off and lost)
# and ordinary ("https://example.com", "http://example.com"). "www." is also counted.
URL_PATTERN = re.compile(r"(?:https?\s*:?\s*//|\bwww\s*\.)", re.IGNORECASE)
# Heuristic, not complete emoji detection. It matches only the blocks U+1F300-U+1FAFF and U+2600-U+27BF. It misses
# emoji outside them (regional-indicator flags U+1F1E6-U+1F1FF, symbols such as U+203C or U+2B50). A multi-codepoint
# sequence is flagged if any one of its symbols is in range. Treat the flag as an approximation.
EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
NEUTRAL = LABELS.index("neutral")


def variant_words():
    cfg = json.loads((ROOT / "configs" / "hinglish_variants.json").read_text(encoding="utf-8"))
    return {v for k, vs in cfg.items() if not k.startswith("_") for v in vs}


def flags(df, variants):
    toks = df["text"].str.lower().str.replace("’", "'").str.split()
    out = pd.DataFrame(index=df.index)
    out["negation_word"] = toks.apply(lambda t: any(w in NEGATION for w in t))
    out["spelling_variant"] = toks.apply(lambda t: any(w in variants for w in t))
    out["truncated"] = df["text"].str.rstrip().str.endswith("…")
    out["url"] = df["text"].apply(lambda s: bool(URL_PATTERN.search(s)))
    out["repeated_letters"] = df["text"].apply(lambda s: bool(re.search(r"([a-zA-Z])\1{2,}", s)))
    out["has_emoji"] = df["text"].apply(lambda s: bool(EMOJI_PATTERN.search(s)))
    return out


def reading_sample_uids(dev, best_predictions, n=60, seed=42):
    """The uids of the random errors that were read by hand. Rows follow the DEV file order (the sample depends
    on row order), then `n` errors are drawn with `random_state=seed`."""
    m = dev[["uid", "label"]].merge(best_predictions[["uid", "pred"]], on="uid", how="inner", validate="one_to_one")
    return m[m["pred"] != m["label"]].sample(n, random_state=seed)["uid"].tolist()


def neutral_error_counts(cm):
    """From a 3x3 confusion matrix (rows true, columns predicted): errors, errors involving neutral, neg<->pos swaps."""
    cm = np.asarray(cm)
    total_errors = int(cm.sum() - np.trace(cm))
    true_neutral_wrong = int(cm[NEUTRAL].sum() - cm[NEUTRAL, NEUTRAL])
    predicted_neutral_wrong = int(cm[:, NEUTRAL].sum() - cm[NEUTRAL, NEUTRAL])
    involving_neutral = true_neutral_wrong + predicted_neutral_wrong
    return {"errors": total_errors, "errors_involving_neutral": involving_neutral,
            "share_involving_neutral": involving_neutral / total_errors,
            "negative_positive_swaps": total_errors - involving_neutral}


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
        check_uid_alignment(d["uid"], dev["uid"], name=p.name, reference_name="dev split")
        m = compute_metrics(d["label"], d["pred"])
        preds[name] = d
        score_rows.append({"model": name, "weighted_f1": m["weighted_f1"], "macro_f1": m["macro_f1"],
                           "accuracy": m["accuracy"]})
        for lab in LABELS:
            class_rows.append({"model": name, "class": lab, **m["per_class"][lab]})
        plot_confusion(confusion(d["label"], d["pred"]), f"{name} (dev)", ROOT / "results" / f"confusion_matrix_{name}.png")
    pd.DataFrame(score_rows).round(4).to_csv(ROOT / "results" / "baseline_scores.csv", index=False)
    pd.DataFrame(class_rows).round(4).to_csv(ROOT / "results" / "baseline_class_scores.csv", index=False)

    best = preds[BEST].merge(dev[["uid", "text", "n_tokens"]], on="uid", how="inner", validate="one_to_one")
    cmi = pd.read_csv(ROOT / "results" / "cmi_per_tweet.csv")
    cmi = cmi[cmi["split"] == "dev"][["uid", "cmi"]]
    check_uid_alignment(cmi["uid"], dev["uid"], name="cmi_per_tweet.csv (dev rows)", reference_name="dev split")
    best = best.merge(cmi, on="uid", how="inner", validate="one_to_one")
    assert len(best) == len(dev), "a merge changed the number of dev rows"
    f = flags(best, variant_words())
    best = pd.concat([best, f], axis=1)
    best["is_error"] = best["label"] != best["pred"]

    manual = pd.read_csv(ROOT / "results" / "error_categories_manual.csv")
    if set(reading_sample_uids(dev, preds[BEST])) != set(manual["uid"]):
        raise ValueError("error_categories_manual.csv is not the seed-42 random sample of 60 errors of " + BEST)
    if not manual["uid"].is_unique:
        raise ValueError("error_categories_manual.csv: duplicate uids")
    unknown = set(manual["uid"]) - set(best.loc[best["is_error"], "uid"])
    if unknown:
        raise ValueError(f"error_categories_manual.csv: {len(unknown)} uids are not errors of {BEST}, e.g. {sorted(unknown)[:5]}")
    best = best.merge(manual, on="uid", how="left", validate="one_to_one")
    errors = best[best["is_error"]].drop(columns="is_error")
    errors.rename(columns={"category": "manual_category"}).round(3).to_csv(ROOT / "results" / "error_analysis.csv", index=False)

    threshold = ALPHA / len(f.columns)
    rows = []
    for col in f.columns:
        e, c = best.loc[best.is_error, col], best.loc[~best.is_error, col]
        table = [[int(e.sum()), int((~e).sum())], [int(c.sum()), int((~c).sum())]]
        _, pval, _, _ = chi2_contingency(table)
        rows.append({"flag": col, "pct_of_errors": 100 * e.mean(), "pct_of_correct": 100 * c.mean(), "p_value": pval,
                     "bonferroni_threshold": threshold, "significant_after_bonferroni": bool(pval < threshold)})
    flag_rates = pd.DataFrame(rows)
    other = [c for c in flag_rates.select_dtypes("float") if c != "p_value"]
    flag_rates[other] = flag_rates[other].round(4)
    # keep 3 significant digits so a tiny p-value shows as 1.2e-06 and not as 0.0
    flag_rates["p_value"] = flag_rates["p_value"].map(lambda p: float(f"{p:.3g}"))
    flag_rates.to_csv(ROOT / "results" / "error_flag_rates.csv", index=False)

    # the share of errors that involve neutral, computed from the confusion matrix and again from the rows
    from_matrix = neutral_error_counts(confusion(best["label"], best["pred"]))
    wrong = best[best["is_error"]]
    from_rows = int(((wrong["label"] == "neutral") | (wrong["pred"] == "neutral")).sum())
    if from_rows != from_matrix["errors_involving_neutral"] or len(wrong) != from_matrix["errors"]:
        raise AssertionError(f"confusion matrix and rows disagree: {from_matrix} vs rows {from_rows}/{len(wrong)}")
    summary = {"model": BEST, **from_matrix, "share_involving_neutral": round(from_matrix["share_involving_neutral"], 4),
               "bonferroni_threshold": round(threshold, 5)}
    (ROOT / "results" / "error_summary.json").write_text(json.dumps(summary, indent=2))

    print(pd.DataFrame(score_rows).round(4).to_string(index=False))
    print(pd.DataFrame(class_rows).round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(json.dumps(summary))
    print("manual categories:", manual["category"].value_counts().to_dict(), "errors:", len(errors))


if __name__ == "__main__":
    main()
