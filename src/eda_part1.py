"""EDA part 1: label distribution, tweet length, language shares, safety checks.

Run from the repo root:  python -m src.eda_part1
Writes figures to results/figures/ and numbers to results/eda_part1_stats.json.
Only train and dev are used for conclusions; the test labels are not looked at
here beyond the class counts the organisers published.
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_split
from src.metrics import LABELS

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NEAR_MONO = 0.90  # a tweet is "near-monolingual" if one language has >= 90% of its Hin+Eng tokens


def norm_text(s):
    """Lower-case the text and fix the spaces, so tweets that differ only in case or spacing match."""
    return " ".join(s.lower().split())


def length_stats(values):
    """Mean, median, 90/95/99th percentile, min and max of the lengths."""
    a = np.array(values)
    return {
        "mean": round(float(a.mean()), 2),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
        "min": int(a.min()),
    }


def dominant_share(langs):
    """Share of the most frequent language among Hin+Eng tokens (None if there are none)."""
    c = Counter(l for l in langs if l in ("Hin", "Eng"))
    n = sum(c.values())
    return None if n == 0 else max(c.values()) / n


def main():
    """Count labels, lengths and languages in all splits, check the data (empty tweets, duplicates,
    overlap between splits) and save the numbers and four figures."""
    # load all three splits; `out` collects every number we compute and is saved as JSON below
    splits = {s: load_split(s) for s in ("train", "dev", "test")}
    out = {}

    # 1. label distribution: how many tweets have each label, and the share in %
    out["labels"] = {}
    for s, df in splits.items():
        c = df["label"].value_counts()
        out["labels"][s] = {l: int(c.get(l, 0)) for l in LABELS}
        out["labels"][s]["total"] = len(df)
        out["labels"][s]["share_%"] = {l: round(100 * c.get(l, 0) / len(df), 1) for l in LABELS}

    # 2. length: tweet length in tokens and in characters, for each split
    out["length"] = {}
    for s, df in splits.items():
        out["length"][s] = {
            "tokens": length_stats(df["n_tokens"]),
            "chars": length_stats(df["text"].str.len()),
        }
    # average tweet length per label (train only)
    tr = splits["train"]
    out["length"]["train_mean_tokens_by_label"] = {
        l: round(float(tr.loc[tr["label"] == l, "n_tokens"].mean()), 2) for l in LABELS
    }

    # 3. language shares: how many tokens carry each language tag (Hin, Eng, O, EMT)
    out["language"] = {}
    for s, df in splits.items():
        c = Counter(l for langs in df["langs"] for l in langs)
        tot = sum(c.values())
        shares = {k: round(100 * v / tot, 1) for k, v in sorted(c.items())}
        # for each tweet: the share of its most common language (1.0 = only one language)
        dom = df["langs"].apply(dominant_share)
        has = dom.notna()  # tweets that have at least one Hin or Eng token
        out["language"][s] = {
            "token_counts": dict(c),
            "token_share_%": shares,
            "tweets_without_any_Hin_or_Eng": int((~has).sum()),
            f"near_monolingual_tweets(>={int(NEAR_MONO*100)}%)": int((dom[has] >= NEAR_MONO).sum()),
            "near_monolingual_%": round(100 * float((dom[has] >= NEAR_MONO).mean()), 1),
            "mean_dominant_share": round(float(dom[has].mean()), 3),
        }
    tr_dom = tr["langs"].apply(dominant_share)  # used for the histogram below

    # 4. safety checks: problems here would make the dev and test scores untrustworthy
    # compare the texts in lower case with clean spaces; `sets` holds the distinct texts of each split
    norm = {s: df["text"].apply(norm_text) for s, df in splits.items()}
    sets = {s: set(v) for s, v in norm.items()}
    # empty tweets, repeated texts, and the same text in two splits (that would be a leak)
    out["checks"] = {
        "empty_tweets": {s: int((df["n_tokens"] == 0).sum()) for s, df in splits.items()},
        "duplicate_texts_within_split": {s: int(len(v) - len(sets[s])) for s, v in norm.items()},
        "texts_in_both_train_and_dev": len(sets["train"] & sets["dev"]),
        "texts_in_both_train_and_test": len(sets["train"] & sets["test"]),
        "texts_in_both_dev_and_test": len(sets["dev"] & sets["test"]),
    }
    # do duplicated texts ever carry different labels inside train?
    g = tr.assign(t=norm["train"]).groupby("t")["label"].nunique()
    out["checks"]["train_texts_with_conflicting_labels"] = int((g > 1).sum())

    # save all the numbers
    (ROOT / "results" / "eda_part1_stats.json").write_text(json.dumps(out, indent=2))

    # ---- figures ----
    # figure 1: label share (%) in train, dev and test
    fig, ax = plt.subplots(figsize=(6, 3.6))
    w = 0.25  # width of one bar
    for i, s in enumerate(("train", "dev", "test")):
        ax.bar(np.arange(3) + i * w, [out["labels"][s]["share_%"][l] for l in LABELS], w, label=s)
    ax.set_xticks(np.arange(3) + w)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("% of tweets")
    ax.set_title("Label distribution per split")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "label_distribution.png", dpi=150); plt.close(fig)

    # figure 2: how long the train tweets are (red line = 95th percentile)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(tr["n_tokens"], bins=range(0, int(tr["n_tokens"].max()) + 2, 2), color="#2b4c9b")
    ax.axvline(out["length"]["train"]["tokens"]["p95"], color="red", ls="--", label="95th percentile")
    ax.set_xlabel("tokens per tweet (train)"); ax.set_ylabel("tweets"); ax.set_title("Tweet length")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "tweet_length_tokens.png", dpi=150); plt.close(fig)

    # figure 3: share of tokens with each language tag
    fig, ax = plt.subplots(figsize=(6, 3.6))
    keys = ["Hin", "Eng", "O", "EMT"]
    for i, s in enumerate(("train", "dev", "test")):
        ax.bar(np.arange(4) + i * w, [out["language"][s]["token_share_%"].get(k, 0) for k in keys], w, label=s)
    ax.set_xticks(np.arange(4) + w); ax.set_xticklabels(keys)
    ax.set_ylabel("% of tokens"); ax.set_title("Language tag share of tokens"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "language_shares.png", dpi=150); plt.close(fig)

    # figure 4: how mixed the train tweets are (1.0 = one language only)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(tr_dom.dropna(), bins=20, color="#2b4c9b")
    ax.set_xlabel("share of the dominant language among Hin+Eng tokens (train)")
    ax.set_ylabel("tweets"); ax.set_title("How mixed are the tweets? (1.0 = one language only)")
    fig.tight_layout(); fig.savefig(FIG / "dominant_language_share.png", dpi=150); plt.close(fig)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
