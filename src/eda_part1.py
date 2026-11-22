"""EDA part 1: label distribution, tweet length, language shares, safety checks.

Run from the repo root:  python -m src.eda_part1
Writes figures to results/figures/ and numbers to results/eda_part1_stats.json.
Only train and dev are used for conclusions; the test labels are not looked at
here beyond the class counts the organisers published.

Purpose:
    First look at the data before any model is trained: how balanced the three classes are, how long tweets
    are, how much Hindi, English and "other" they contain, and whether the splits overlap.

Why this file exists:
    Choices later in the project depend on these facts: the metric (weighted F1 suits the class mix), the
    input length of the models, and the confidence that no tweet leaks from train into dev or test.

Place in the pipeline:
    src/data.py (load_split)  ->  this script  ->  results/eda_part1_stats.json + four PNG figures.
    docs/eda_part1.md summarises these numbers; it was written by hand from the JSON file, it is not generated.

Inputs:
    The three official splits through ``load_split`` (train, dev, test). The test split is loaded because its
    class counts, length, language tags and text overlap with train and dev are part of the checks. Labels
    of the test split are used only to count classes.

Outputs:
    results/eda_part1_stats.json (sections ``labels``, ``length``, ``language``, ``checks``) and the figures
    label_distribution.png, tweet_length_tokens.png, language_shares.png, dominant_language_share.png in
    results/figures/. The JSON is also printed.

Used by:
    Nothing imports this module; it is a command-line script that is run once. Its numbers are quoted in
    docs/eda_part1.md and in the report draft.

Depends on:
    src/data.py, src/metrics.py (only for the fixed class order ``LABELS``), matplotlib, numpy.

Notes for maintainers:
    * Importing this module creates results/figures/ (a side effect at import time, line ``FIG.mkdir``).
    * ``matplotlib.use("Agg")`` is called before pyplot is imported so the script runs without a display.
    * ``dominant_share`` is implemented again, with the same meaning, in src/eda_part2.py (open point:
      the duplicate could be shared).
    * Rerunning overwrites the JSON and the four figures; the numbers in the JSON are the same for the same data.
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
    """Normalise a tweet for duplicate detection: lower-case and collapse runs of whitespace.

    Purpose:
        Two tweets that differ only in capital letters or spacing count as the same text in the overlap
        checks.

    Args:
        s: Tweet text (str).

    Returns:
        The text in lower case with single spaces between words.

    Example:
        norm_text("Hello   World")  ->  "hello world"
    """
    return " ".join(s.lower().split())


def length_stats(values):
    """Summarise a list of lengths (tokens or characters) with the numbers used in the report.

    Args:
        values: Non-empty sequence of numbers.

    Returns:
        A dict with ``mean`` (rounded to 2 decimals), ``median``, the percentiles ``p90``, ``p95`` and
        ``p99`` (NumPy's default linear interpolation), ``max`` and ``min`` (ints).

    Raises:
        ValueError: ``values`` is empty (NumPy cannot take the maximum of an empty array).
    """
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
    """Share of the most frequent language among Hin+Eng tokens (None if there are none).

    Purpose:
        Measures how one-sided a tweet is: 1.0 means all language tokens are in one language, 0.5 means an
        even split between Hindi and English. Tags ``O`` (other) and ``EMT`` (emoticon) are ignored.

    Args:
        langs: List of language tags of one tweet.

    Returns:
        A float in [0.5, 1.0], or None if the tweet has no Hindi or English token.

    Example:
        dominant_share(["Hin", "Hin", "Eng", "O"])  ->  0.666...
    """
    c = Counter(l for l in langs if l in ("Hin", "Eng"))
    n = sum(c.values())
    return None if n == 0 else max(c.values()) / n


def main():
    """Compute all statistics, write the JSON file and the figures, and print the JSON.

    Stages (each fills one section of the output dictionary ``out``):
        1. ``labels``: class counts and percentages per split.
        2. ``length``: token and character length statistics per split, plus the mean token length per label
           in train.
        3. ``language``: share of each language tag among all tokens, and how many tweets are almost
           single-language (dominant share >= ``NEAR_MONO``).
        4. ``checks``: empty tweets, duplicate texts inside a split, texts shared between splits, and
           duplicated train texts that carry different labels. These checks use ``norm_text``.
        Then the four figures are drawn from the same numbers.

    Side effects:
        Writes results/eda_part1_stats.json and four PNG files in results/figures/; prints the JSON.
    """
    splits = {s: load_split(s) for s in ("train", "dev", "test")}
    out = {}

    # 1. label distribution
    out["labels"] = {}
    for s, df in splits.items():
        c = df["label"].value_counts()
        out["labels"][s] = {l: int(c.get(l, 0)) for l in LABELS}
        out["labels"][s]["total"] = len(df)
        out["labels"][s]["share_%"] = {l: round(100 * c.get(l, 0) / len(df), 1) for l in LABELS}

    # 2. length
    out["length"] = {}
    for s, df in splits.items():
        out["length"][s] = {
            "tokens": length_stats(df["n_tokens"]),
            "chars": length_stats(df["text"].str.len()),
        }
    tr = splits["train"]
    out["length"]["train_mean_tokens_by_label"] = {
        l: round(float(tr.loc[tr["label"] == l, "n_tokens"].mean()), 2) for l in LABELS
    }

    # 3. language shares
    out["language"] = {}
    for s, df in splits.items():
        c = Counter(l for langs in df["langs"] for l in langs)
        tot = sum(c.values())
        shares = {k: round(100 * v / tot, 1) for k, v in sorted(c.items())}
        dom = df["langs"].apply(dominant_share)
        has = dom.notna()  # tweets with at least one Hin or Eng token; the others have no dominant share
        out["language"][s] = {
            "token_counts": dict(c),
            "token_share_%": shares,
            "tweets_without_any_Hin_or_Eng": int((~has).sum()),
            f"near_monolingual_tweets(>={int(NEAR_MONO*100)}%)": int((dom[has] >= NEAR_MONO).sum()),
            "near_monolingual_%": round(100 * float((dom[has] >= NEAR_MONO).mean()), 1),
            "mean_dominant_share": round(float(dom[has].mean()), 3),
        }
    tr_dom = tr["langs"].apply(dominant_share)

    # 4. safety checks
    norm = {s: df["text"].apply(norm_text) for s, df in splits.items()}
    sets = {s: set(v) for s, v in norm.items()}
    out["checks"] = {
        "empty_tweets": {s: int((df["n_tokens"] == 0).sum()) for s, df in splits.items()},
        # tweets minus distinct texts = number of surplus copies inside the split
        "duplicate_texts_within_split": {s: int(len(v) - len(sets[s])) for s, v in norm.items()},
        "texts_in_both_train_and_dev": len(sets["train"] & sets["dev"]),
        "texts_in_both_train_and_test": len(sets["train"] & sets["test"]),
        "texts_in_both_dev_and_test": len(sets["dev"] & sets["test"]),
    }
    # do duplicated texts ever carry different labels inside train?
    g = tr.assign(t=norm["train"]).groupby("t")["label"].nunique()
    out["checks"]["train_texts_with_conflicting_labels"] = int((g > 1).sum())

    (ROOT / "results" / "eda_part1_stats.json").write_text(json.dumps(out, indent=2))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(6, 3.6))
    w = 0.25  # bar width; the three splits are drawn side by side
    for i, s in enumerate(("train", "dev", "test")):
        ax.bar(np.arange(3) + i * w, [out["labels"][s]["share_%"][l] for l in LABELS], w, label=s)
    ax.set_xticks(np.arange(3) + w)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("% of tweets")
    ax.set_title("Label distribution per split")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "label_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(tr["n_tokens"], bins=range(0, int(tr["n_tokens"].max()) + 2, 2), color="#2b4c9b")
    ax.axvline(out["length"]["train"]["tokens"]["p95"], color="red", ls="--", label="95th percentile")
    ax.set_xlabel("tokens per tweet (train)"); ax.set_ylabel("tweets"); ax.set_title("Tweet length")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "tweet_length_tokens.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    keys = ["Hin", "Eng", "O", "EMT"]
    for i, s in enumerate(("train", "dev", "test")):
        ax.bar(np.arange(4) + i * w, [out["language"][s]["token_share_%"].get(k, 0) for k in keys], w, label=s)
    ax.set_xticks(np.arange(4) + w); ax.set_xticklabels(keys)
    ax.set_ylabel("% of tokens"); ax.set_title("Language tag share of tokens"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "language_shares.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(tr_dom.dropna(), bins=20, color="#2b4c9b")
    ax.set_xlabel("share of the dominant language among Hin+Eng tokens (train)")
    ax.set_ylabel("tweets"); ax.set_title("How mixed are the tweets? (1.0 = one language only)")
    fig.tight_layout(); fig.savefig(FIG / "dominant_language_share.png", dpi=150); plt.close(fig)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
