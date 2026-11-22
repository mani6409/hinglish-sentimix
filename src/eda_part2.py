"""EDA part 2: CMI per tweet, CMI vs label, and hand-reading samples.

Run from the repo root:  python -m src.eda_part2
Outputs:
  results/cmi_per_tweet.csv        uid, split, label, cmi
  results/eda_part2_stats.json     summary numbers
  results/figures/cmi_*.png        plots
  docs/reading_sample.md           150 train tweets (50 per class) to read by hand
  docs/near_monolingual_tweets.md  train tweets where one language has >= 90% of Hin+Eng tokens

Purpose:
    Computes the Code-Mixing Index (CMI) of every tweet, summarises how CMI relates to the sentiment label,
    and prepares two reading aids for the team: a random sample of tweets per class and the list of
    almost single-language tweets.

Why this file exists:
    Our research question asks whether the amount of code-mixing changes accuracy. This script produces the
    per-tweet CMI values (the file everything else reads) and the first descriptive numbers.

Place in the pipeline:
    src/data.py (load_split) + src/cmi.py (cmi)  ->  this script  ->  results/cmi_per_tweet.csv
    ->  read by src/cmi_evaluation.py, which builds the low/medium/high buckets and scores the models.

Inputs:
    All three splits through ``load_split``. The train split drives the statistics (except the mean CMI per
    split, which uses all three), the figures and the samples; the dev and test CMI values are computed and
    saved in the CSV.

Outputs (see the list at the top of the file):
    * results/cmi_per_tweet.csv has the columns uid, split, label, cmi (3 decimals) for ALL 20,000 tweets,
      including the test rows and their labels. src/cmi_evaluation.py therefore drops the test rows when it
      reads the file.
    * results/eda_part2_stats.json holds the mean CMI per split next to the paper's averages
      (``PAPER_AVG``), CMI quantiles of train, CMI by label and the label shares in the CMI terciles.
    * results/figures/cmi_distribution.png and cmi_by_label.png.
    * docs/reading_sample.md and docs/near_monolingual_tweets.md (Markdown tables).

Used by:
    Nothing imports this module. Its outputs are read by src/cmi_evaluation.py (the CSV) and quoted in
    docs/eda_part2.md.

Depends on:
    src/cmi.py, src/data.py, src/metrics.py (class order), src/seed.py, pandas, numpy, matplotlib.

Notes for maintainers:
    * Our mean CMI (about 29.8 on train) is higher than the paper's 25.3. See docs/eda_part2.md; the
      exact definition used by the organisers is an open question.
    * ``dominant_share`` and ``md_escape`` are also defined in src/eda_part1.py and
      src/tfidf_error_analysis.py (same meaning; open point: they could be shared).
    * The 90% threshold for "near-monolingual" is written as a literal 0.9 here and as ``NEAR_MONO`` in
      src/eda_part1.py.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cmi import cmi
from src.data import load_split
from src.metrics import LABELS
from src.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
PAPER_AVG = {"train": 25.32, "dev": 25.53, "test": 25.13}  # average CMI reported in the task paper (Patwa et al., 2020)


def dominant_share(langs):
    """Share of the most frequent language among Hin+Eng tokens (None if there are none).

    Same meaning as ``dominant_share`` in src/eda_part1.py.

    Args:
        langs: List of language tags of one tweet.

    Returns:
        A float in [0.5, 1.0] (1.0 = one language only), or None when the tweet has no Hin or Eng token.
    """
    toks = [l for l in langs if l in ("Hin", "Eng")]
    if not toks:
        return None
    return max(toks.count("Hin"), toks.count("Eng")) / len(toks)


def md_escape(s):
    """Make a tweet safe to put into a Markdown table cell.

    Args:
        s: Tweet text.

    Returns:
        The text with ``|`` written as ``\\|`` (a bare bar would start a new column) and newlines replaced by a
        space (a newline would end the table row).
    """
    return s.replace("|", "\\|").replace("\n", " ")


def main():
    """Compute CMI for every tweet and write the CSV, statistics, figures and the two reading files.

    Stages:
        1. Load the three splits, compute ``cmi`` per tweet from its language tags and save
           results/cmi_per_tweet.csv.
        2. Compute the statistics of the TRAIN split: mean CMI per split against the paper, quantiles, CMI
           by label, and label shares inside three equal-sized CMI groups (``qcut`` terciles; groups with
           identical edges would be merged, ``duplicates="drop"``).
        3. Draw the CMI histogram and the CMI-by-label box plot.
        4. Write the hand-reading sample: 50 random train tweets per class (``random_state=42``).
        5. Write the near-monolingual list: train tweets where one language holds at least 90% of the
           Hin+Eng tokens (tweets without any Hin or Eng token have no share and are not selected).

    Side effects:
        Overwrites the files listed at the top of the module and prints the statistics as JSON. The random
        seed is fixed (``set_seed(42)``), and the sample seed is the same value.
    """
    set_seed(42)
    FIG.mkdir(parents=True, exist_ok=True)
    frames = []
    for s in ("train", "dev", "test"):
        df = load_split(s)
        df["cmi"] = df["langs"].apply(cmi)
        df["split"] = s
        frames.append(df)
    allx = pd.concat(frames, ignore_index=True)
    allx[["uid", "split", "label", "cmi"]].round(3).to_csv(ROOT / "results" / "cmi_per_tweet.csv", index=False)

    tr = allx[allx["split"] == "train"]
    out = {"note": "standard CMI formula, language-independent tags: O and EMT"}
    out["mean_cmi"] = {s: round(float(allx.loc[allx.split == s, "cmi"].mean()), 2) for s in PAPER_AVG}
    out["paper_mean_cmi"] = PAPER_AVG
    q = tr["cmi"].quantile([0, .1, .25, .5, .75, .9, 1]).round(2)
    out["train_cmi_quantiles"] = {str(k): float(v) for k, v in q.items()}
    out["train_tweets_with_cmi_zero"] = int((tr["cmi"] == 0).sum())
    out["train_cmi_by_label"] = {
        l: {
            "mean": round(float(tr.loc[tr.label == l, "cmi"].mean()), 2),
            "median": round(float(tr.loc[tr.label == l, "cmi"].median()), 2),
            "std": round(float(tr.loc[tr.label == l, "cmi"].std()), 2),
        }
        for l in LABELS
    }
    out["train_label_share_by_cmi_tercile_%"] = {}
    # Three groups with (about) the same number of tweets, cut at CMI quantiles of the training data.
    tr = tr.assign(tercile=pd.qcut(tr["cmi"], 3, labels=["low", "mid", "high"], duplicates="drop"))
    for t, g in tr.groupby("tercile", observed=True):
        out["train_label_share_by_cmi_tercile_%"][str(t)] = {
            "cmi_range": [round(float(g["cmi"].min()), 1), round(float(g["cmi"].max()), 1)],
            "tweets": len(g),
            **{l: round(100 * float((g.label == l).mean()), 1) for l in LABELS},
        }
    (ROOT / "results" / "eda_part2_stats.json").write_text(json.dumps(out, indent=2))

    # figures
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(tr["cmi"], bins=30, color="#2b4c9b")
    ax.axvline(tr["cmi"].mean(), color="red", ls="--", label=f"mean {tr['cmi'].mean():.1f}")
    ax.set_xlabel("CMI (train)"); ax.set_ylabel("tweets"); ax.set_title("Code-Mixing Index distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "cmi_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.boxplot([tr.loc[tr.label == l, "cmi"] for l in LABELS], tick_labels=LABELS, showmeans=True)
    ax.set_ylabel("CMI (train)"); ax.set_title("CMI by sentiment label")
    fig.tight_layout(); fig.savefig(FIG / "cmi_by_label.png", dpi=150); plt.close(fig)

    # reading sample: 50 per class, fixed seed
    lines = [
        "# Hand-reading sample (train): 50 tweets per class",
        "",
        "Read each tweet and note what you see: emojis, slang, repeated letters, mixed script, sarcasm, negation,",
        "or a label that looks wrong. Fill the last column. Sample drawn with seed 42 (`python -m src.eda_part2`).",
        "",
    ]
    for l in LABELS:
        g = tr[tr.label == l].sample(50, random_state=42)
        lines += [f"## {l}", "", "| # | uid | CMI | tweet | my notes |", "|---|---|---|---|---|"]
        for i, r in enumerate(g.itertuples(), 1):
            lines.append(f"| {i} | {r.uid} | {r.cmi:.0f} | {md_escape(r.text)} | |")
        lines.append("")
    (ROOT / "docs" / "reading_sample.md").write_text("\n".join(lines), encoding="utf-8")

    # near-monolingual tweets
    dom = tr["langs"].apply(dominant_share)
    nm = tr[dom >= 0.9]
    lines = ["# Near-monolingual training tweets (one language has >= 90% of Hin+Eng tokens)", "",
             f"{len(nm)} tweets.", "", "| uid | label | CMI | tweet |", "|---|---|---|---|"]
    for r in nm.itertuples():
        lines.append(f"| {r.uid} | {r.label} | {r.cmi:.0f} | {md_escape(r.text)} |")
    (ROOT / "docs" / "near_monolingual_tweets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
