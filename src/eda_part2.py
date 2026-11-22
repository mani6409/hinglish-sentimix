"""EDA part 2: CMI per tweet, CMI vs label, and hand-reading samples.

Run from the repo root:  python -m src.eda_part2
Outputs:
  results/cmi_per_tweet.csv        uid, split, label, cmi
  results/eda_part2_stats.json     summary numbers
  results/figures/cmi_*.png        plots
  docs/reading_sample.md           150 train tweets (50 per class) to read by hand
  docs/near_monolingual_tweets.md  train tweets where one language has >= 90% of Hin+Eng tokens
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
PAPER_AVG = {"train": 25.32, "dev": 25.53, "test": 25.13}


def dominant_share(langs):
    """Share of the most common language among Hin and Eng tokens (None if there are none)."""
    toks = [l for l in langs if l in ("Hin", "Eng")]
    if not toks:
        return None
    return max(toks.count("Hin"), toks.count("Eng")) / len(toks)


def md_escape(s):
    """Make a tweet safe for a Markdown table (no '|' and no new lines)."""
    return s.replace("|", "\\|").replace("\n", " ")


def main():
    """Compute CMI for every tweet, summarise it, draw two figures and write two documents to read by hand."""
    set_seed(42)
    FIG.mkdir(parents=True, exist_ok=True)
    # step 1: compute the CMI of every tweet in every split and put everything in one table
    frames = []
    for s in ("train", "dev", "test"):
        df = load_split(s)
        df["cmi"] = df["langs"].apply(cmi)
        df["split"] = s
        frames.append(df)
    allx = pd.concat(frames, ignore_index=True)
    # save uid, split, label and CMI of every tweet (src/cmi_evaluation.py reads this file later)
    allx[["uid", "split", "label", "cmi"]].round(3).to_csv(ROOT / "results" / "cmi_per_tweet.csv", index=False)

    # step 2: summary numbers; from here on the statistics use the train rows only
    tr = allx[allx["split"] == "train"]
    out = {"note": "standard CMI formula, language-independent tags: O and EMT"}
    # our mean CMI per split, next to the mean published in the paper (a quick check that we match it)
    out["mean_cmi"] = {s: round(float(allx.loc[allx.split == s, "cmi"].mean()), 2) for s in PAPER_AVG}
    out["paper_mean_cmi"] = PAPER_AVG
    # how spread out the CMI values are in train
    q = tr["cmi"].quantile([0, .1, .25, .5, .75, .9, 1]).round(2)
    out["train_cmi_quantiles"] = {str(k): float(v) for k, v in q.items()}
    out["train_tweets_with_cmi_zero"] = int((tr["cmi"] == 0).sum())
    # does CMI differ between negative, neutral and positive tweets?
    out["train_cmi_by_label"] = {
        l: {
            "mean": round(float(tr.loc[tr.label == l, "cmi"].mean()), 2),
            "median": round(float(tr.loc[tr.label == l, "cmi"].median()), 2),
            "std": round(float(tr.loc[tr.label == l, "cmi"].std()), 2),
        }
        for l in LABELS
    }
    # split train into three equal groups by CMI (low, mid, high) and show the label mix in each group
    out["train_label_share_by_cmi_tercile_%"] = {}
    tr = tr.assign(tercile=pd.qcut(tr["cmi"], 3, labels=["low", "mid", "high"], duplicates="drop"))
    for t, g in tr.groupby("tercile", observed=True):
        out["train_label_share_by_cmi_tercile_%"][str(t)] = {
            "cmi_range": [round(float(g["cmi"].min()), 1), round(float(g["cmi"].max()), 1)],
            "tweets": len(g),
            **{l: round(100 * float((g.label == l).mean()), 1) for l in LABELS},
        }
    # save the numbers
    (ROOT / "results" / "eda_part2_stats.json").write_text(json.dumps(out, indent=2))

    # step 3: figures
    # figure 1: how CMI is spread in train (red line = mean)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(tr["cmi"], bins=30, color="#2b4c9b")
    ax.axvline(tr["cmi"].mean(), color="red", ls="--", label=f"mean {tr['cmi'].mean():.1f}")
    ax.set_xlabel("CMI (train)"); ax.set_ylabel("tweets"); ax.set_title("Code-Mixing Index distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "cmi_distribution.png", dpi=150); plt.close(fig)

    # figure 2: CMI for each label as a box plot (triangle = mean)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.boxplot([tr.loc[tr.label == l, "cmi"] for l in LABELS], tick_labels=LABELS, showmeans=True)
    ax.set_ylabel("CMI (train)"); ax.set_title("CMI by sentiment label")
    fig.tight_layout(); fig.savefig(FIG / "cmi_by_label.png", dpi=150); plt.close(fig)

    # step 4: a document of train tweets to read by hand: 50 per class, picked with a fixed seed
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

    # step 5: list the train tweets that are almost one language (>= 90% of the Hin+Eng tokens)
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
