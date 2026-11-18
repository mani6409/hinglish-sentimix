"""Score dev predictions separately for low, medium and high code-mixing (issue #3).

Run from the repo root:  python -m src.cmi_evaluation
- Bucket edges are the 1/3 and 2/3 quantiles of CMI on TRAIN only (no dev/test information).
- Every results/predictions/*_dev.csv file is joined to results/cmi_per_tweet.csv by uid.
- Writes results/cmi_bucket_scores.csv. A bootstrap (1,000 resamples, seed 42) gives a 95% interval
  for weighted F1 in each bucket, because a bucket holds only about 1,000 tweets.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import LABELS, compute_metrics
from src.seed import DEFAULT_SEED, set_seed

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ["low", "medium", "high"]
N_BOOT = 1000


def bucket_edges(train_cmi):
    """Two cut points that split the TRAIN CMI values into three equal-sized groups."""
    return tuple(float(x) for x in np.quantile(train_cmi, [1 / 3, 2 / 3]))


def assign_bucket(cmi, edges):
    lo, hi = edges
    return pd.cut(cmi, bins=[-np.inf, lo, hi, np.inf], labels=BUCKETS)


def bootstrap_ci(y_true, y_pred, rng, n=N_BOOT):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(compute_metrics(y_true[idx], y_pred[idx])["weighted_f1"])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate_file(path, cmi, edges, rng):
    pred = pd.read_csv(path)
    merged = pred.merge(cmi[["uid", "cmi"]], on="uid", how="left", validate="one_to_one")
    if merged["cmi"].isna().any():
        raise ValueError(f"{path.name}: {int(merged['cmi'].isna().sum())} uids have no CMI value")
    merged["bucket"] = assign_bucket(merged["cmi"], edges)
    rows = []
    for name, g in [("all", merged)] + [(b, merged[merged["bucket"] == b]) for b in BUCKETS]:
        m = compute_metrics(g["label"], g["pred"])
        lo, hi = bootstrap_ci(g["label"], g["pred"], rng)
        row = {"model": path.stem.replace("_dev", ""), "bucket": name, "tweets": len(g),
               "cmi_min": round(float(g["cmi"].min()), 1), "cmi_max": round(float(g["cmi"].max()), 1),
               "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
               "weighted_f1_ci_low": lo, "weighted_f1_ci_high": hi}
        for lab in LABELS:
            row[f"precision_{lab}"] = m["per_class"][lab]["precision"]
            row[f"recall_{lab}"] = m["per_class"][lab]["recall"]
        rows.append(row)
    return rows


def main():
    set_seed(DEFAULT_SEED)
    rng = np.random.default_rng(DEFAULT_SEED)
    cmi = pd.read_csv(ROOT / "results" / "cmi_per_tweet.csv")
    edges = bucket_edges(cmi.loc[cmi["split"] == "train", "cmi"])
    dev_cmi = cmi[cmi["split"] == "dev"]
    print(f"bucket edges from train: low <= {edges[0]:.2f} < medium <= {edges[1]:.2f} < high")
    rows = []
    for path in sorted((ROOT / "results" / "predictions").glob("*_dev.csv")):
        rows += evaluate_file(path, dev_cmi, edges, rng)
    out = pd.DataFrame(rows)
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(4)
    out.to_csv(ROOT / "results" / "cmi_bucket_scores.csv", index=False)
    print(out[["model", "bucket", "tweets", "cmi_min", "cmi_max", "accuracy", "macro_f1", "weighted_f1",
               "weighted_f1_ci_low", "weighted_f1_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
