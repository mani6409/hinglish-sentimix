"""Score dev predictions separately for low, medium and high code-mixing (issue #3).

Run from the repo root:  python -m src.cmi_evaluation
- Bucket edges are the 1/3 and 2/3 quantiles of CMI on TRAIN only (no dev/test information).
- Every results/predictions/*_dev.csv is joined to results/cmi_per_tweet.csv by uid. The join is checked:
  no duplicate, missing or unexpected uids in either file (see src/uid_checks.py).
- Writes results/cmi_bucket_scores.csv (scores per bucket, with a 95% bootstrap interval for weighted F1)
  results/cmi_bucket_pairwise.csv (difference in weighted F1 between two buckets, with a 95% interval) and
  results/cmi_bucket_group_summary.csv (mean and sd of weighted F1 per bucket over the seeds of one setting).

Bootstrap: 1,000 resamples. Each prediction file gets its own random generator, seeded from 42 and the file name,
so adding a new prediction file never changes the intervals of the others. Each resample draws, WITH replacement, as many tweets as the bucket
holds and recomputes weighted F1. The two buckets of a comparison are disjoint sets of tweets, so they are
resampled independently and the difference of the two resampled scores is taken.
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import LABELS, compute_metrics
from src.seed import DEFAULT_SEED, set_seed
from src.uid_checks import check_uid_alignment

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ["low", "medium", "high"]
PAIRS = [("low", "medium"), ("medium", "high"), ("low", "high")]
N_BOOT = 1000
K = len(LABELS)
LABEL_INDEX = {lab: i for i, lab in enumerate(LABELS)}


def bucket_edges(train_cmi):
    """Two cut points that split the TRAIN CMI values into three equal-sized groups."""
    return tuple(float(x) for x in np.quantile(train_cmi, [1 / 3, 2 / 3]))


def assign_bucket(cmi, edges):
    """Label each CMI value "low", "medium" or "high" using the two train cut points (edges are inclusive on the right)."""
    lo, hi = edges
    return pd.cut(cmi, bins=[-np.inf, lo, hi, np.inf], labels=BUCKETS)


def weighted_f1(y_true, y_pred):
    """Weighted F1 from integer labels (0..2). Same value as sklearn's with zero_division=0, but fast."""
    cm = np.bincount(y_true * K + y_pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    pred_n, true_n = cm.sum(axis=0).astype(float), cm.sum(axis=1).astype(float)
    denom = pred_n + true_n
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float((f1 * true_n).sum() / true_n.sum())


def resample_indices(n_obs, rng):
    """Indices of one bootstrap resample: n_obs draws WITH replacement from range(n_obs)."""
    return rng.integers(0, n_obs, size=n_obs)


def bootstrap_scores(y_true, y_pred, rng, n=N_BOOT):
    """Weighted F1 of `n` bootstrap resamples (array of length n)."""
    y_true = np.asarray([LABEL_INDEX[x] for x in y_true])
    y_pred = np.asarray([LABEL_INDEX[x] for x in y_pred])
    scores = np.empty(n)
    for b in range(n):
        idx = resample_indices(len(y_true), rng)
        scores[b] = weighted_f1(y_true[idx], y_pred[idx])
    return scores


def file_rng(name, seed=DEFAULT_SEED):
    """A random generator that depends only on the seed and the file name (stable when files are added)."""
    return np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])


def read_cmi_without_test(path, chunksize=2000):
    """Read the per-tweet CMI table keeping only train and dev rows: test rows are dropped chunk by chunk and
    are never held in memory or used."""
    kept = [c[c["split"] != "test"] for c in pd.read_csv(path, chunksize=chunksize)]
    return pd.concat(kept, ignore_index=True)


def summarize_by_group(scores):
    """Mean and sample standard deviation of weighted F1 per bucket over runs of the same kind.

    Runs are grouped by file name without the trailing seed (for example bilstm_full_raw_s42_seed42 and
    bilstm_full_raw_s43_seed43 form the group bilstm_full_raw). Only groups with more than one run are kept."""
    df = scores.copy()
    df["group"] = df["model"].str.replace(r"_s\d+_seed\d+$", "", regex=True)
    out = df.groupby(["group", "bucket"], sort=False)["weighted_f1"].agg(["count", "mean", "std"]).reset_index()
    out = out[out["count"] > 1].rename(columns={"count": "runs", "mean": "weighted_f1_mean", "std": "weighted_f1_sd"})
    return out


def interval(scores):
    """95% percentile interval of bootstrap scores: the 2.5th and 97.5th percentiles."""
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate_file(path, cmi, edges, rng=None):
    """Return (bucket score rows, pairwise difference rows) for one prediction file."""
    if rng is None:
        rng = file_rng(path.stem)
    pred = pd.read_csv(path)
    check_uid_alignment(pred["uid"], cmi["uid"], name=path.name, reference_name="cmi_per_tweet.csv (dev rows)")
    merged = pred.merge(cmi[["uid", "cmi"]], on="uid", how="inner", validate="one_to_one")
    assert len(merged) == len(pred) == len(cmi)
    merged["bucket"] = assign_bucket(merged["cmi"], edges)
    model = path.stem.replace("_dev", "")
    rows, boots = [], {}
    for name, g in [("all", merged)] + [(b, merged[merged["bucket"] == b]) for b in BUCKETS]:
        m = compute_metrics(g["label"], g["pred"])
        scores = bootstrap_scores(g["label"], g["pred"], rng)
        lo, hi = interval(scores)
        boots[name] = scores
        row = {"model": model, "bucket": name, "tweets": len(g),
               "cmi_min": round(float(g["cmi"].min()), 1), "cmi_max": round(float(g["cmi"].max()), 1),
               "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
               "weighted_f1_ci_low": lo, "weighted_f1_ci_high": hi}
        for lab in LABELS:
            row[f"precision_{lab}"] = m["per_class"][lab]["precision"]
            row[f"recall_{lab}"] = m["per_class"][lab]["recall"]
        rows.append(row)
    score = {r["bucket"]: r["weighted_f1"] for r in rows}
    pair_rows = []
    for a, b in PAIRS:
        diff = boots[a] - boots[b]
        lo, hi = interval(diff)
        pair_rows.append({"model": model, "comparison": f"{a} - {b}", "difference": score[a] - score[b],
                          "ci_low": lo, "ci_high": hi, "interval_includes_zero": bool(lo <= 0 <= hi)})
    return rows, pair_rows


def main():
    set_seed(DEFAULT_SEED)
    cmi = read_cmi_without_test(ROOT / "results" / "cmi_per_tweet.csv")
    check_uid_alignment(cmi["uid"], cmi["uid"], name="cmi_per_tweet.csv", reference_name="itself")
    edges = bucket_edges(cmi.loc[cmi["split"] == "train", "cmi"])
    dev_cmi = cmi[cmi["split"] == "dev"]
    print(f"bucket edges from train: low <= {edges[0]:.2f} < medium <= {edges[1]:.2f} < high")
    rows, pair_rows = [], []
    for path in sorted((ROOT / "results" / "predictions").glob("*_dev.csv")):
        r, p = evaluate_file(path, dev_cmi, edges)
        rows += r
        pair_rows += p
    out = pd.DataFrame(rows)
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(4)
    out.to_csv(ROOT / "results" / "cmi_bucket_scores.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    pnum = pairs.select_dtypes("float").columns
    pairs[pnum] = pairs[pnum].round(4)
    pairs.to_csv(ROOT / "results" / "cmi_bucket_pairwise.csv", index=False)
    summary = summarize_by_group(out)
    summary[["weighted_f1_mean", "weighted_f1_sd"]] = summary[["weighted_f1_mean", "weighted_f1_sd"]].round(4)
    summary.to_csv(ROOT / "results" / "cmi_bucket_group_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(out[["model", "bucket", "tweets", "weighted_f1", "weighted_f1_ci_low", "weighted_f1_ci_high"]].to_string(index=False))
    print(pairs.to_string(index=False))
    print(f"{int((~pairs['interval_includes_zero']).sum())} of {len(pairs)} pairwise intervals exclude zero")


if __name__ == "__main__":
    main()
