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

Purpose:
    Answers the research question "does the amount of code-mixing change how well a model classifies
    sentiment?" by scoring each model's dev predictions in three groups of tweets (low, medium, high CMI) and
    asking, with a bootstrap, whether the groups differ by more than sampling noise.

Why this file exists:
    A single dev score hides whether a model is better on lightly or heavily mixed tweets. Each bucket holds
    only about 1,000 tweets, so a difference of a few points can be chance; the intervals prevent
    over-reading it (docs/cmi_analysis.md explains the findings).

Place in the pipeline:
    results/cmi_per_tweet.csv (from src/eda_part2.py)  +  results/predictions/*_dev.csv (from the model scripts)
    ->  this script  ->  three CSV files in results/  ->  docs/cmi_analysis.md and the report.

Inputs:
    * results/cmi_per_tweet.csv, read without its test rows (see ``read_cmi_without_test``).
    * Every file matching results/predictions/*_dev.csv with the columns uid, label, pred and exactly the
      3,000 dev uids. A file that covers only a subset of dev (for example a 200-tweet sample) stops the run
      with a uid-alignment error, by design.

Outputs:
    cmi_bucket_scores.csv (per model and bucket: accuracy, macro and weighted F1, per-class precision and
    recall, 95% interval of weighted F1), cmi_bucket_pairwise.csv (differences between buckets with 95%
    intervals) and cmi_bucket_group_summary.csv (mean and sample sd over runs that differ only in seed).
    A summary is also printed.

Used by:
    Run from the command line, and again whenever new dev prediction files are added. Imported by
    src/error_analysis_models.py (``assign_bucket``, ``bucket_edges``, ``read_cmi_without_test``),
    src/transformer_comparison.py, tests/test_cmi_evaluation.py and tests/test_error_analysis_models.py.

Depends on:
    src/metrics.py (``compute_metrics``, ``LABELS``), src/seed.py, src/uid_checks.py, numpy, pandas.

Design decisions:
    * Bucket cut points come from train only, so no dev or test information defines the groups.
    * A fast exact weighted F1 (``weighted_f1``) is used inside the bootstrap because calling scikit-learn
      four groups of 1,000 resamples per prediction file (4,000 calls per file) was slow; a test checks that it equals ``compute_metrics``.
    * The random generator of each file is seeded from the file name with CRC32 (not Python's ``hash``, which
      changes between runs), so results repeat exactly.
    * Intervals are not corrected for the number of comparisons (3 comparisons per prediction file; 96 intervals for the 32 files in results/predictions/ when this was written).

Limits:
    Every tweet in this data set is mixed, so "low" means moderately mixed, not single-language.
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
N_BOOT = 1000  # bootstrap resamples per bucket
K = len(LABELS)  # number of classes
LABEL_INDEX = {lab: i for i, lab in enumerate(LABELS)}  # label string -> integer 0..2 for fast counting


def bucket_edges(train_cmi):
    """Two cut points that split the TRAIN CMI values into three equal-sized groups.

    Args:
        train_cmi: Series or array of CMI values of the training tweets.

    Returns:
        ``(low_edge, high_edge)``: the 1/3 and 2/3 quantiles (linear interpolation) as floats. On this data
        they are 23.81 and 34.78.
    """
    return tuple(float(x) for x in np.quantile(train_cmi, [1 / 3, 2 / 3]))


def assign_bucket(cmi, edges):
    """Put each tweet into the low, medium or high bucket.

    Args:
        cmi: Series of CMI values (any split).
        edges: ``(low_edge, high_edge)`` from ``bucket_edges``.

    Returns:
        A categorical Series with the values ``low`` (CMI <= low_edge), ``medium`` (low_edge < CMI <=
        high_edge) or ``high`` (CMI > high_edge); the right edge of each interval is included.
    """
    lo, hi = edges
    return pd.cut(cmi, bins=[-np.inf, lo, hi, np.inf], labels=BUCKETS)


def weighted_f1(y_true, y_pred):
    """Weighted F1 from integer labels (0..2). Same value as sklearn's with zero_division=0, but fast.

    How it works:
        Build the 3x3 confusion matrix with one ``bincount`` (index = true * K + predicted; rows are true
        labels, columns predicted). Per class, F1 = 2 * TP / (predicted count + true count), which equals
        the harmonic mean of precision and recall; it is 0 when the class never occurs in either list. The
        class F1 values are averaged with weights equal to the number of true tweets of the class.

    Args:
        y_true: NumPy int array of gold labels (0 = negative, 1 = neutral, 2 = positive).
        y_pred: NumPy int array of predictions, same length.

    Returns:
        The weighted F1 as a float.
    """
    cm = np.bincount(y_true * K + y_pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    pred_n, true_n = cm.sum(axis=0).astype(float), cm.sum(axis=1).astype(float)
    denom = pred_n + true_n
    # where denom == 0 the class is absent everywhere: keep F1 at 0 instead of dividing by zero
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float((f1 * true_n).sum() / true_n.sum())


def resample_indices(n_obs, rng):
    """Indices of one bootstrap resample: n_obs draws WITH replacement from range(n_obs).

    Args:
        n_obs: Number of tweets in the bucket.
        rng: A NumPy ``Generator``.

    Returns:
        An int array of length ``n_obs``; the same tweet can appear several times and some not at all.
    """
    return rng.integers(0, n_obs, size=n_obs)


def bootstrap_scores(y_true, y_pred, rng, n=N_BOOT):
    """Weighted F1 of ``n`` bootstrap resamples.

    Args:
        y_true: Sequence of gold label strings.
        y_pred: Sequence of predicted label strings, same order.
        rng: A NumPy ``Generator`` (consumed; its state advances).
        n: Number of resamples.

    Returns:
        A float array of length ``n``; its 2.5th and 97.5th percentiles form the 95% interval.

    Raises:
        KeyError: A label is not one of ``LABELS``.
    """
    y_true = np.asarray([LABEL_INDEX[x] for x in y_true])
    y_pred = np.asarray([LABEL_INDEX[x] for x in y_pred])
    scores = np.empty(n)
    for b in range(n):
        # The same resampled tweets are used for truth and prediction, so each resample is a fair new "test set".
        idx = resample_indices(len(y_true), rng)
        scores[b] = weighted_f1(y_true[idx], y_pred[idx])
    return scores


def file_rng(name, seed=DEFAULT_SEED):
    """A random generator that depends only on the seed and the file name (stable when files are added).

    Args:
        name: File stem, for example ``tfidf_char_2-5_dev``.
        seed: Base seed.

    Returns:
        A NumPy ``Generator`` seeded with ``[seed, crc32(name)]``. CRC32 is deterministic across runs and
        machines, unlike Python's built-in ``hash`` for strings.
    """
    return np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])


def read_cmi_without_test(path, chunksize=2000):
    """Read the per-tweet CMI table keeping only train and dev rows: test rows are dropped chunk by chunk and
    are never held in memory or used.

    Why: results/cmi_per_tweet.csv also contains the test tweets (written by src/eda_part2.py). This analysis
    must not use them, so they are discarded as soon as each chunk is read. (The rows are still read from disk.)

    Args:
        path: Path of ``cmi_per_tweet.csv`` (columns uid, split, label, cmi).
        chunksize: Number of rows read at a time.

    Returns:
        A DataFrame with the train and dev rows.
    """
    kept = [c[c["split"] != "test"] for c in pd.read_csv(path, chunksize=chunksize)]
    return pd.concat(kept, ignore_index=True)


def summarize_by_group(scores):
    """Mean and sample standard deviation of weighted F1 per bucket over runs of the same kind.

    Runs are grouped by file name without the trailing seed (for example bilstm_full_raw_s42_seed42 and
    bilstm_full_raw_s43_seed43 form the group bilstm_full_raw). Only groups with more than one run are kept.

    Args:
        scores: The rows of cmi_bucket_scores.csv (needs columns ``model``, ``bucket``, ``weighted_f1``).

    Returns:
        A DataFrame with columns group, bucket, runs, weighted_f1_mean, weighted_f1_sd. The sd uses pandas'
        default (n - 1), the sample standard deviation. Models with a single run (the TF-IDF baselines) do not
        appear.
    """
    df = scores.copy()
    df["group"] = df["model"].str.replace(r"_s\d+_seed\d+$", "", regex=True)
    out = df.groupby(["group", "bucket"], sort=False)["weighted_f1"].agg(["count", "mean", "std"]).reset_index()
    out = out[out["count"] > 1].rename(columns={"count": "runs", "mean": "weighted_f1_mean", "std": "weighted_f1_sd"})
    return out


def interval(scores):
    """95% interval of a bootstrap distribution: its 2.5th and 97.5th percentiles (floats)."""
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate_file(path, cmi, edges, rng=None):
    """Score one prediction file overall and in each CMI bucket, and compare the buckets.

    Purpose:
        The core of the analysis for one model run.

    Args:
        path: Path of a ``*_dev.csv`` file with columns uid, label, pred.
        cmi: DataFrame of the DEV rows of the CMI table (columns uid, cmi, ...).
        edges: ``(low_edge, high_edge)`` from ``bucket_edges`` (train-based).
        rng: Optional NumPy ``Generator``. By default a generator seeded from the file name is used.

    Returns:
        ``(rows, pair_rows)``. ``rows`` has four dicts (``all``, ``low``, ``medium``, ``high``) with the tweet
        count, CMI range, accuracy, macro and weighted F1, per-class precision and recall and the 95% interval
        of weighted F1. ``pair_rows`` has three dicts (low - medium, medium - high, low - high) with the
        difference in weighted F1, its 95% interval and whether the interval contains zero.

    Raises:
        ValueError: Duplicate, missing or unexpected uids between the file and the CMI rows
            (``check_uid_alignment``).
        KeyError: The file lacks a needed column.

    Notes:
        The pairwise differences use the bootstrap scores of the two buckets, drawn independently because the
        buckets contain different tweets. An interval that contains 0 means no difference is detected; it does
        not prove there is none.
    """
    if rng is None:
        rng = file_rng(path.stem)
    pred = pd.read_csv(path)
    # Fail loudly before merging: an inner join would otherwise drop unmatched tweets silently.
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
        # Difference of the two buckets' resampled scores: its spread is the uncertainty of "a minus b".
        diff = boots[a] - boots[b]
        lo, hi = interval(diff)
        pair_rows.append({"model": model, "comparison": f"{a} - {b}", "difference": score[a] - score[b],
                          "ci_low": lo, "ci_high": hi, "interval_includes_zero": bool(lo <= 0 <= hi)})
    return rows, pair_rows


def main():
    """Run the whole analysis over every dev prediction file and write the three CSV files.

    Stages:
        1. Read the CMI table without test rows, check its uids, and compute the bucket edges from TRAIN.
        2. For each ``results/predictions/*_dev.csv`` (sorted by name) call ``evaluate_file``.
        3. Write cmi_bucket_scores.csv and cmi_bucket_pairwise.csv (floats rounded to 4 decimals) and
           cmi_bucket_group_summary.csv, and print the summaries and how many pairwise intervals exclude 0.

    Side effects:
        Overwrites the three CSV files in results/ and prints. Takes a few seconds.
    """
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
