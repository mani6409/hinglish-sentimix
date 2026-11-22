"""Tests for src/cmi_evaluation.py: the CMI bucket analysis.

What is covered:
    Bucket edges and assignment, the fast weighted F1 and the bootstrap (size, replacement, interval), uid
    checks on the join, the shape of the output rows, per-file random generators, the reader that drops test
    rows, and the per-setting mean and standard deviation.

Why it matters:
    The conclusion "no clear evidence that CMI matters" rests on these functions. A wrong bucket edge, a
    wrong F1 or a bootstrap that did not resample correctly would change that conclusion without any error
    message.

Test data:
    Tiny synthetic tables written to a temporary folder (``tmp_path``); no project data files are needed.

How to run:
    ``python -m pytest tests/test_cmi_evaluation.py -q``.
"""
import numpy as np
import pandas as pd
import pytest

from src.cmi_evaluation import BUCKETS, assign_bucket, bucket_edges, evaluate_file


def test_edges_split_train_into_three_equal_groups():
    """The two edges cut 300 evenly spaced values into three groups of about 100.

    Input: the numbers 0 to 299 as "CMI of training tweets". Expected: after assigning buckets with the
    computed edges, each bucket holds 100 values, give or take one. Prevents edges that are computed with the
    wrong quantiles (for example the 25th and 75th) and produce unequal groups.
    """
    train = pd.Series(np.arange(300, dtype=float))
    lo, hi = bucket_edges(train)
    counts = assign_bucket(train, (lo, hi)).value_counts()
    assert all(abs(counts[b] - 100) <= 1 for b in BUCKETS)


def test_edges_use_only_the_values_given():
    """The edges are the 1/3 and 2/3 quantiles of exactly the values passed in.

    Input: [0, 30, 60]. By linear interpolation the 1/3 quantile is 20 and the 2/3 quantile is 40. This
    guards the rule that the edges are computed from the TRAIN values only: the function has no other
    data source it could mix in.
    """
    assert bucket_edges(pd.Series([0.0, 30.0, 60.0])) == pytest.approx((20.0, 40.0))


def test_assign_bucket_boundaries():
    """Values on and just above an edge go to the documented side.

    Input: CMI values 5, 20, 20.0001, 40 and 40.1 with edges (20, 40). Expected: low, low, medium, medium,
    high. So the right end of each interval belongs to the lower bucket (a value equal to the edge is not
    moved up). Prevents an off-by-one at the edges, which would move tweets between buckets.
    """
    b = assign_bucket(pd.Series([5.0, 20.0, 20.0001, 40.0, 40.1]), (20.0, 40.0))
    assert list(b) == ["low", "low", "medium", "medium", "high"]


def test_missing_cmi_raises(tmp_path):
    """A prediction file with a tweet that has no CMI value raises ValueError.

    Input: predictions for uids 1 and 2, CMI table with only uid 1. Expected: ValueError from the uid check,
    not a silently shorter merge. Prevents scoring on fewer tweets than the file contains.
    """
    p = tmp_path / "m_dev.csv"
    pd.DataFrame({"uid": [1, 2], "label": ["negative", "neutral"], "pred": ["negative", "neutral"]}).to_csv(p, index=False)
    cmi = pd.DataFrame({"uid": [1], "cmi": [10.0]})
    with pytest.raises(ValueError):
        evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))


def test_scores_per_bucket(tmp_path):
    """``evaluate_file`` returns one row for all tweets and one per bucket, with correct counts and scores.

    Input: 18 tweets with perfect predictions and CMI values spread from 0 to 50. Expected: rows in the order
    all, low, medium, high; the three bucket sizes add up to 18; weighted F1 is 1.0 in every row (a perfect
    model scores 1 everywhere). Prevents lost tweets or a wrong row order, which later code relies on.
    """
    labels = ["negative", "neutral", "positive"] * 6
    p = tmp_path / "m_dev.csv"
    pd.DataFrame({"uid": range(18), "label": labels, "pred": labels}).to_csv(p, index=False)
    cmi = pd.DataFrame({"uid": range(18), "cmi": np.linspace(0, 50, 18)})
    rows, _ = evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))
    assert [r["bucket"] for r in rows] == ["all", "low", "medium", "high"]
    assert sum(r["tweets"] for r in rows[1:]) == 18
    assert all(r["weighted_f1"] == pytest.approx(1.0) for r in rows)


# ---- bootstrap correctness, uid checks and pairwise rows ----------------------------------------------------
from src.cmi_evaluation import bootstrap_scores, interval, resample_indices, weighted_f1
from src.metrics import LABELS, compute_metrics


def _write_preds(path, uids, labels, preds):
    """Test helper: write a prediction file with the columns uid, label, pred."""
    pd.DataFrame({"uid": uids, "label": labels, "pred": preds}).to_csv(path, index=False)


def test_fast_weighted_f1_matches_the_project_metric():
    """The fast weighted F1 used in the bootstrap equals ``compute_metrics`` on random data.

    Input: 20 random label lists of 200 tweets; every fifth one never predicts the neutral class (the
    zero-division case). Expected: equal to ``compute_metrics(...)["weighted_f1"]`` in every case. Prevents the
    optimisation from drifting from the official metric, including for empty classes.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 3, 200)
        p = rng.integers(0, 3, 200)
        if _ % 5 == 0:
            p[p == 1] = 0  # a class that is never predicted (zero division case)
        names_y, names_p = [LABELS[i] for i in y], [LABELS[i] for i in p]
        assert weighted_f1(y, p) == pytest.approx(compute_metrics(names_y, names_p)["weighted_f1"])


def test_resample_has_same_size_and_draws_with_replacement():
    """One bootstrap resample has the same size as the sample and contains repeats.

    Input: 500 observations. Expected: 500 indices in the range 0..499, fewer than 500 distinct values
    (with replacement some indices repeat; the chance of no repeat in 500 draws is negligible). Prevents a
    bootstrap that samples without replacement, which would give intervals of width zero.
    """
    rng = np.random.default_rng(42)
    idx = resample_indices(500, rng)
    assert len(idx) == 500
    assert idx.min() >= 0 and idx.max() < 500
    assert len(np.unique(idx)) < 500  # with replacement: repeats occur (probability of none is negligible)


def test_bootstrap_of_perfect_predictions_is_one_and_interval_is_ordered():
    """Perfect predictions give a score of 1 in every resample, and the interval brackets the mean.

    Part 1: 120 perfect predictions, 50 resamples; all scores must be 1.0. Part 2: predictions shifted by one
    position, so EVERY prediction is wrong; all scores are 0, and the check ``lo <= mean <= hi`` then holds
    trivially. Part 2 is therefore a weak check of the interval ordering (a more informative test would use
    partly correct predictions); the interval itself is also exercised in the other tests.
    """
    labels = ["negative", "neutral", "positive"] * 40
    s = bootstrap_scores(labels, labels, np.random.default_rng(1), n=50)
    assert len(s) == 50 and np.allclose(s, 1.0)
    noisy = bootstrap_scores(labels, labels[1:] + labels[:1], np.random.default_rng(1), n=200)
    lo, hi = interval(noisy)
    assert lo <= noisy.mean() <= hi


def test_duplicate_uid_in_predictions_raises(tmp_path):
    """A repeated uid in a prediction file is rejected.

    Input: uids [1, 1, 2]. Expected: ValueError whose message contains "duplicate". Prevents a repeated tweet
    from being counted twice in a bucket.
    """
    p = tmp_path / "m_dev.csv"
    _write_preds(p, [1, 1, 2], ["negative"] * 3, ["negative"] * 3)
    cmi = pd.DataFrame({"uid": [1, 2], "cmi": [5.0, 30.0]})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))


def test_duplicate_uid_in_cmi_raises(tmp_path):
    """A repeated uid in the CMI table is rejected.

    Input: CMI uids [1, 2, 2]. Expected: ValueError mentioning "duplicate". Prevents a one-to-many join that
    would duplicate predictions.
    """
    p = tmp_path / "m_dev.csv"
    _write_preds(p, [1, 2], ["negative"] * 2, ["negative"] * 2)
    cmi = pd.DataFrame({"uid": [1, 2, 2], "cmi": [5.0, 30.0, 31.0]})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))


def test_unexpected_uid_in_predictions_raises(tmp_path):
    """A prediction uid that is not in the CMI table is rejected.

    Input: predictions for uids 1, 2 and 3 against a CMI table with 1 and 2. Expected: ValueError whose message
    contains "not in". Prevents predictions for tweets from another split being scored silently.
    """
    p = tmp_path / "m_dev.csv"
    _write_preds(p, [1, 2, 3], ["negative"] * 3, ["negative"] * 3)
    cmi = pd.DataFrame({"uid": [1, 2], "cmi": [5.0, 30.0]})
    with pytest.raises(ValueError, match="not in"):
        evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))


def test_pairwise_rows_have_three_comparisons(tmp_path):
    """The pairwise output has the three comparisons in a fixed order.

    Input: 36 tweets with perfect predictions. Expected comparisons: "low - medium", "medium - high",
    "low - high". The last assertion (interval around the difference) holds trivially here, because perfect
    predictions give a difference of exactly 0; the test mainly protects the names and the order of the rows.
    """
    labels = ["negative", "neutral", "positive"] * 12
    p = tmp_path / "m_dev.csv"
    _write_preds(p, range(36), labels, labels)
    cmi = pd.DataFrame({"uid": range(36), "cmi": np.linspace(0, 50, 36)})
    _, pairs = evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))
    assert [r["comparison"] for r in pairs] == ["low - medium", "medium - high", "low - high"]
    assert all(r["ci_low"] <= r["difference"] <= r["ci_high"] or r["difference"] == 0 for r in pairs)


# ---- per-file random generators, reader without test rows, per-setting summary -----------------------------
from src.cmi_evaluation import file_rng, read_cmi_without_test, summarize_by_group


def test_file_rng_depends_only_on_the_file_name():
    """The generator is a function of the file name only.

    Input: the same name twice and a different name. Expected: the same name gives identical numbers, a
    different name gives different numbers. Prevents seeding with something that changes between runs (such as
    Python's string ``hash``).
    """
    a = file_rng("tfidf_char_2-5_dev").integers(0, 10**9, 5)
    b = file_rng("tfidf_char_2-5_dev").integers(0, 10**9, 5)
    c = file_rng("another_model_dev").integers(0, 10**9, 5)
    assert list(a) == list(b) and list(a) != list(c)


def test_adding_a_file_does_not_change_other_files_results(tmp_path):
    """Evaluating another file in between does not change a file's intervals.

    Input: file b evaluated alone, then file a, then file b again. Expected: b's lower interval bounds are
    the same both times. Prevents a shared generator, with which adding a new model file (for example a
    transformer) would silently change the intervals of every file after it.
    """
    labels = ["negative", "neutral", "positive"] * 20
    cmi = pd.DataFrame({"uid": range(60), "cmi": np.linspace(0, 50, 60)})
    a, b = tmp_path / "a_dev.csv", tmp_path / "b_dev.csv"
    _write_preds(a, range(60), labels, labels[1:] + labels[:1])
    _write_preds(b, range(60), labels, labels[2:] + labels[:2])
    alone, _ = evaluate_file(b, cmi, (20.0, 40.0))
    evaluate_file(a, cmi, (20.0, 40.0))           # another file evaluated before b
    after, _ = evaluate_file(b, cmi, (20.0, 40.0))
    assert [r["weighted_f1_ci_low"] for r in alone] == [r["weighted_f1_ci_low"] for r in after]


def test_cmi_reader_drops_test_rows(tmp_path):
    """The reader keeps only train and dev rows.

    Input: a 7-row table with 3 test rows, read in chunks of 2 rows (so the filtering across chunk borders
    is exercised). Expected: 4 rows left, with the splits train and dev only. Prevents the test rows from
    reaching the analysis, which the project's "test set used once" rule forbids.
    """
    p = tmp_path / "cmi.csv"
    pd.DataFrame({"uid": range(7), "split": ["train", "dev", "test", "test", "dev", "train", "test"],
                  "label": ["neutral"] * 7, "cmi": [1.0] * 7}).to_csv(p, index=False)
    out = read_cmi_without_test(p, chunksize=2)
    assert set(out["split"]) == {"train", "dev"} and len(out) == 4


def test_group_summary_means_and_sds_over_seeds():
    """Runs that differ only in seed are grouped; their mean and sample sd are correct.

    Input: three runs with weighted F1 0.60, 0.62 and 0.64 in the "low" bucket, plus one single run of another
    model. Expected: one group ``bilstm_full_raw`` with 3 runs, mean 0.62 and sample sd 0.02 (deviations of
    plus and minus 0.02, divided by n - 1 = 2), and the single run does not appear. Prevents a wrong
    group name pattern or a population sd instead of the sample sd.
    """
    rows = []
    for seed, f in ((42, 0.60), (43, 0.62), (44, 0.64)):
        rows.append({"model": f"bilstm_full_raw_s{seed}_seed{seed}", "bucket": "low", "weighted_f1": f})
    rows.append({"model": "tfidf_char_2-5", "bucket": "low", "weighted_f1": 0.61})
    g = summarize_by_group(pd.DataFrame(rows))
    assert list(g["group"]) == ["bilstm_full_raw"]
    assert g["weighted_f1_mean"].iloc[0] == pytest.approx(0.62)
    assert g["weighted_f1_sd"].iloc[0] == pytest.approx(0.02)
    assert g["runs"].iloc[0] == 3
