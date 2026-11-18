import numpy as np
import pandas as pd
import pytest

from src.cmi_evaluation import BUCKETS, assign_bucket, bucket_edges, evaluate_file


def test_edges_split_train_into_three_equal_groups():
    train = pd.Series(np.arange(300, dtype=float))
    lo, hi = bucket_edges(train)
    counts = assign_bucket(train, (lo, hi)).value_counts()
    assert all(abs(counts[b] - 100) <= 1 for b in BUCKETS)


def test_edges_use_only_the_values_given():
    assert bucket_edges(pd.Series([0.0, 30.0, 60.0])) == pytest.approx((20.0, 40.0))


def test_assign_bucket_boundaries():
    b = assign_bucket(pd.Series([5.0, 20.0, 20.0001, 40.0, 40.1]), (20.0, 40.0))
    assert list(b) == ["low", "low", "medium", "medium", "high"]


def test_missing_cmi_raises(tmp_path):
    p = tmp_path / "m_dev.csv"
    pd.DataFrame({"uid": [1, 2], "label": ["negative", "neutral"], "pred": ["negative", "neutral"]}).to_csv(p, index=False)
    cmi = pd.DataFrame({"uid": [1], "cmi": [10.0]})
    with pytest.raises(ValueError):
        evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))


def test_scores_per_bucket(tmp_path):
    labels = ["negative", "neutral", "positive"] * 6
    p = tmp_path / "m_dev.csv"
    pd.DataFrame({"uid": range(18), "label": labels, "pred": labels}).to_csv(p, index=False)
    cmi = pd.DataFrame({"uid": range(18), "cmi": np.linspace(0, 50, 18)})
    rows, _ = evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))
    assert [r["bucket"] for r in rows] == ["all", "low", "medium", "high"]
    assert sum(r["tweets"] for r in rows[1:]) == 18
    assert all(r["weighted_f1"] == pytest.approx(1.0) for r in rows)


# ---- review additions: bootstrap correctness, uid checks, pairwise rows -------------------------------------
from src.cmi_evaluation import bootstrap_scores, interval, resample_indices, weighted_f1
from src.metrics import LABELS, compute_metrics


def _write_preds(path, uids, labels, preds):
    pd.DataFrame({"uid": uids, "label": labels, "pred": preds}).to_csv(path, index=False)


def test_fast_weighted_f1_matches_the_project_metric():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 3, 200)
        p = rng.integers(0, 3, 200)
        if _ % 5 == 0:
            p[p == 1] = 0  # a class that is never predicted (zero division case)
        names_y, names_p = [LABELS[i] for i in y], [LABELS[i] for i in p]
        assert weighted_f1(y, p) == pytest.approx(compute_metrics(names_y, names_p)["weighted_f1"])


def test_resample_has_same_size_and_draws_with_replacement():
    rng = np.random.default_rng(42)
    idx = resample_indices(500, rng)
    assert len(idx) == 500
    assert idx.min() >= 0 and idx.max() < 500
    assert len(np.unique(idx)) < 500  # with replacement: repeats occur (probability of none is negligible)


def test_bootstrap_of_perfect_predictions_is_one_and_interval_is_ordered():
    labels = ["negative", "neutral", "positive"] * 40
    s = bootstrap_scores(labels, labels, np.random.default_rng(1), n=50)
    assert len(s) == 50 and np.allclose(s, 1.0)
    noisy = bootstrap_scores(labels, labels[1:] + labels[:1], np.random.default_rng(1), n=200)
    lo, hi = interval(noisy)
    assert lo <= noisy.mean() <= hi


def test_duplicate_uid_in_predictions_raises(tmp_path):
    p = tmp_path / "m_dev.csv"
    _write_preds(p, [1, 1, 2], ["negative"] * 3, ["negative"] * 3)
    cmi = pd.DataFrame({"uid": [1, 2], "cmi": [5.0, 30.0]})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))


def test_duplicate_uid_in_cmi_raises(tmp_path):
    p = tmp_path / "m_dev.csv"
    _write_preds(p, [1, 2], ["negative"] * 2, ["negative"] * 2)
    cmi = pd.DataFrame({"uid": [1, 2, 2], "cmi": [5.0, 30.0, 31.0]})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))


def test_unexpected_uid_in_predictions_raises(tmp_path):
    p = tmp_path / "m_dev.csv"
    _write_preds(p, [1, 2, 3], ["negative"] * 3, ["negative"] * 3)
    cmi = pd.DataFrame({"uid": [1, 2], "cmi": [5.0, 30.0]})
    with pytest.raises(ValueError, match="not in"):
        evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))


def test_pairwise_rows_have_three_comparisons(tmp_path):
    labels = ["negative", "neutral", "positive"] * 12
    p = tmp_path / "m_dev.csv"
    _write_preds(p, range(36), labels, labels)
    cmi = pd.DataFrame({"uid": range(36), "cmi": np.linspace(0, 50, 36)})
    _, pairs = evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))
    assert [r["comparison"] for r in pairs] == ["low - medium", "medium - high", "low - high"]
    assert all(r["ci_low"] <= r["difference"] <= r["ci_high"] or r["difference"] == 0 for r in pairs)
