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
    rows = evaluate_file(p, cmi, (20.0, 40.0), np.random.default_rng(0))
    assert [r["bucket"] for r in rows] == ["all", "low", "medium", "high"]
    assert sum(r["tweets"] for r in rows[1:]) == 18
    assert all(r["weighted_f1"] == pytest.approx(1.0) for r in rows)
