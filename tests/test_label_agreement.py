import pandas as pd
import pytest

from src.label_agreement import agreement


def frame(cats):
    return pd.DataFrame({"uid": range(len(cats)), "category": cats})


def test_identical_labels_agree_fully():
    a = frame(["negation", "sarcasm_irony", "slang_informal", "negation"])
    r, diffs = agreement(a, a.copy())
    assert r["raw_agreement"] == 1.0 and r["cohen_kappa"] == pytest.approx(1.0) and diffs.empty


def test_partial_agreement_and_kappa():
    a = frame(["x", "x", "y", "y"])
    b = frame(["x", "y", "y", "y"])
    r, diffs = agreement(a, b)
    assert r["raw_agreement"] == pytest.approx(0.75)
    assert r["cohen_kappa"] == pytest.approx(0.5)   # observed 0.75, expected 0.5 -> (0.75 - 0.5) / 0.5
    assert list(diffs["uid"]) == [1]


def test_order_of_rows_does_not_matter():
    a = frame(["x", "y", "x"])
    b = a.iloc[::-1].reset_index(drop=True)
    assert agreement(a, b)[0]["raw_agreement"] == 1.0


def test_second_file_must_have_the_same_uids():
    a = frame(["x", "y"])
    with pytest.raises(ValueError):
        agreement(a, frame(["x"]))
    with pytest.raises(ValueError):
        agreement(a, pd.DataFrame({"uid": [0, 1, 7], "category": ["x", "y", "x"]}))
