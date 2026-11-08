import pytest

from src.metrics import compute_metrics, confusion


def test_perfect_prediction():
    y = ["positive", "negative", "neutral", "neutral"]
    m = compute_metrics(y, y)
    assert m["weighted_f1"] == 1.0 and m["accuracy"] == 1.0


def test_weighted_f1_hand_calculated():
    # negative: P=1/2 R=1 F1=2/3 (support 1); neutral: F1=1 (support 1); positive: P=1 R=1/2 F1=2/3 (support 2)
    # weighted = (1*2/3 + 1*1 + 2*2/3) / 4 = 0.75 ; macro = (2/3 + 1 + 2/3) / 3 = 7/9
    y_true = ["positive", "positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "negative", "neutral"]
    m = compute_metrics(y_true, y_pred)
    assert m["weighted_f1"] == pytest.approx(0.75)
    assert m["macro_f1"] == pytest.approx(7 / 9)
    assert m["per_class"]["positive"]["support"] == 2


def test_class_never_predicted_gives_zero_not_error():
    m = compute_metrics(["positive", "negative"], ["negative", "negative"])
    assert m["per_class"]["positive"]["f1"] == 0.0


def test_confusion_matrix_order():
    cm = confusion(["negative", "positive"], ["negative", "neutral"])
    # rows/cols order: negative, neutral, positive
    assert cm == [[1, 0, 0], [0, 0, 0], [0, 1, 0]]


@pytest.mark.parametrize(
    "t,p",
    [(["positive"], ["positive", "negative"]), ([], []), (["good"], ["positive"])],
)
def test_bad_input_raises(t, p):
    with pytest.raises(ValueError):
        compute_metrics(t, p)
