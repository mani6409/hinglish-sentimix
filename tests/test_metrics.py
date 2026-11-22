"""Tests for src/metrics.py: the shared evaluation metrics.

What is covered:
    Weighted and macro F1 on tiny inputs that were calculated by hand, accuracy on a perfect prediction, a class
    that is never predicted, the order of the confusion matrix, and input validation.

Why it matters:
    Every reported score in the project comes from ``compute_metrics``. If it were wrong, every table in the
    report would be wrong in the same way, and nobody would notice by comparing models with each other.

How to run:
    ``python -m pytest tests/test_metrics.py -q``. No data files are needed.
"""
import pytest

from src.metrics import compute_metrics, confusion


def test_perfect_prediction():
    """Predicting every label correctly gives weighted F1 = 1 and accuracy = 1.

    Input: four labels used as both truth and prediction. Prevents a regression in which a perfect model
    would not reach the maximum score (for example a wrong averaging weight).
    """
    y = ["positive", "negative", "neutral", "neutral"]
    m = compute_metrics(y, y)
    assert m["weighted_f1"] == 1.0 and m["accuracy"] == 1.0


def test_weighted_f1_hand_calculated():
    """Weighted and macro F1 match a calculation done by hand.

    Input: truth [pos, pos, neg, neu], predictions [pos, neg, neg, neu]. Per class: negative has precision
    1/2, recall 1, F1 2/3 (one true tweet); neutral has F1 1 (one true tweet); positive has precision 1,
    recall 1/2, F1 2/3 (two true tweets). Weighted F1 = (1*2/3 + 1*1 + 2*2/3) / 4 = 0.75; macro F1 =
    (2/3 + 1 + 2/3) / 3 = 7/9. The weights are the numbers of TRUE tweets per class (the official
    definition). Also checks the ``support`` field. Prevents confusing the weighted with the macro average.
    """
    # negative: P=1/2 R=1 F1=2/3 (support 1); neutral: F1=1 (support 1); positive: P=1 R=1/2 F1=2/3 (support 2)
    # weighted = (1*2/3 + 1*1 + 2*2/3) / 4 = 0.75 ; macro = (2/3 + 1 + 2/3) / 3 = 7/9
    y_true = ["positive", "positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "negative", "neutral"]
    m = compute_metrics(y_true, y_pred)
    assert m["weighted_f1"] == pytest.approx(0.75)
    assert m["macro_f1"] == pytest.approx(7 / 9)
    assert m["per_class"]["positive"]["support"] == 2


def test_class_never_predicted_gives_zero_not_error():
    """A class that the model never predicts scores 0 instead of raising an error.

    Input: the model always says negative. The positive class then has no predictions, and its F1 must be
    0.0 (``zero_division=0``). Prevents crashes (or NaN) on degenerate predictors such as "always neutral",
    which is the floor model of the project.
    """
    m = compute_metrics(["positive", "negative"], ["negative", "negative"])
    assert m["per_class"]["positive"]["f1"] == 0.0


def test_confusion_matrix_order():
    """Rows are true labels and columns predicted labels, both in the order negative, neutral, positive.

    Input: truth [negative, positive], predictions [negative, neutral]. Expected: the negative tweet is on the
    diagonal (row 0, column 0); the neutral row is empty; the positive tweet predicted as neutral sits in
    row 2, column 1. Prevents mixing up rows and columns or the class order.
    """
    cm = confusion(["negative", "positive"], ["negative", "neutral"])
    # rows/cols order: negative, neutral, positive
    assert cm == [[1, 0, 0], [0, 0, 0], [0, 1, 0]]


@pytest.mark.parametrize(
    "t,p",
    [(["positive"], ["positive", "negative"]), ([], []), (["good"], ["positive"])],
)
def test_bad_input_raises(t, p):
    """Invalid input raises ValueError.

    Three cases: different lengths (usually a misaligned join), empty input, and a label that is not one of
    the three classes (``good``). Prevents believable but wrong scores from misaligned or misspelled labels.
    """
    with pytest.raises(ValueError):
        compute_metrics(t, p)
