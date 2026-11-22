"""Evaluation metrics shared by every model in this project.

Purpose:
    One implementation of the scores we report (weighted F1, macro F1, accuracy, per-class precision,
    recall, F1 and support) and of the confusion matrix.

Why this file exists:
    The official SemEval-2020 Task 9 metric is the *weighted* F1 score (Patwa et al., 2020, Section 3.1): F1
    is computed for each class, then the classes are averaged, weighted by how many true examples each class
    has. If every script computed its own scores, small differences (class order, how empty classes are
    handled) could make results incomparable. Every result we report must use ``weighted_f1`` from this
    file so that all models are compared the same way.

Place in the pipeline:
    predictions (lists of label strings)  ->  compute_metrics / confusion  ->  tables, logs, reports

Inputs:
    ``y_true`` and ``y_pred``: equally long, non-empty sequences of label strings, each one of ``LABELS``
    (lists, numpy arrays and pandas Series all work).

Outputs:
    ``compute_metrics`` returns a dict; ``confusion`` returns a 3x3 list of lists.

Used by (importers found on main when this was written):
    src/baselines_tfidf.py, src/bilstm.py, src/cmi_evaluation.py, src/eda_part1.py, src/eda_part2.py,
    src/error_analysis.py, src/error_analysis_models.py, src/eval_protocol_utils.py,
    src/evaluate_transformer_test.py, src/indian_transformer.py, src/log_metrics.py,
    src/train_bilstm.py, src/train_transformer.py, src/transformer_comparison.py,
    src/transformer_utils.py, src/zero_shot_llm.py, tests/synthetic_conll.py,
    tests/test_bilstm_checkpoint.py, tests/test_bilstm_model.py, tests/test_cmi_evaluation.py,
    tests/test_evaluate_transformer_test.py, tests/test_final_comparison.py,
    tests/test_indian_transformer.py, tests/test_metrics.py, tests/test_transformer_experiments.py,
    tests/test_transformer_pipeline.py.

Depends on:
    scikit-learn (``precision_recall_fscore_support``, ``accuracy_score``, ``confusion_matrix``).

Design decisions:
    * ``labels=LABELS`` is always passed to scikit-learn, so all three classes are reported in a fixed order
      even when one class is absent from the predictions or from the truth.
    * ``zero_division=0``: a class that is never predicted (or never true) scores 0 instead of raising a
      warning, which keeps weighted and macro F1 defined for degenerate predictors such as "always neutral".
    * Inputs are validated (length, emptiness, label names) and wrong input raises, because silently scoring
      misaligned or misspelled labels would produce believable but wrong numbers.

Assumptions and limits:
    * Label strings are case-sensitive and must match ``LABELS`` exactly.
    * Predictions must already be label strings; integer class ids must be converted first.
"""
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = ["negative", "neutral", "positive"]  # fixed order, used everywhere


def _check_labels(y_true, y_pred):
    """Validate a pair of label sequences before they are scored.

    Purpose:
        Stop early, with a clear message, on the three input problems that would otherwise give plausible
        but wrong scores.

    Args:
        y_true: Gold labels as a list.
        y_pred: Predicted labels as a list.

    Raises:
        ValueError: The lengths differ (usually a misaligned join), the input is empty, or a label is not in
            ``LABELS`` (usually a typo, a different class order or a missing value).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")
    if len(y_true) == 0:
        raise ValueError("empty input")
    bad = (set(y_true) | set(y_pred)) - set(LABELS)
    if bad:
        raise ValueError(f"unknown labels {sorted(bad)}; expected {LABELS}")


def compute_metrics(y_true, y_pred):
    """Return the official score and the diagnostics we use for analysis.

    Purpose:
        The single place where scores are computed. ``weighted_f1`` is the number to report; the other
        fields explain where a model is strong or weak.

    How it works:
        Three calls to scikit-learn's ``precision_recall_fscore_support`` with the fixed label list:
        ``average="weighted"`` (per-class F1 weighted by the number of true tweets of each class, the
        official metric), ``average="macro"`` (plain mean over the three classes) and ``average=None``
        (the per-class values). Accuracy is the share of exact matches.

    Args:
        y_true: Gold labels (any iterable of label strings).
        y_pred: Predicted labels, in the same order and of the same length.

    Returns:
        A dict with:
            ``weighted_f1`` (float): OFFICIAL metric.
            ``macro_f1`` (float): unweighted mean of the three class F1 values.
            ``accuracy`` (float): fraction of correct predictions.
            ``per_class`` (dict): for each label in ``LABELS``, a dict with ``precision``, ``recall`` and
                ``f1`` (floats) and ``support`` (int, the number of true tweets of that class).

    Raises:
        ValueError: See ``_check_labels``.

    Edge cases:
        If a class is never predicted, its precision is 0 (``zero_division=0``). If predictions are all
        "neutral", weighted F1 is small but defined (0.205 on the dev set).

    Example:
        compute_metrics(["negative", "neutral"], ["negative", "positive"])["accuracy"]  ->  0.5
    """
    y_true, y_pred = list(y_true), list(y_pred)
    _check_labels(y_true, y_pred)

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    # average=None returns one value per class, in the order of LABELS.
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    return {
        "weighted_f1": float(weighted_f1),  # OFFICIAL metric
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lab in enumerate(LABELS)
        },
    }


def confusion(y_true, y_pred):
    """Confusion matrix as a list of rows (rows = true label, columns = predicted label).

    Purpose:
        Shows which classes are confused with which. Rows and columns are both in the order of ``LABELS``
        (negative, neutral, positive).

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels, same length and order.

    Returns:
        A 3x3 list of lists of ints. Entry ``[i][j]`` counts tweets whose true label is ``LABELS[i]`` and
        whose predicted label is ``LABELS[j]``; the diagonal holds the correct predictions.

    Raises:
        ValueError: See ``_check_labels``.

    Example:
        confusion(["neutral"], ["positive"])[1][2]  ->  1
    """
    y_true, y_pred = list(y_true), list(y_pred)
    _check_labels(y_true, y_pred)
    return confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
