"""Evaluation metrics shared by every model in this project.

The official SemEval-2020 Task 9 metric is the *weighted* F1 score
(Patwa et al., 2020, Section 3.1): F1 is computed for each class, then the
classes are averaged, weighted by how many true examples each class has.
Every result we report must use `weighted_f1` from this file so that all
models are compared the same way.
"""
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = ["negative", "neutral", "positive"]  # fixed order, used everywhere


def _check_labels(y_true, y_pred):
    """Stop with a clear error if the input is wrong (different lengths, empty, unknown label)
    instead of giving a wrong score."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")
    if len(y_true) == 0:
        raise ValueError("empty input")
    bad = (set(y_true) | set(y_pred)) - set(LABELS)
    if bad:
        raise ValueError(f"unknown labels {sorted(bad)}; expected {LABELS}")


def compute_metrics(y_true, y_pred):
    """Return a dict with the official score and extra diagnostics.

    `weighted_f1` is the number to report. The others help us understand
    where a model is strong or weak.
    """
    y_true, y_pred = list(y_true), list(y_pred)
    _check_labels(y_true, y_pred)

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
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
    """Confusion matrix as a list of rows, rows = true label, columns = predicted,
    both in the order of LABELS."""
    y_true, y_pred = list(y_true), list(y_pred)
    _check_labels(y_true, y_pred)
    return confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
