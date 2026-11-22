"""Shared helpers for the single, frozen test-set evaluations (hashing, atomic writes, strict file parsing, joining by uid,
metrics).

These functions were written for the BiLSTM test evaluation (src/evaluate_bilstm_test.py, which produced
results/bilstm_final_test/ under docs/bilstm_final_evaluation_protocol.md; the script was later removed from the repository and is
in git history, e.g. at 0f17dc4). They are copied here unchanged so the transformer evaluation (src/evaluate_transformer_test.py)
uses the same, already-reviewed code without depending on a deleted module.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.data import parse_conll, to_frame
from src.metrics import LABELS, compute_metrics, confusion

SCALARS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1")
PRIMARY = ("macro_f1", "weighted_f1")
SECONDARY = ("accuracy", "macro_precision", "macro_recall", "weighted_precision", "weighted_recall")
CLASS_KEYS = ("precision", "recall", "f1")


class ProtocolError(RuntimeError):
    """A protocol rule or an integrity/validation check failed."""


class HashMismatchError(ProtocolError):
    """A frozen file is missing, has the wrong size, or its MD5 differs from the manifest."""


class ConfirmationRequired(ProtocolError):
    """The explicit --confirm-final-test flag was not given."""


# ============================================================== small helpers ===
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, data) -> None:
    """Write bytes or text to `path` through a temporary file and an atomic rename (flushed and fsynced)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(data, bytes):
        f = open(tmp, "wb")
    else:
        f = open(tmp, "w", encoding="utf-8", newline="")
    with f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=2) + "\n")


def csv_text(header, rows) -> str:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def verify_files(entries, *, check_md5: bool = True) -> dict:
    """entries: [{"label", "path", "md5", "size"}]. Missing file, wrong size or (if check_md5) wrong MD5 raises
    HashMismatchError listing every problem. Returns {label: md5} (empty hashes when check_md5 is False)."""
    problems, hashes = [], {}
    for e in entries:
        p = Path(e["path"])
        if not p.is_file():
            problems.append(f"{e['label']}: file missing ({p})")
            continue
        size = p.stat().st_size
        if size != e["size"]:
            problems.append(f"{e['label']}: size {size} != expected {e['size']}")
            continue
        if check_md5:
            got = md5_file(p)
            hashes[e["label"]] = got
            if got != e["md5"]:
                problems.append(f"{e['label']}: MD5 {got} != expected {e['md5']}")
    if problems:
        raise HashMismatchError("integrity check failed:\n  - " + "\n  - ".join(problems))
    return hashes


def read_unlabeled_test(path, expected_n: int) -> pd.DataFrame:
    """Parse the UNLABELED test file (no label file involved). Requires unique uids and the expected tweet count."""
    tweets = parse_conll(path)
    if not tweets:
        raise ProtocolError("the unlabeled test file contains no tweets")
    df = to_frame(tweets)
    if df["label"].notna().any():
        raise ProtocolError("the 'unlabeled' test file unexpectedly contains labels; refusing to continue")
    uids = df["uid"].tolist()
    if len(set(uids)) != len(uids):
        raise ProtocolError(f"the unlabeled test file has duplicate uids ({len(uids) - len(set(uids))} extra)")
    if len(uids) != expected_n:
        raise ProtocolError(f"the unlabeled test file has {len(uids)} tweets, expected {expected_n}")
    return df[["uid", "text"]].reset_index(drop=True)


def validate_predictions(expected_uids, pred_uids, preds, what: str = "predictions") -> None:
    """Unique uids, complete coverage of the expected uid set, one valid label per uid."""
    if len(pred_uids) != len(preds):
        raise ProtocolError(f"{what}: {len(pred_uids)} uids but {len(preds)} predictions")
    if len(set(pred_uids)) != len(pred_uids):
        raise ProtocolError(f"{what}: duplicate uids")
    exp, got = set(expected_uids), set(pred_uids)
    if exp != got:
        raise ProtocolError(f"{what}: coverage mismatch ({len(exp - got)} uids missing, {len(got - exp)} unexpected)")
    bad = set(preds) - set(LABELS)
    if bad:
        raise ProtocolError(f"{what}: invalid predicted labels {sorted(map(str, bad))}")


def read_predictions_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if header != ["uid", "pred"]:
            raise ProtocolError(f"{path}: unexpected header {header}")
        rows = [(int(a), b) for a, b in r]
    return [a for a, _ in rows], [b for _, b in rows]


def read_labels_strict(path) -> dict:
    """Read the separate gold-label file: header Uid,Sentiment; unique integer uids; labels in LABELS. First (and only) place the
    labels file is parsed. Error messages never include label content."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None or [h.strip() for h in header] != ["Uid", "Sentiment"]:
            raise ProtocolError("labels file: unexpected header (expected Uid,Sentiment)")
        for line_no, row in enumerate(r, start=2):
            if not row or not any(c.strip() for c in row):
                continue
            if len(row) != 2:
                raise ProtocolError(f"labels file line {line_no}: expected 2 columns")
            uid, lab = int(row[0]), row[1].strip()
            if uid in out:
                raise ProtocolError(f"labels file line {line_no}: duplicate uid {uid}")
            if lab not in LABELS:
                raise ProtocolError(f"labels file line {line_no}: invalid label")
            out[uid] = lab
    return out


def join_by_uid(pred_uids, labels: dict) -> list:
    """Gold label for each predicted uid, looked up BY UID. Raises on duplicate uids, uids without label, or unused labels."""
    if len(set(pred_uids)) != len(pred_uids):
        raise ProtocolError("join: duplicate uids among the predictions")
    missing = [u for u in pred_uids if u not in labels]
    extra = set(labels) - set(pred_uids)
    if missing or extra:
        raise ProtocolError(f"join: {len(missing)} predicted uids have no label, {len(extra)} labels have no prediction")
    return [labels[u] for u in pred_uids]


# ===================================================================== metrics ===
def full_metrics(y_true, y_pred) -> dict:
    """Predefined metrics, class order LABELS = [negative, neutral, positive], zero_division=0 (as src.metrics)."""
    y_true, y_pred = list(y_true), list(y_pred)
    out = {"accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average=avg, zero_division=0)
        out[f"{avg}_precision"], out[f"{avg}_recall"], out[f"{avg}_f1"] = float(p), float(r), float(f)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out["per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(LABELS)}
    ref = compute_metrics(y_true, y_pred)  # the project's own implementation must agree
    checks = [out["accuracy"] - ref["accuracy"], out["macro_f1"] - ref["macro_f1"], out["weighted_f1"] - ref["weighted_f1"]]
    checks += [out["per_class"][c][k] - ref["per_class"][c][k] for c in LABELS for k in CLASS_KEYS]
    if max(abs(v) for v in checks) > 1e-12:
        raise ProtocolError("full_metrics disagrees with src.metrics.compute_metrics")
    out["confusion"] = confusion(y_true, y_pred)  # rows = true, columns = predicted, LABELS order
    return out
