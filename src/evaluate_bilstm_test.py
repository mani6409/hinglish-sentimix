"""Final, single evaluation of the nine frozen BiLSTM checkpoints on the official SentiMix Hinglish TEST set.

READ docs/bilstm_final_evaluation_protocol.md FIRST. This script is meant to be run ONCE.

    python -m src.evaluate_bilstm_test --validate-only                 # checks only: no test labels, no inference, nothing written
    python -m src.evaluate_bilstm_test --confirm-final-test            # the real, one-time evaluation

What it guarantees
- The nine frozen checkpoints (configs/bilstm_frozen_checkpoints.json) are hash-verified (MD5) before any inference, together
  with the unlabeled test file; the labels file is only size-checked before inference.
- Two stages. Stage A: the unlabeled test file is parsed (no labels), all nine checkpoints predict, and the nine prediction files are
  written to disk and read back and validated. Only then does Stage B start: the labels file is hash-verified and read for the FIRST
  time, labels are joined to predictions by UID (never by row order), and metrics are computed from the saved prediction files.
- After evaluation all hashed inputs are re-hashed and must be unchanged; an integrity manifest records everything.
- Outputs are written to a staging folder and published with one atomic rename. An existing output folder is never overwritten
  (there is no --overwrite option): pick a new --out-dir deliberately, or remove the old one yourself.
- If anything fails BEFORE the labels file is touched, the staging folder is discarded and a corrected rerun is possible. If it fails
  AFTER the labels file was touched, the saved predictions are kept in "<out-dir>.failed_<time>", which blocks a silent rerun.
- Nothing is selected, tuned or changed from test results; this script only measures the frozen models.

--validate-only never opens the labels file, never parses or hashes the test files (it only checks that they exist and have the
expected size), never runs inference, and writes nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import warnings
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.data import parse_conll, to_frame
from src.metrics import LABELS, compute_metrics, confusion
from src.preprocess import PRESETS

ROOT = Path(__file__).resolve().parents[1]
MODES = ("raw", "clean", "normalized")
SEEDS = (42, 43, 44)
RUNS = tuple(product(MODES, SEEDS))  # the nine frozen runs
DEFAULT_MANIFEST = "configs/bilstm_frozen_checkpoints.json"
DEFAULT_OUT_DIR = "results/bilstm_final_test"
SCALARS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1")
PRIMARY = ("macro_f1", "weighted_f1")
SECONDARY = ("accuracy", "macro_precision", "macro_recall", "weighted_precision", "weighted_recall")
CLASS_KEYS = ("precision", "recall", "f1")
PROTOCOL_DOC = "docs/bilstm_final_evaluation_protocol.md"


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


# ================================================================ the manifest ===
def validate_manifest(m: dict) -> dict:
    """Structure checks of the frozen manifest: exactly the nine (mode, seed) runs, no duplicates, well-formed hashes."""
    for key in ("labels", "expected", "checkpoints", "test_files"):
        if key not in m:
            raise ProtocolError(f"manifest is missing '{key}'")
    if list(m["labels"]) != LABELS:
        raise ProtocolError(f"manifest labels {m['labels']} differ from the project order {LABELS}")
    exp_keys = {"n_test", "max_words", "max_chars", "train_size", "dev_size", "format_version", "selected_on", "variants_sha256"}
    if not exp_keys <= set(m["expected"]):
        raise ProtocolError(f"manifest 'expected' lacks {sorted(exp_keys - set(m['expected']))}")
    pairs = [(c.get("mode"), c.get("seed")) for c in m["checkpoints"]]
    dup = sorted({p for p in pairs if pairs.count(p) > 1}, key=str)
    if dup:
        raise ProtocolError(f"duplicate checkpoint entries for {dup}")
    if set(pairs) != set(RUNS) or len(pairs) != len(RUNS):
        raise ProtocolError(f"the manifest must list exactly the nine runs {list(RUNS)}; found {pairs}")
    for c in m["checkpoints"]:
        if not (isinstance(c.get("path"), str) and isinstance(c.get("md5"), str) and len(c["md5"]) == 32
                and isinstance(c.get("size"), int) and c["size"] > 0):
            raise ProtocolError(f"malformed checkpoint entry {c.get('mode')}/{c.get('seed')}")
    for key in ("unlabeled", "labels"):
        t = m["test_files"].get(key, {})
        if not (isinstance(t.get("file"), str) and isinstance(t.get("md5"), str) and len(t["md5"]) == 32 and isinstance(t.get("size"), int)):
            raise ProtocolError(f"malformed test_files.{key} entry")
    return m


def load_manifest(path) -> dict:
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


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


# ======================================================= checkpoints and checks ===
def load_frozen_checkpoints(manifest: dict, repo_root: Path) -> dict:
    """Load the nine checkpoints (weights only) and verify that each one IS the run the manifest says it is."""
    import dataclasses

    from src.bilstm_checkpoint import load_checkpoint

    exp = manifest["expected"]
    loaded, first_cfg, problems = {}, None, []
    for c in manifest["checkpoints"]:
        mode, seed = c["mode"], c["seed"]
        tag = f"{mode}/seed{seed}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the dictionary fingerprint is checked explicitly below
            ck = load_checkpoint(repo_root / c["path"])
        meta, enc = ck.meta, ck.encoder
        if ck.config.mode != mode or meta.get("mode") != mode:
            problems.append(f"{tag}: checkpoint was trained in mode '{ck.config.mode}'/'{meta.get('mode')}'")
        if meta.get("seed") != seed:
            problems.append(f"{tag}: checkpoint seed is {meta.get('seed')}")
        if meta.get("preprocess_options") != PRESETS[mode]:
            problems.append(f"{tag}: preprocessing options differ from PRESETS['{mode}']")
        if (enc.max_words, enc.max_chars) != (exp["max_words"], exp["max_chars"]):
            problems.append(f"{tag}: length limits {(enc.max_words, enc.max_chars)} != expected")
        if tuple(enc.special_words) != ("USER", "URL"):
            problems.append(f"{tag}: placeholder words are {enc.special_words}")
        if (meta.get("train_size"), meta.get("dev_size")) != (exp["train_size"], exp["dev_size"]):
            problems.append(f"{tag}: trained on {meta.get('train_size')}/{meta.get('dev_size')} tweets")
        if meta.get("selected_on") != exp["selected_on"]:
            problems.append(f"{tag}: selected on '{meta.get('selected_on')}'")
        if meta.get("variants_sha256") != exp["variants_sha256"]:
            problems.append(f"{tag}: trained with a different Hinglish dictionary than the frozen one")
        rest = {k: v for k, v in dataclasses.asdict(ck.config).items() if k != "mode"}
        if first_cfg is None:
            first_cfg = rest
        elif rest != first_cfg:
            problems.append(f"{tag}: hyperparameters differ from the other checkpoints")
        loaded[(mode, seed)] = ck
    from src.bilstm_checkpoint import variants_fingerprint

    if variants_fingerprint() != exp["variants_sha256"]:
        problems.append("configs/hinglish_variants.json has changed since the checkpoints were frozen")
    if problems:
        raise ProtocolError("checkpoint verification failed:\n  - " + "\n  - ".join(problems))
    return loaded


# ================================================================ data handling ===
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


def aggregate_runs(per_run: dict) -> dict:
    """Mean and SAMPLE standard deviation (ddof = 1) across seeds 42, 43, 44 for every mode, for all scalar and class-wise metrics."""
    missing = [r for r in RUNS if r not in per_run]
    if missing:
        raise ProtocolError(f"aggregation needs all nine runs; missing {missing}")
    out = {}
    for mode in MODES:
        runs = [per_run[(mode, s)] for s in SEEDS]

        def stat(vals):
            v = np.asarray(vals, float)
            return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "values": [float(x) for x in v]}

        out[mode] = {k: stat([r[k] for r in runs]) for k in SCALARS}
        out[mode]["per_class"] = {c: {k: stat([r["per_class"][c][k] for r in runs]) for k in CLASS_KEYS} for c in LABELS}
        out[mode]["confusion_summed"] = np.sum([np.asarray(r["confusion"]) for r in runs], axis=0).tolist()
    return out


# ===================================================================== report ===
def _ms(stat, d=4):
    return f"{stat['mean']:.{d}f} ± {stat['std']:.{d}f}"


def render_report(ctx: dict) -> str:
    agg, per_run, n = ctx["aggregate"], ctx["per_run"], ctx["n_test"]
    L = []
    add = L.append
    add("# BiLSTM final test-set evaluation (single, frozen-model run)")
    add("")
    add(f"*Evaluates the nine frozen checkpoints (raw / clean / normalized x seeds 42, 43, 44) on the official SentiMix Hinglish test set ({n:,} tweets). "
        f"Protocol: `{PROTOCOL_DOC}`. Produced by `python -m src.evaluate_bilstm_test --confirm-final-test` on {ctx['finished_utc']}.*")
    add("")
    add("**Nothing was selected, tuned or changed based on these results.** The checkpoints, preprocessing and metrics were frozen before the test labels were read. "
        "This report states measurements only; no preprocessing mode is declared the winner, and any difference smaller than the seed-to-seed spread must not be interpreted as a real difference.")
    add("")
    add("## 1. Primary results: mean ± sample standard deviation over seeds 42, 43, 44")
    add("")
    add("| Mode | Macro F1 | Weighted F1 |")
    add("|---|---|---|")
    for m in MODES:
        add(f"| {m} | **{_ms(agg[m]['macro_f1'])}** | **{_ms(agg[m]['weighted_f1'])}** |")
    add("")
    add("Per-seed values (seed 42 / 43 / 44):")
    add("")
    add("| Mode | Macro F1 | Weighted F1 |")
    add("|---|---|---|")
    for m in MODES:
        add(f"| {m} | " + " / ".join(f"{v:.4f}" for v in agg[m]["macro_f1"]["values"]) + " | " + " / ".join(f"{v:.4f}" for v in agg[m]["weighted_f1"]["values"]) + " |")
    add("")
    add("Standard deviations are sample standard deviations (n = 3, ddof = 1) and are themselves rough with only three seeds.")
    add("")
    add("## 2. Secondary metrics (mean ± sample std over seeds)")
    add("")
    add("| Mode | Accuracy | Macro precision | Macro recall | Weighted precision | Weighted recall |")
    add("|---|---|---|---|---|---|")
    for m in MODES:
        add(f"| {m} | " + " | ".join(_ms(agg[m][k]) for k in SECONDARY) + " |")
    add("")
    add("Weighted recall equals accuracy by definition.")
    add("")
    add("## 3. Class-wise precision, recall and F1 (mean ± sample std over seeds)")
    add("")
    add("| Mode | Class | Support | Precision | Recall | F1 |")
    add("|---|---|---|---|---|---|")
    for m in MODES:
        for c in LABELS:
            sup = per_run[(m, SEEDS[0])]["per_class"][c]["support"]
            add(f"| {m} | {c} | {sup:,} | " + " | ".join(_ms(agg[m]["per_class"][c][k]) for k in CLASS_KEYS) + " |")
    add("")
    add("## 4. Every run")
    add("")
    add("| Run | Accuracy | Macro P | Macro R | Macro F1 | Weighted P | Weighted R | Weighted F1 |")
    add("|---|---|---|---|---|---|---|---|")
    for (m, s) in RUNS:
        r = per_run[(m, s)]
        add(f"| {m} seed {s} | " + " | ".join(f"{r[k]:.4f}" for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1")) + " |")
    add("")
    add("## 5. Confusion matrices")
    add("")
    add("Rows = true class, columns = predicted class, order negative / neutral / positive. Each matrix is the **sum of the three seeds' matrices** "
        f"({3 * n:,} predictions per mode); percentages are row-normalised. Per-run matrices are in `confusion_matrices.json`.")
    add("")
    for m in MODES:
        cm = np.asarray(agg[m]["confusion_summed"])
        add(f"**{m}**")
        add("")
        add("| | pred negative | pred neutral | pred positive | Row total |")
        add("|---|---|---|---|---|")
        for i, c in enumerate(LABELS):
            tot = cm[i].sum()
            add(f"| true {c} | " + " | ".join(f"{cm[i, j]:,} ({100 * cm[i, j] / tot:.1f}%)" if tot else "0" for j in range(3)) + f" | {tot:,} |")
        add("")
    if ctx.get("figures"):
        add("Figures: " + ", ".join(f"`figures/{f}`" for f in ctx["figures"]) + ".")
        add("")
    elif ctx.get("figures_error"):
        add(f"*Figures were not generated: {ctx['figures_error']}*")
        add("")
    add("## 6. Integrity and procedure")
    add("")
    add("- All hashes below are MD5. The nine checkpoints and the unlabeled test file were verified **before** inference; the labels file was verified and read for the "
        "**first time only after** all nine prediction files had been written and validated; every hashed input was verified **again after** evaluation and was unchanged.")
    add("- Predictions were joined to the gold labels by **UID**, never by row order. Metrics were computed from the saved prediction files.")
    add("- Full hashes, timestamps of each stage, the environment and output hashes are in `integrity_manifest.json`.")
    add("")
    add("| Item | MD5 (before = after) |")
    add("|---|---|")
    for label, h in ctx["input_hashes"].items():
        add(f"| {label} | `{h}` |")
    add("")
    add("## 7. Limitations")
    add("")
    add("- Three seeds per mode: the standard deviation is rough, and differences between modes cannot be judged from the means alone.")
    add("- The test set is one fixed sample of 3,000 tweets; scores carry sampling error of about ±0.01 on F1, which this report does not estimate.")
    add("- The team's EDA computed descriptive statistics on the test split earlier (label counts, overlap with train and dev, CMI). No model decision used them (see the protocol).")
    add("- Dev scores used for early stopping are optimistic; test scores are the unbiased estimate for these frozen models.")
    add("")
    return "\n".join(L) + "\n"


def make_figures(fig_dir: Path, aggregate: dict) -> list:
    """Row-normalised confusion-matrix heatmaps (single-hue light-to-dark ramp). Returns the file names written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"], "figure.facecolor": surface, "savefig.facecolor": surface, "axes.facecolor": surface, "text.color": ink})
    fig_dir.mkdir(parents=True, exist_ok=True)

    def draw(ax, conf, title, fs=11):
        conf = np.asarray(conf)
        pct = 100 * conf / conf.sum(1, keepdims=True)
        im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, aspect="equal", interpolation="nearest")
        for i in range(3):
            for j in range(3):
                dark = pct[i, j] > 48
                ax.text(j, i - 0.12, f"{pct[i, j]:.1f}%", ha="center", va="center", fontsize=fs, fontweight="bold" if i == j else "normal", color="#ffffff" if dark else ink)
                ax.text(j, i + 0.22, f"n = {conf[i, j]:,}", ha="center", va="center", fontsize=fs - 3.5, color="#e6eefb" if dark else ink2)
        ax.set_xticks(range(3), [c.capitalize() for c in LABELS])
        ax.set_yticks(range(3), [c.capitalize() for c in LABELS])
        ax.set_xlabel("Predicted label", color=ink2)
        ax.set_ylabel("True label", color=ink2)
        ax.tick_params(length=0, colors=ink2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for k in (0.5, 1.5):
            ax.axhline(k, color=surface, linewidth=3, solid_capstyle="butt")
            ax.axvline(k, color=surface, linewidth=3, solid_capstyle="butt")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
        return im

    written = []
    sub = "Row-normalised; counts summed over seeds 42, 43, 44 (3 x the test size per mode)"
    for m in MODES:
        fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=300)
        fig.subplots_adjust(left=0.2, right=0.96, top=0.80, bottom=0.14)
        draw(ax, aggregate[m]["confusion_summed"], m.capitalize())
        fig.text(0.04, 0.95, "BiLSTM confusion matrix on the test set", fontsize=12.5, fontweight="bold")
        fig.text(0.04, 0.905, sub, fontsize=8.2, color=ink2)
        name = f"bilstm_final_test_confusion_{m}.png"
        fig.savefig(fig_dir / name)
        plt.close(fig)
        written.append(name)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.0), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.78, bottom=0.14, wspace=0.30)
    im = None
    for ax, m in zip(axes, MODES):
        im = draw(ax, aggregate[m]["confusion_summed"], m.capitalize())
    cax = fig.add_axes([0.915, 0.25, 0.011, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("% of the true class", color=ink2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.text(0.07, 0.94, "BiLSTM confusion matrices on the test set, by preprocessing mode", fontsize=13, fontweight="bold")
    fig.text(0.07, 0.895, sub + "; one shared color scale", fontsize=9, color=ink2)
    name = "bilstm_final_test_confusion_all_modes.png"
    fig.savefig(fig_dir / name)
    plt.close(fig)
    written.append(name)
    return written


# ================================================================ orchestration ===
def environment_info(checkpoints: dict) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        info["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    info["torch_version_recorded_in_checkpoints"] = sorted({str(ck.meta.get("torch_version")) for ck in checkpoints.values()})
    return info


def evaluate(manifest_path, *, repo_root=ROOT, data_root=None, out_dir=None, validate_only: bool = False, confirm: bool = False,
             make_figs: bool = True, log=print) -> dict:
    """Run the protocol. validate_only=True performs the pre-flight checks only. Otherwise `confirm=True` is mandatory."""
    if not validate_only and not confirm:
        raise ConfirmationRequired("refusing to run: the final test evaluation needs the explicit --confirm-final-test flag "
                                   "(use --validate-only for the pre-flight checks).")
    repo_root = Path(repo_root)
    data_root = Path(data_root) if data_root else repo_root / "data" / "raw"
    out_dir = Path(out_dir) if out_dir else repo_root / DEFAULT_OUT_DIR
    staging = out_dir.with_name(out_dir.name + ".staging")
    # never overwrite: any earlier output, staging folder or failed attempt blocks the run
    blockers = [p for p in (out_dir, staging, *out_dir.parent.glob(out_dir.name + ".failed_*")) if p.exists()] if out_dir.parent.exists() else []
    if blockers:
        raise FileExistsError("an earlier output or attempt exists and is never overwritten; remove it deliberately or choose another --out-dir: "
                              + ", ".join(str(b) for b in blockers))
    manifest = load_manifest(manifest_path)
    exp, tf = manifest["expected"], manifest["test_files"]
    unl_path, lab_path = data_root / tf["unlabeled"]["file"], data_root / tf["labels"]["file"]
    events = []

    def ev(name, **kw):
        events.append({"event": name, "utc": utc_now(), **kw})
        log(f"[{events[-1]['utc']}] {name}" + (f" {kw}" if kw else ""))

    ev("start", mode="validate-only" if validate_only else "final-evaluation", manifest=str(manifest_path))

    # ---- pre-flight (before any inference): hashes of the nine checkpoints and the UNLABELED test file
    ck_entries = [{"label": f"checkpoint {c['mode']} seed {c['seed']}", "path": repo_root / c["path"], "md5": c["md5"], "size": c["size"]} for c in manifest["checkpoints"]]
    unl_entry = {"label": "test file (unlabeled)", "path": unl_path, "md5": tf["unlabeled"]["md5"], "size": tf["unlabeled"]["size"]}
    lab_entry = {"label": "test file (labels)", "path": lab_path, "md5": tf["labels"]["md5"], "size": tf["labels"]["size"]}
    if validate_only:  # existence and size ONLY: test files are neither opened nor hashed
        verify_files(ck_entries, check_md5=True)
        verify_files([unl_entry, lab_entry], check_md5=False)
        hashes_before = {}
    else:
        hashes_before = verify_files(ck_entries + [unl_entry], check_md5=True)
        verify_files([lab_entry], check_md5=False)  # size only: the labels file is not read before predictions are saved
    ev("hashes_verified_before_inference", checkpoints=9, unlabeled_test_file=not validate_only, labels_file_size_only=True)
    checkpoints = load_frozen_checkpoints(manifest, repo_root)
    ev("checkpoints_loaded_and_mapped", runs=len(checkpoints))
    env = environment_info(checkpoints)
    if validate_only:
        ev("validation_only_complete")
        return {"validated": True, "events": events, "environment": env}

    # ---- the unlabeled file is parsed and validated BEFORE anything is created on disk (still no labels involved)
    test_df = read_unlabeled_test(unl_path, exp["n_test"])
    uids, texts = test_df["uid"].tolist(), test_df["text"].tolist()
    ev("unlabeled_test_parsed", tweets=len(uids))

    # ---- everything below writes only into the staging folder
    staging.mkdir(parents=True)
    stage_b_started = False
    try:
        # ===== Stage A: predictions (NO labels) =====
        pred_hashes = {}
        for mode, seed in RUNS:
            preds = checkpoints[(mode, seed)].predict_texts(texts, preprocess=True)
            validate_predictions(uids, uids, preds, f"{mode}/seed{seed} predictions")
            fp = staging / "predictions" / f"bilstm_final_test_{mode}_seed{seed}_predictions.csv"
            atomic_write(fp, csv_text(["uid", "pred"], zip(uids, preds)))
            back_uids, back_preds = read_predictions_csv(fp)  # validate what is actually on disk
            validate_predictions(uids, back_uids, back_preds, f"saved {fp.name}")
            pred_hashes[fp.name] = md5_file(fp)
        atomic_write_json(staging / "predictions_manifest.json", {
            "written_before_labels_were_read": True, "utc": utc_now(), "n_test": len(uids), "prediction_files_md5": pred_hashes,
            "inputs_md5": {k: v for k, v in hashes_before.items()}})
        ev("predictions_saved_and_validated", files=len(pred_hashes))

        # ===== Stage B: labels (first and only read of the labels file) =====
        stage_b_started = True
        labels_md5 = md5_file(lab_path)
        if labels_md5 != lab_entry["md5"]:
            raise HashMismatchError(f"integrity check failed: labels file MD5 {labels_md5} != expected {lab_entry['md5']}")
        labels = read_labels_strict(lab_path)
        ev("labels_read_after_predictions_were_saved", labels=len(labels))
        if set(labels) != set(uids):
            raise ProtocolError(f"label coverage mismatch: {len(set(uids) - set(labels))} test uids without label, {len(set(labels) - set(uids))} unexpected labels")
        per_run = {}
        for mode, seed in RUNS:
            fp = staging / "predictions" / f"bilstm_final_test_{mode}_seed{seed}_predictions.csv"
            p_uids, p_preds = read_predictions_csv(fp)  # metrics come from the SAVED predictions
            y_true = join_by_uid(p_uids, labels)
            per_run[(mode, seed)] = full_metrics(y_true, p_preds)
        aggregate = aggregate_runs(per_run)
        ev("metrics_computed")

        # ---- post-evaluation integrity: every hashed input must be unchanged
        hashes_after = verify_files(ck_entries + [unl_entry, lab_entry], check_md5=True)
        hashes_before = {**hashes_before, lab_entry["label"]: labels_md5}
        if hashes_after != hashes_before:
            raise HashMismatchError("an input file changed during the evaluation")
        ev("hashes_verified_after_evaluation", files=len(hashes_after))

        # ---- outputs
        figures, fig_err = [], None
        if make_figs:
            try:
                figures = make_figures(staging / "figures", aggregate)
            except Exception as e:  # noqa: BLE001  (figures are cosmetic; the numbers are in the JSON and the report)
                fig_err = repr(e)[:200]
        per_run_json = {m: {str(s): {k: v for k, v in per_run[(m, s)].items() if k != "confusion"} for s in SEEDS} for m in MODES}
        atomic_write_json(staging / "metrics_per_run.json", per_run_json)
        atomic_write_json(staging / "metrics_summary.json", {m: {k: v for k, v in aggregate[m].items() if k != "confusion_summed"} for m in MODES}
                          | {"_description": "mean and sample standard deviation (ddof=1) over seeds 42, 43, 44", "_primary_metrics": list(PRIMARY)})
        atomic_write_json(staging / "confusion_matrices.json", {
            "labels_order": LABELS, "rows": "true", "columns": "predicted",
            "per_run": {m: {str(s): per_run[(m, s)]["confusion"] for s in SEEDS} for m in MODES},
            "summed_over_seeds": {m: aggregate[m]["confusion_summed"] for m in MODES}})
        report = render_report({"aggregate": aggregate, "per_run": per_run, "n_test": len(uids), "finished_utc": utc_now(), "figures": figures,
                                "figures_error": fig_err, "input_hashes": {k: v for k, v in hashes_after.items()}})
        atomic_write(staging / "bilstm_final_test_report.md", report)
        ev("outputs_written", figures=len(figures))
        out_hashes = {str(p.relative_to(staging)).replace("\\", "/"): md5_file(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        atomic_write_json(staging / "integrity_manifest.json", {
            "protocol": PROTOCOL_DOC, "frozen_manifest": str(manifest_path), "frozen_manifest_md5": md5_file(manifest_path),
            "environment": env, "events": events,
            "inputs": {"md5_before_and_after_identical": True, "md5": hashes_after,
                       "labels_file_first_read_after_predictions_saved": True},
            "outputs_md5": out_hashes})
        os.replace(staging, out_dir)  # atomic publish of the whole result folder
        ev("published", out_dir=str(out_dir))
    except BaseException:
        if staging.exists():
            if stage_b_started:  # the labels file was touched: keep the saved predictions; the failed folder blocks a silent rerun
                failed = out_dir.with_name(f"{out_dir.name}.failed_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}")
                os.replace(staging, failed)
                log(f"evaluation did not finish after the labels file was read; partial outputs kept in {failed}")
            else:  # nothing label-related happened: discard the staging folder so that a corrected rerun is possible
                shutil.rmtree(staging, ignore_errors=True)
                log("evaluation stopped before the labels file was touched; staging folder removed (a rerun is possible after fixing the problem)")
        raise
    return {"validated": True, "per_run": per_run, "aggregate": aggregate, "events": events, "out_dir": out_dir, "environment": env}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Final single test-set evaluation of the nine frozen BiLSTM checkpoints (see the protocol).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate-only", action="store_true", help="pre-flight checks only: no test labels read, no inference, nothing written")
    g.add_argument("--confirm-final-test", action="store_true", help="run the real, one-time test evaluation")
    ap.add_argument("--manifest", default=None, help=f"frozen manifest (default {DEFAULT_MANIFEST})")
    ap.add_argument("--data-root", default=None, help="folder with the official SentiMix files (default data/raw)")
    ap.add_argument("--out-dir", default=None, help=f"new output folder (default {DEFAULT_OUT_DIR}); never overwritten")
    ap.add_argument("--repo-root", default=str(ROOT), help="root that the manifest's relative paths refer to (for testing)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    repo = Path(a.repo_root)
    try:
        res = evaluate(a.manifest or repo / DEFAULT_MANIFEST, repo_root=repo, data_root=a.data_root, out_dir=a.out_dir, validate_only=a.validate_only,
                       confirm=a.confirm_final_test, make_figs=not a.no_figures)
    except (ProtocolError, FileExistsError) as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    print("VALIDATION OK: nothing was evaluated." if a.validate_only else f"DONE: results in {res['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
