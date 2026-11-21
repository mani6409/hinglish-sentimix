"""Descriptive analysis of the nine frozen BiLSTM runs (paired bootstrap, error analysis, dev-vs-test gap).

READ-ONLY with respect to every frozen artifact. It uses the saved test predictions
(`results/bilstm_final_test/predictions/`), the saved dev predictions (`results/predictions/`), the official gold-label files and the
frozen checkpoints (only their vocabularies, to count out-of-vocabulary words; no model is run). Nothing here selects a model, a seed or
a setting; all results are exploratory/descriptive and are labelled as such in the report.

Safety:
  * Frozen inputs (nine checkpoints, test prediction files, test files, final-evaluation outputs) are hashed before and after.
  * Everything is written into `<out-dir>.staging` and published with one atomic rename. If `<out-dir>` already exists the program
    refuses to run (there is no overwrite option). It also refuses to write anywhere inside `results/bilstm_final_test`.

Usage (repository root, PyTorch environment):
    python -m src.analyze_bilstm_test --data-root data/raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DEFAULT_ROOT, FILES, TEST_LABELS_FILE, load_split, parse_conll, to_frame
from src.evaluate_bilstm_test import (
    MODES,
    RUNS,
    SEEDS,
    ProtocolError,
    atomic_write,
    atomic_write_json,
    csv_text,
    full_metrics,
    join_by_uid,
    md5_file,
    read_labels_strict,
    read_predictions_csv,
    validate_predictions,
)
from src.metrics import LABELS
from src.preprocess import PRESETS, preprocess_text, tokenize

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = "results/bilstm_final_test"
DEV_PRED_DIR = "results/predictions"
DEFAULT_OUT = "results/bilstm_test_analysis"
MANIFEST = "configs/bilstm_frozen_checkpoints.json"
BOOT_B = 10_000
BOOT_SEED = 20261116  # same fixed seed as the development-set analysis
CMI_EDGES = (23.81, 34.783)  # terciles of the TRAIN split (low <= 23.81 < medium <= 34.783 < high)
CMI_NAMES = ("low", "medium", "high")
NEUTRAL = LABELS.index("neutral")
PAIRS = (("clean", "raw"), ("normalized", "raw"), ("normalized", "clean"))  # (a, b): difference a - b
CASES = ("dev", "test")


# ================================================================= small helpers ===
def ms(vals, d=4) -> str:
    v = np.asarray(vals, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def idx(labels) -> np.ndarray:
    m = {c: i for i, c in enumerate(LABELS)}
    return np.array([m[x] for x in labels], dtype=np.int64)


def f1_from_counts(conf: np.ndarray):
    """conf: (..., 3, 3) counts, rows = true class. Returns (macro_f1, weighted_f1) with zero_division = 0."""
    conf = np.asarray(conf, float)
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    fp = conf.sum(-2) - tp
    fn = conf.sum(-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    support = conf.sum(-1)
    total = support.sum(-1)
    return f1.mean(-1), (f1 * support).sum(-1) / total


def conf_matrix(y, p) -> np.ndarray:
    c = np.zeros((3, 3), dtype=np.int64)
    np.add.at(c, (y, p), 1)
    return c


# ============================================================= integrity / inputs ===
def frozen_input_paths(manifest: dict, data_root: Path) -> dict:
    paths = {}
    for r in manifest["checkpoints"]:
        paths[f"checkpoint:{r['mode']}:{r['seed']}"] = ROOT / r["path"]
    for mode, seed in RUNS:
        paths[f"test_pred:{mode}:{seed}"] = ROOT / FINAL_DIR / "predictions" / f"bilstm_final_test_{mode}_seed{seed}_predictions.csv"
    paths["test_file:unlabeled"] = data_root / manifest["test_files"]["unlabeled"]["file"]
    paths["test_file:labels"] = data_root / manifest["test_files"]["labels"]["file"]
    im = json.loads((ROOT / FINAL_DIR / "integrity_manifest.json").read_text(encoding="utf-8"))
    for rel in im["outputs_md5"]:
        paths[f"final_output:{rel}"] = ROOT / FINAL_DIR / rel
    paths["final_output:integrity_manifest.json"] = ROOT / FINAL_DIR / "integrity_manifest.json"
    paths["manifest"] = ROOT / MANIFEST
    return paths


def hash_all(paths: dict) -> dict:
    return {k: md5_file(p) for k, p in paths.items()}


def verify_against_records(manifest: dict, hashes: dict) -> None:
    """The hashes must equal the frozen records (checkpoints, test files, saved predictions, final outputs)."""
    bad = []
    for r in manifest["checkpoints"]:
        if hashes[f"checkpoint:{r['mode']}:{r['seed']}"] != r["md5"]:
            bad.append(f"checkpoint {r['mode']} {r['seed']}")
    for kind in ("unlabeled", "labels"):
        if hashes[f"test_file:{kind}"] != manifest["test_files"][kind]["md5"]:
            bad.append(f"test file {kind}")
    pm = json.loads((ROOT / FINAL_DIR / "predictions_manifest.json").read_text(encoding="utf-8"))["prediction_files_md5"]
    for mode, seed in RUNS:
        name = f"bilstm_final_test_{mode}_seed{seed}_predictions.csv"
        rec = pm.get(name) or pm.get(f"predictions/{name}")
        if rec is None or hashes[f"test_pred:{mode}:{seed}"] != rec:
            bad.append(f"prediction file {mode} {seed}")
    im = json.loads((ROOT / FINAL_DIR / "integrity_manifest.json").read_text(encoding="utf-8"))["outputs_md5"]
    for rel, h in im.items():
        if hashes[f"final_output:{rel}"] != h:
            bad.append(f"final output {rel}")
    if bad:
        raise ProtocolError("frozen artifacts differ from their records: " + ", ".join(bad))


def read_dev_predictions(path: Path):
    df = pd.read_csv(path, dtype={"uid": np.int64, "label": str, "pred": str})
    if list(df.columns) != ["uid", "label", "pred"]:
        raise ProtocolError(f"{path}: unexpected header {list(df.columns)}")
    return df["uid"].tolist(), df["label"].tolist(), df["pred"].tolist()


def load_runs(manifest: dict, data_root: Path):
    """Aligned (sorted by uid) numpy arrays for dev and test. Returns data[case][(mode, seed)] = (y, p), uids[case], meta."""
    test_unl = to_frame(parse_conll(data_root / FILES["test"]))
    test_labels = read_labels_strict(data_root / TEST_LABELS_FILE)
    dev_df = load_split("dev", root=data_root)
    uids = {"dev": sorted(dev_df["uid"].tolist()), "test": sorted(test_unl["uid"].tolist())}
    if len(uids["dev"]) != manifest["expected"]["dev_size"] or len(uids["test"]) != manifest["expected"]["n_test"]:
        raise ProtocolError("unexpected split sizes")
    dev_gold_official = dict(zip(dev_df["uid"], dev_df["label"]))
    data = {"dev": {}, "test": {}}
    for mode, seed in RUNS:
        tp = ROOT / FINAL_DIR / "predictions" / f"bilstm_final_test_{mode}_seed{seed}_predictions.csv"
        u, pred = read_predictions_csv(tp)
        validate_predictions(uids["test"], u, pred, f"test predictions {mode}/{seed}")
        gold = join_by_uid(u, test_labels)
        order = np.argsort(u)
        data["test"][(mode, seed)] = (idx(np.array(gold)[order]), idx(np.array(pred)[order]))
        dp = ROOT / DEV_PRED_DIR / f"bilstm_full_{mode}_s{seed}_seed{seed}_dev.csv"
        u, gold, pred = read_dev_predictions(dp)
        validate_predictions(uids["dev"], u, pred, f"dev predictions {mode}/{seed}")
        if [dev_gold_official[x] for x in u] != gold:
            raise ProtocolError(f"dev predictions {mode}/{seed}: label column differs from the official dev labels")
        order = np.argsort(u)
        data["dev"][(mode, seed)] = (idx(np.array(gold)[order]), idx(np.array(pred)[order]))
    # the gold labels must be identical in all nine runs of a split
    for case in CASES:
        ys = [v[0] for v in data[case].values()]
        if any(not np.array_equal(ys[0], y) for y in ys[1:]):
            raise ProtocolError(f"{case}: gold labels differ between runs")
    texts = {
        "test": test_unl.set_index("uid").loc[uids["test"]],
        "dev": dev_df.set_index("uid").loc[uids["dev"]],
    }
    return data, uids, texts, test_labels, dev_df


# ================================================================ bootstrap =======
def bootstrap(runs: dict, n: int, B: int, seed: int) -> dict:
    """Paired bootstrap over tweets: every resample (the same multiset of uids) is applied to all nine runs.

    Returns per_run[(mode, seed)] = {"macro_f1": (B,), "weighted_f1": (B,)}. Resample weights are multinomial counts, which is the
    same as drawing n uids with replacement.
    """
    rng = np.random.default_rng(seed)
    onehot = {}
    for key, (y, p) in runs.items():
        cell = y * 3 + p
        m = np.zeros((n, 9), dtype=np.float64)
        m[np.arange(n), cell] = 1.0
        onehot[key] = m
    out = {k: {"macro_f1": np.empty(B), "weighted_f1": np.empty(B)} for k in runs}
    done, chunk = 0, 500
    while done < B:
        b = min(chunk, B - done)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)
        for key in runs:
            conf = (W @ onehot[key]).reshape(b, 3, 3)
            mf, wf = f1_from_counts(conf)
            out[key]["macro_f1"][done:done + b] = mf
            out[key]["weighted_f1"][done:done + b] = wf
        done += b
    return out


def pct(a, lo=2.5, hi=97.5):
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def bootstrap_tables(boot: dict, point: dict, metric: str) -> dict:
    """Mode means over seeds per resample, mode CIs, pairwise differences (seed-averaged and seed-index-paired)."""
    mode_boot = {m: np.mean([boot[(m, s)][metric] for s in SEEDS], axis=0) for m in MODES}
    mode_point = {m: float(np.mean([point[(m, s)][metric] for s in SEEDS])) for m in MODES}
    modes = {m: {"point": mode_point[m], "ci95": pct(mode_boot[m])} for m in MODES}
    pairs = []
    for a, b in PAIRS:
        d = mode_boot[a] - mode_boot[b]
        pairs.append({
            "a": a, "b": b, "point": mode_point[a] - mode_point[b],
            "ci95": pct(d), "ci95_bonferroni3": pct(d, 100 * 0.05 / 3 / 2, 100 - 100 * 0.05 / 3 / 2),
            "boot_mean": float(d.mean()), "boot_sd": float(d.std(ddof=1)),
            "frac_boot_diff_gt0": float((d > 0).mean()),
            "per_seed_point": [point[(a, s)][metric] - point[(b, s)][metric] for s in SEEDS],
            "per_seed_ci95": [pct(boot[(a, s)][metric] - boot[(b, s)][metric]) for s in SEEDS],
        })
    return {"modes": modes, "pairs": pairs}


# ================================================================ error analysis ===
def point_metrics(runs: dict) -> dict:
    out = {}
    for key, (y, p) in runs.items():
        out[key] = full_metrics([LABELS[i] for i in y], [LABELS[i] for i in p])
    return out


def error_breakdown(y, p) -> dict:
    wrong = y != p
    n_err = int(wrong.sum())
    gold_neu = wrong & (y == NEUTRAL)
    pred_neu = wrong & (p == NEUTRAL)
    neu_in = gold_neu | pred_neu
    flip = wrong & (y != NEUTRAL) & (p != NEUTRAL)
    conf = conf_matrix(y, p)
    out = {
        "n": int(len(y)), "errors": n_err, "error_rate": n_err / len(y),
        "gold_neutral_errors": int(gold_neu.sum()), "predicted_neutral_errors": int(pred_neu.sum()),
        "neutral_involved_errors": int(neu_in.sum()), "negative_positive_flips": int(flip.sum()),
        "share_neutral_involved": float(neu_in.sum() / n_err) if n_err else 0.0,
        "share_gold_neutral": float(gold_neu.sum() / n_err) if n_err else 0.0,
        "share_predicted_neutral": float(pred_neu.sum() / n_err) if n_err else 0.0,
        "share_flips": float(flip.sum() / n_err) if n_err else 0.0,
    }
    for i, a in enumerate(LABELS):
        for j, b in enumerate(LABELS):
            if i != j:
                out[f"true_{a}__pred_{b}"] = int(conf[i, j])
    return out


def bucket_ids(values, edges) -> np.ndarray:
    return np.digitize(np.asarray(values, float), edges, right=True)  # 0: <= e0, 1: <= e1, 2: above


def bucket_metrics(runs: dict, bucket: np.ndarray, names) -> list:
    rows = []
    for mode in MODES:
        for k, name in enumerate(names):
            mask = bucket == k
            accs, mfs = [], []
            for s in SEEDS:
                y, p = runs[(mode, s)]
                accs.append(float((y[mask] == p[mask]).mean()))
                mfs.append(float(f1_from_counts(conf_matrix(y[mask], p[mask]))[0]))
            rows.append({"mode": mode, "bucket": name, "n": int(mask.sum()), "accuracy_mean": float(np.mean(accs)),
                         "accuracy_sd": float(np.std(accs, ddof=1)), "macro_f1_mean": float(np.mean(mfs)),
                         "macro_f1_sd": float(np.std(mfs, ddof=1))})
    return rows


# ================================================================ gap investigation ===
def reweighted_f1(runs: dict, y_ref_props: np.ndarray, y_true: np.ndarray) -> dict:
    """Macro/weighted F1 of each run after re-weighting its tweets so the gold-class mix equals y_ref_props."""
    own = np.bincount(y_true, minlength=3) / len(y_true)
    w_class = y_ref_props / own
    out = {}
    for key, (y, p) in runs.items():
        conf = np.zeros((3, 3))
        np.add.at(conf, (y, p), w_class[y])
        out[key] = tuple(float(v) for v in f1_from_counts(conf))
    return out


def oov_stats(manifest: dict, frames: dict) -> list:
    import torch

    from src.bilstm_data import TextEncoder

    rows = []
    for mode in MODES:
        encs = []
        for s in SEEDS:
            ck = next(r for r in manifest["checkpoints"] if r["mode"] == mode and r["seed"] == s)
            payload = torch.load(ROOT / ck["path"], map_location="cpu", weights_only=True)
            encs.append(TextEncoder.from_dict(payload["encoder"]))
        if any(e.word_vocab.itos != encs[0].word_vocab.itos for e in encs[1:]):
            raise ProtocolError(f"{mode}: word vocabularies differ between seeds")
        enc = encs[0]
        for case in CASES:
            toks = []
            for t in frames[case]["text"]:
                words = tokenize(preprocess_text(t, **PRESETS[mode]))
                toks.append([w.lower() if enc.lowercase_words and w not in enc.special_words else w for w in words])
            flat = [w for ws in toks for w in ws]
            oov_flat = [w for w in flat if w not in enc.word_vocab.stoi]
            types = set(flat)
            oov_types = {w for w in types if w not in enc.word_vocab.stoi}
            rows.append({
                "mode": mode, "split": case, "vocab_size": len(enc.word_vocab), "tokens": len(flat),
                "oov_token_rate": len(oov_flat) / len(flat), "types": len(types), "oov_type_rate": len(oov_types) / len(types),
                "tweets_with_oov": float(np.mean([any(w not in enc.word_vocab.stoi for w in ws) for ws in toks])),
                "oov_per_tweet": len(oov_flat) / len(toks),
            })
    return rows


HASHTAG_RE = re.compile(r"#\s*\w+")


def mask_text(t: str) -> str:
    """Anonymise a tweet for display: hashtags -> HASHTAG, then the project's `clean` preprocessing, which turns @mentions
    (including the split-up '@ name _ part' form of this data) into USER and links (including 'https // t . co / x') into URL."""
    return preprocess_text(HASHTAG_RE.sub("HASHTAG", str(t)), **PRESETS["clean"])


# ===================================================================== build =======
def build(data_root: Path, out_dir: Path, B: int, log=print) -> dict:
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    final_abs = (ROOT / FINAL_DIR).resolve()
    if final_abs in (out_dir.resolve(), *out_dir.resolve().parents):
        raise ProtocolError("refusing to write inside the frozen final-test folder")
    staging = out_dir.with_name(out_dir.name + ".staging")
    if out_dir.exists() or staging.exists():
        raise ProtocolError(f"{out_dir} (or its staging folder) already exists; this analysis never overwrites")
    paths = frozen_input_paths(manifest, data_root)
    log("hashing frozen inputs (before) ...")
    before = hash_all(paths)
    verify_against_records(manifest, before)

    data, uids, frames, test_labels, dev_df = load_runs(manifest, data_root)
    n = {c: len(uids[c]) for c in CASES}
    point = {c: point_metrics(data[c]) for c in CASES}

    # --- consistency of the saved test metrics with an independent recomputation
    saved = json.loads((ROOT / FINAL_DIR / "metrics_per_run.json").read_text(encoding="utf-8"))
    max_diff_test = max(abs(point["test"][(m, s)][k] - saved[m][str(s)][k])
                        for m, s in RUNS for k in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"))
    max_diff_dev = 0.0
    for m, s in RUNS:
        d = json.loads((ROOT / "results" / f"bilstm_full_{m}_s{s}_dev_scores.json").read_text(encoding="utf-8"))["per_seed"][0]["metrics"]
        max_diff_dev = max(max_diff_dev, *(abs(point["dev"][(m, s)][k] - d[k]) for k in ("accuracy", "macro_f1", "weighted_f1")))
    from sklearn.metrics import f1_score
    sk_diff = max(abs(f1_score(data[c][k][0], data[c][k][1], average="macro") - point[c][k]["macro_f1"]) for c in CASES for k in RUNS)
    log(f"max |recomputed - saved|: test {max_diff_test:.2e}, dev {max_diff_dev:.2e}; sklearn macro-F1 {sk_diff:.2e}")
    if max(max_diff_test, max_diff_dev, sk_diff) > 1e-9:
        raise ProtocolError("recomputed metrics do not match the saved ones")

    # --- bootstrap
    log(f"paired bootstrap: B={B}, seed={BOOT_SEED} ...")
    boots = {c: bootstrap(data[c], n[c], B, BOOT_SEED) for c in CASES}
    boot_tab = {c: {met: bootstrap_tables(boots[c], point[c], met) for met in ("macro_f1", "weighted_f1")} for c in CASES}

    # --- seed variation (kept separate from sampling uncertainty)
    seed_rows = []
    for c in CASES:
        for m in MODES:
            for met in ("macro_f1", "weighted_f1", "accuracy"):
                v = [point[c][(m, s)][met] for s in SEEDS]
                seed_rows.append([c, m, met] + [f"{x:.6f}" for x in v] + [f"{np.mean(v):.6f}", f"{np.std(v, ddof=1):.6f}", f"{max(v) - min(v):.6f}"])

    # --- per-class table
    class_rows = []
    for c in CASES:
        for m in MODES:
            for ci, cl in enumerate(LABELS):
                for k in ("precision", "recall", "f1"):
                    v = [point[c][(m, s)]["per_class"][cl][k] for s in SEEDS]
                    class_rows.append([c, m, cl, k, f"{np.mean(v):.6f}", f"{np.std(v, ddof=1):.6f}",
                                       point[c][(m, RUNS[0][1])]["per_class"][cl]["support"]])

    # --- error breakdown per run, and summed confusion matrices
    err_rows, err_summary = [], {}
    conf_sum = {}
    for c in CASES:
        for m in MODES:
            per = [error_breakdown(*data[c][(m, s)]) for s in SEEDS]
            err_summary[(c, m)] = per
            for s, e in zip(SEEDS, per):
                err_rows.append([c, m, s] + [e[k] if isinstance(e[k], int) else f"{e[k]:.6f}" for k in per[0]])
            conf_sum[(c, m)] = sum(conf_matrix(*data[c][(m, s)]) for s in SEEDS)
    err_header = ["split", "mode", "seed"] + list(per[0].keys())

    # --- buckets (CMI from the existing per-tweet file; length from train terciles)
    cmi_df = pd.read_csv(ROOT / "results" / "cmi_per_tweet.csv")
    train_df = load_split("train", root=data_root)
    len_edges = tuple(float(x) for x in np.quantile(train_df["n_tokens"], [1 / 3, 2 / 3]))
    bucket_rows = []
    cmi_share = {}
    for c in CASES:
        sub = cmi_df[cmi_df["split"] == c].set_index("uid").loc[uids[c]]
        b_cmi = bucket_ids(sub["cmi"].to_numpy(), CMI_EDGES)
        cmi_share[c] = [float((b_cmi == k).mean()) for k in range(3)]
        for r in bucket_metrics(data[c], b_cmi, CMI_NAMES):
            bucket_rows.append({"split": c, "by": "cmi", **r})
        b_len = bucket_ids(frames[c]["n_tokens"].to_numpy(), len_edges)
        for r in bucket_metrics(data[c], b_len, ("short", "medium", "long")):
            bucket_rows.append({"split": c, "by": "length", **r})

    # --- gap investigation
    y_dev, y_test = data["dev"][RUNS[0]][0], data["test"][RUNS[0]][0]
    props = {c: np.bincount(data[c][RUNS[0]][0], minlength=3) / n[c] for c in CASES}
    props["train"] = np.array([float((train_df["label"] == cl).mean()) for cl in LABELS])
    pred_props = {(c, m): np.mean([np.bincount(data[c][(m, s)][1], minlength=3) / n[c] for s in SEEDS], axis=0) for c in CASES for m in MODES}
    rew = reweighted_f1({k: v for k, v in data["test"].items()}, props["dev"], y_test)
    rew_mode = {m: (float(np.mean([rew[(m, s)][0] for s in SEEDS])), float(np.mean([rew[(m, s)][1] for s in SEEDS]))) for m in MODES}
    oov = oov_stats(manifest, frames)
    length_stats = {}
    for c in CASES:
        tok = frames[c]["n_tokens"].to_numpy()
        chars = frames[c]["text"].str.len().to_numpy()
        length_stats[c] = {"tokens_mean": float(tok.mean()), "tokens_median": float(np.median(tok)), "tokens_p90": float(np.percentile(tok, 90)),
                           "chars_mean": float(chars.mean()), "chars_median": float(np.median(chars))}
    cmi_stats = {c: {"mean": float(cmi_df[cmi_df["split"] == c]["cmi"].mean()), "share_buckets": cmi_share[c]} for c in CASES}
    checks = {
        "dev_uids_unique": len(set(uids["dev"])) == n["dev"], "test_uids_unique": len(set(uids["test"])) == n["test"],
        "dev_test_uid_overlap": len(set(uids["dev"]) & set(uids["test"])),
        "train_dev_uid_overlap": len(set(train_df["uid"]) & set(uids["dev"])),
        "train_test_uid_overlap": len(set(train_df["uid"]) & set(uids["test"])),
        "test_label_uids_equal_test_uids": set(test_labels) == set(uids["test"]),
        "label_order": list(LABELS), "gold_labels_identical_across_nine_runs": True,
        "max_abs_diff_saved_vs_recomputed_test": max_diff_test, "max_abs_diff_saved_vs_recomputed_dev": max_diff_dev,
        "max_abs_diff_sklearn_macro_f1": sk_diff,
    }

    # --- masked examples (rule-based selection)
    wrong_all = np.all([data["test"][k][0] != data["test"][k][1] for k in RUNS], axis=0)
    cand = np.flatnonzero(wrong_all)
    rng = np.random.default_rng(BOOT_SEED)
    pick = np.sort(rng.choice(cand, size=min(10, len(cand)), replace=False)) if len(cand) else np.array([], dtype=int)
    cmi_test = cmi_df[cmi_df["split"] == "test"].set_index("uid").loc[uids["test"]]["cmi"].to_numpy()
    ex_rows = []
    for j, i in enumerate(pick, 1):
        preds = {LABELS[data["test"][k][1][i]] for k in RUNS}
        pred_txt = next(iter(preds)) if len(preds) == 1 else "/".join(sorted(preds))
        gold = LABELS[y_test[i]]
        et = ("gold-neutral error" if gold == "neutral" else "predicted-neutral error" if preds == {"neutral"} else "negative/positive flip")
        ex_rows.append([j, gold, pred_txt, et, CMI_NAMES[int(bucket_ids([cmi_test[i]], CMI_EDGES)[0])], mask_text(frames["test"]["text"].iloc[i])])

    result = dict(manifest=manifest, before=before, paths=paths, data=data, n=n, point=point, boot_tab=boot_tab, boots=boots,
                  seed_rows=seed_rows, class_rows=class_rows, err_rows=err_rows, err_header=err_header, err_summary=err_summary,
                  conf_sum=conf_sum, bucket_rows=bucket_rows, len_edges=len_edges, props=props, pred_props=pred_props,
                  rew_mode=rew_mode, oov=oov, length_stats=length_stats, cmi_stats=cmi_stats, checks=checks, ex_rows=ex_rows,
                  n_all_wrong=int(wrong_all.sum()), B=B)
    return result


# ===================================================================== output =======
def fmt_ci(ci, d=4):
    return f"[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"


def render_report(R: dict) -> str:
    pt, bt = R["point"], R["boot_tab"]
    L = []
    w = L.append
    w("# BiLSTM final analysis (descriptive, exploratory)\n")
    w("Everything below is computed from the **saved** predictions of the nine frozen runs (3 preprocessing modes x seeds 42, 43, 44) and the official gold labels. "
      "Nothing was retrained, no model was run on the test set again, and no result here selects a model, a seed or a setting. "
      "The single pre-specified test evaluation is in `results/bilstm_final_test/`; this analysis is **exploratory** and was written after those results were known. "
      "Earlier descriptive test access by the team's EDA scripts is disclosed in section 8 of `docs/bilstm_final_evaluation_protocol.md`.\n")

    w("## 1. Final results (mean ± sample std over seeds 42, 43, 44)\n")
    rows = []
    for m in MODES:
        row = [m]
        for c in ("test", "dev"):
            for met in ("macro_f1", "weighted_f1", "accuracy"):
                row.append(ms([pt[c][(m, s)][met] for s in SEEDS]))
        rows.append(row)
    w(md_table(["mode", "test macro F1", "test weighted F1", "test accuracy", "dev macro F1", "dev weighted F1", "dev accuracy"], rows))
    w("\nDev numbers are the best epoch per run (early stopping on dev weighted F1), so they are optimistically selected; test numbers are the unbiased estimate for these frozen models. "
      "The official metric of the task is weighted F1.\n")

    w("## 2. Statistical analysis\n")
    w(f"**Two kinds of uncertainty are reported separately.**\n\n"
      "* **Seed variation** (training randomness): the sample standard deviation across the three seeds of one mode, and the range. No resampling is involved.\n"
      f"* **Test-sampling uncertainty** (the test set is one fixed sample of 3,000 tweets): a **paired bootstrap** over tweets, {R['B']:,} resamples, fixed seed {BOOT_SEED}. "
      "In every resample the *same* multiset of tweet uids is used for all nine runs, so each comparison is paired. "
      "For each resample the macro F1 of every run is computed, averaged over the three seeds of a mode, and the modes are compared. "
      "The interval therefore describes the uncertainty from the test sample **for these three trained seeds**; it does not include the uncertainty from training new seeds (that is what the seed variation shows, roughly).\n")
    w("### 2.1 Seed variation\n")
    w(md_table(["split", "mode", "metric", "s42", "s43", "s44", "mean", "sd (ddof=1)", "range"], R["seed_rows"]) + "\n")
    w("### 2.2 Bootstrap intervals for the test set\n")
    for met, title in (("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")):
        w(f"**{title} per mode (mean over seeds, 95% percentile interval from the test-sampling bootstrap)**\n")
        w(md_table(["mode", "mean", "95% interval"], [[m, f"{bt['test'][met]['modes'][m]['point']:.4f}", f"[{bt['test'][met]['modes'][m]['ci95'][0]:.4f}, {bt['test'][met]['modes'][m]['ci95'][1]:.4f}]"] for m in MODES]) + "\n")
        w(f"**{title}: pairwise differences a − b (mean over seeds)**\n")
        rows = []
        for p in bt["test"][met]["pairs"]:
            rows.append([f"{p['a']} − {p['b']}", f"{p['point']:+.4f}", fmt_ci(p["ci95"]), fmt_ci(p["ci95_bonferroni3"]),
                         "yes" if p["ci95"][0] > 0 or p["ci95"][1] < 0 else "no",
                         ", ".join(f"{x:+.4f}" for x in p["per_seed_point"])])
        w(md_table(["comparison", "difference", "95% interval", "Bonferroni 98.3% interval (3 comparisons)", "95% interval excludes 0", "same-seed differences (s42, s43, s44)"], rows) + "\n")
    w("### 2.3 The same intervals on the development set (for reference)\n")
    rows = []
    for p in bt["dev"]["macro_f1"]["pairs"]:
        rows.append([f"{p['a']} − {p['b']}", f"{p['point']:+.4f}", fmt_ci(p["ci95"]), fmt_ci(p["ci95_bonferroni3"])])
    w(md_table(["macro F1 comparison", "difference", "95% interval", "Bonferroni 98.3% interval"], rows) + "\n")
    w("### 2.4 How to read these intervals\n")
    w("* A 95% interval is the range of differences that the resampling procedure produced for 95% of resamples. It is a statement about the test sample, conditional on the three trained models per mode.\n"
      "* There are three pairwise comparisons on the same data. The unadjusted intervals are the main presentation; the Bonferroni intervals show how much wider they become when the three comparisons are treated as one family. Neither is a hypothesis test and **no statistical significance is claimed**.\n"
      "* An interval that contains zero means the data do not separate the modes; it is **not** evidence that they are equivalent. An interval that excludes zero says the difference is unlikely to be only test-sampling noise for these three seeds; it does not show that a new seed would reproduce it, because seed variation is separate (section 2.1).\n"
      "* The comparisons were chosen after seeing the dev results; the test intervals do not select anything.\n")
    ex = []
    for p in bt["test"]["macro_f1"]["pairs"]:
        lo, hi = p["ci95"]
        ex.append(f"* **{p['a']} − {p['b']}** (macro F1): difference {p['point']:+.4f}, 95% interval {fmt_ci(p['ci95'])}"
                  f" — {'excludes zero' if lo > 0 or hi < 0 else 'contains zero'}; seed-to-seed range within the two modes: "
                  f"{max(pt['test'][(p['a'], s)]['macro_f1'] for s in SEEDS) - min(pt['test'][(p['a'], s)]['macro_f1'] for s in SEEDS):.4f} / "
                  f"{max(pt['test'][(p['b'], s)]['macro_f1'] for s in SEEDS) - min(pt['test'][(p['b'], s)]['macro_f1'] for s in SEEDS):.4f}.")
    w("**Test macro F1, in numbers:**\n\n" + "\n".join(ex) + "\n")

    ev = []
    for met_name, met in (("macro F1", "macro_f1"), ("weighted F1", "weighted_f1")):
        for p in bt["test"][met]["pairs"]:
            ex0 = p["ci95"][0] > 0 or p["ci95"][1] < 0
            exb = p["ci95_bonferroni3"][0] > 0 or p["ci95_bonferroni3"][1] < 0
            signs = {np.sign(x) for x in p["per_seed_point"]}
            ev.append((p, met_name, ex0, exb, len(signs) == 1 and 0 not in signs))
    lines = []
    for p in bt["test"]["macro_f1"]["pairs"]:
        mine = {e[1]: e for e in ev if e[0]["a"] == p["a"] and e[0]["b"] == p["b"]}
        mf, wf = mine["macro F1"], mine["weighted F1"]
        lines.append(f"* **{p['a']} − {p['b']}**: macro F1 interval {'excludes' if mf[2] else 'contains'} 0 (Bonferroni: {'excludes' if mf[3] else 'contains'} 0); "
                     f"weighted F1 interval {'excludes' if wf[2] else 'contains'} 0 (Bonferroni: {'excludes' if wf[3] else 'contains'} 0); "
                     f"the three same-seed differences in macro F1 {'all have the same sign' if mf[4] else 'do not all have the same sign'} "
                     f"(range {min(p['per_seed_point']):+.4f} to {max(p['per_seed_point']):+.4f}).")
    w("**Evidence per comparison (test, both primary metrics):**\n\n" + "\n".join(lines) + "\n")
    strong = [f"{e[0]['a']} − {e[0]['b']}" for e in ev if e[1] == "macro F1" and e[2] and e[3] and e[4]
              and any(x[0]['a'] == e[0]['a'] and x[0]['b'] == e[0]['b'] and x[1] == "weighted F1" and x[2] and x[3] and x[4] for x in ev)]
    w("Comparisons for which all of the following hold: the unadjusted and Bonferroni intervals exclude 0 on **both** macro and weighted F1, and the three same-seed macro-F1 differences share a sign: "
      + (", ".join(strong) if strong else "none") + ". Comparisons not listed are not separated by these data. "
      "This describes the strength of evidence in this sample; it is not a significance test and does not establish that a different set of seeds or test tweets would give the same ordering.\n")

    w("## 3. Error analysis (computed counts only)\n")
    w("Class order: negative, neutral, positive. Rows = true class, columns = predicted class. Matrices are summed over the three seeds of a mode (3 x 3,000 predictions).\n")
    for c in ("test", "dev"):
        w(f"### 3.{1 if c == 'test' else 2} Confusion matrices ({c})\n")
        for m in MODES:
            cm = R["conf_sum"][(c, m)]
            w(f"**{m}**\n")
            w(md_table(["true \\ pred"] + list(LABELS), [[LABELS[i]] + [int(cm[i, j]) for j in range(3)] for i in range(3)]) + "\n")
    w("### 3.3 Per-class precision, recall and F1 (mean ± sd over seeds)\n")
    for c in ("test", "dev"):
        rows = []
        for m in MODES:
            for cl in LABELS:
                vals = {k: [pt[c][(m, s)]["per_class"][cl][k] for s in SEEDS] for k in ("precision", "recall", "f1")}
                rows.append([m, cl, pt[c][(m, 42)]["per_class"][cl]["support"]] + [ms(vals[k]) for k in ("precision", "recall", "f1")])
        w(f"**{c}**\n")
        w(md_table(["mode", "class", "support", "precision", "recall", "F1"], rows) + "\n")
    w("### 3.4 Where the errors are (neutral-class analysis)\n")
    w("Definitions, all per run, then mean ± sd over seeds: an *error* is a tweet whose predicted label differs from the gold label. "
      "**Neutral-involved error** = gold is neutral **or** predicted is neutral (a negative/positive confusion is the only kind that never involves neutral). "
      "**Gold-neutral error** = gold neutral, predicted negative/positive. **Predicted-neutral error** = predicted neutral, gold negative/positive. "
      "**Negative↔positive flip** = gold and predicted are the two non-neutral classes, in either order. The three kinds are disjoint and add up to 100%; "
      "neutral-involved = gold-neutral + predicted-neutral.\n")
    for c in ("test", "dev"):
        rows = []
        for m in MODES:
            per = R["err_summary"][(c, m)]
            rows.append([m, ms([e["errors"] for e in per], 1),
                         ms([100 * e["share_neutral_involved"] for e in per], 1) + "%",
                         ms([100 * e["share_gold_neutral"] for e in per], 1) + "%",
                         ms([100 * e["share_predicted_neutral"] for e in per], 1) + "%",
                         ms([100 * e["share_flips"] for e in per], 1) + "%"])
        w(f"**{c}** (3,000 tweets)\n")
        w(md_table(["mode", "errors per run", "neutral-involved", "gold-neutral", "predicted-neutral", "negative↔positive flips"], rows) + "\n")
    w("### 3.5 Misclassification patterns by tweet property\n")
    w(f"Accuracy and macro F1 (mean over seeds) inside code-mixing-index buckets (CMI terciles of the **train** split, edges {CMI_EDGES[0]} and {CMI_EDGES[1]}; "
      f"low <= {CMI_EDGES[0]}, high > {CMI_EDGES[1]}) and tweet-length buckets (word-count terciles of the train split, edges {R['len_edges'][0]:.1f} and {R['len_edges'][1]:.1f}). "
      "These are descriptions of where the models do better or worse; they are not causal explanations.\n")
    for by in ("cmi", "length"):
        for c in ("test", "dev"):
            rows = [[r["mode"], r["bucket"], r["n"], f"{r['accuracy_mean']:.4f} ± {r['accuracy_sd']:.4f}", f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_sd']:.4f}"]
                    for r in R["bucket_rows"] if r["split"] == c and r["by"] == by]
            w(f"**{c}, by {by}**\n")
            w(md_table(["mode", "bucket", "n", "accuracy", "macro F1"], rows) + "\n")
    def extreme(c, by):
        out = []
        for m in MODES:
            rs = [r for r in R["bucket_rows"] if r["split"] == c and r["by"] == by and r["mode"] == m]
            hi = max(rs, key=lambda r: r["accuracy_mean"])
            lo = min(rs, key=lambda r: r["accuracy_mean"])
            out.append(f"{m}: highest {hi['bucket']} ({hi['accuracy_mean']:.3f}), lowest {lo['bucket']} ({lo['accuracy_mean']:.3f})")
        return "; ".join(out)
    w("Highest- and lowest-accuracy bucket per mode (mean over seeds): " + " | ".join(f"**{c}, {by}**: {extreme(c, by)}" for by in ("cmi", "length") for c in ("test", "dev")) + ".\n")
    w("### 3.6 Examples\n")
    w(f"{R['n_all_wrong']} test tweets are misclassified by **all nine** runs. {len(R['ex_rows'])} of them were drawn at random (fixed seed {BOOT_SEED}; no hand selection). "
      "The tweets come from the SemEval-2020 Task 9 data, released under CC BY 4.0 (Patwa et al., 2020); @mentions are replaced by USER and links by URL, and no ids are shown. "
      "No reason for the error is given, because none was analysed.\n")
    if R["ex_rows"]:
        w(md_table(["#", "gold", "predicted (all nine)", "error kind", "CMI bucket", "tweet (masked)"], [[r[0], r[1], r[2], r[3], r[4], str(r[5]).replace("|", "\\|")] for r in R["ex_rows"]]) + "\n")

    w("## 4. Investigation of the dev–test gap\n")
    gap_rows = [[m, f"{np.mean([pt['dev'][(m, s)]['macro_f1'] for s in SEEDS]):.4f}", f"{np.mean([pt['test'][(m, s)]['macro_f1'] for s in SEEDS]):.4f}",
                 f"{np.mean([pt['test'][(m, s)]['macro_f1'] for s in SEEDS]) - np.mean([pt['dev'][(m, s)]['macro_f1'] for s in SEEDS]):+.4f}",
                 f"{R['rew_mode'][m][0]:.4f}"] for m in MODES]
    w("Test macro F1 is higher than dev macro F1 for every mode:\n")
    w(md_table(["mode", "dev macro F1", "test macro F1", "gap", "test macro F1 re-weighted to the dev class mix"], gap_rows) + "\n")
    ch = R["checks"]
    w("### 4.1 Evaluation and data integrity checks\n")
    w(md_table(["check", "result"], [
        ["dev and test uids unique", f"{ch['dev_uids_unique']} / {ch['test_uids_unique']}"],
        ["uid overlap dev∩test / train∩dev / train∩test", f"{ch['dev_test_uid_overlap']} / {ch['train_dev_uid_overlap']} / {ch['train_test_uid_overlap']}"],
        ["test label uids equal test tweet uids", ch["test_label_uids_equal_test_uids"]],
        ["gold labels identical across the nine runs (per split)", ch["gold_labels_identical_across_nine_runs"]],
        ["label order used everywhere", ", ".join(ch["label_order"])],
        ["max |recomputed − saved| test metrics (5 metrics x 9 runs)", f"{ch['max_abs_diff_saved_vs_recomputed_test']:.2e}"],
        ["max |recomputed − saved| dev metrics (3 metrics x 9 runs)", f"{ch['max_abs_diff_saved_vs_recomputed_dev']:.2e}"],
        ["max |sklearn macro F1 − project macro F1| (both splits, 9 runs)", f"{ch['max_abs_diff_sklearn_macro_f1']:.2e}"],
    ]) + "\n")
    w("Predictions and gold labels were joined by uid (never by row position); the same metric function is used for dev and test, and it reproduces the saved numbers of both. "
      "These checks make an implementation, alignment or label-mapping error unlikely as the cause of the gap.\n")
    w("### 4.2 Class distributions\n")
    pr = R["props"]
    w(md_table(["split", "negative", "neutral", "positive"], [[s] + [f"{100 * v:.1f}%" for v in pr[s]] for s in ("train", "dev", "test")]) + "\n")
    w("Share of predicted labels (mean over seeds):\n")
    w(md_table(["split", "mode", "negative", "neutral", "positive"], [[c, m] + [f"{100 * v:.1f}%" for v in R["pred_props"][(c, m)]] for c in ("dev", "test") for m in MODES]) + "\n")
    w("The last column of the first table above re-weights each test tweet so that the gold-class mix equals the dev mix. "
      "The difference between that column and the plain test macro F1 is the part of the gap attributable to the class prior alone.\n")
    w("### 4.3 Tweet length, code mixing, and out-of-vocabulary words\n")
    ls = R["length_stats"]
    w(md_table(["split", "words mean", "words median", "words 90th pct", "chars mean", "chars median", "CMI mean", "CMI low/medium/high share"],
               [[c, f"{ls[c]['tokens_mean']:.2f}", f"{ls[c]['tokens_median']:.1f}", f"{ls[c]['tokens_p90']:.1f}", f"{ls[c]['chars_mean']:.1f}", f"{ls[c]['chars_median']:.1f}",
                 f"{R['cmi_stats'][c]['mean']:.2f}", " / ".join(f"{100 * v:.1f}%" for v in R["cmi_stats"][c]["share_buckets"])] for c in CASES]) + "\n")
    w("Out-of-vocabulary (OOV) rates against each mode's **train-only** word vocabulary (the vocabulary stored in the checkpoints; the same for all three seeds of a mode):\n")
    w(md_table(["mode", "split", "vocabulary size", "OOV token rate", "OOV type rate", "tweets with ≥1 OOV word", "OOV words per tweet"],
               [[o["mode"], o["split"], o["vocab_size"], f"{100 * o['oov_token_rate']:.2f}%", f"{100 * o['oov_type_rate']:.2f}%",
                 f"{100 * o['tweets_with_oov']:.1f}%", f"{o['oov_per_tweet']:.3f}"] for o in R["oov"]]) + "\n")
    ep = {c: {m: float(np.mean([e["errors"] for e in R["err_summary"][(c, m)]])) for m in MODES} for c in CASES}
    fl = {c: {m: float(np.mean([e["negative_positive_flips"] for e in R["err_summary"][(c, m)]])) for m in MODES} for c in CASES}
    oovd = {(o["mode"], o["split"]): o for o in R["oov"]}
    cls_gap = {m: {cl: float(np.mean([pt["test"][(m, s)]["per_class"][cl]["f1"] - pt["dev"][(m, s)]["per_class"][cl]["f1"] for s in SEEDS])) for cl in LABELS} for m in MODES}
    w("**Measured differences between dev and test** (computed from the tables above):\n")
    w("* Errors per run fall from " + ", ".join(f"{ep['dev'][m]:.0f} to {ep['test'][m]:.0f} ({m})" for m in MODES) + ". "
      "Of that reduction, negative↔positive flips account for " + ", ".join(f"{fl['dev'][m] - fl['test'][m]:.0f} of {ep['dev'][m] - ep['test'][m]:.0f} ({m})" for m in MODES)
      + "; the remainder is fewer errors that involve neutral.")
    w("* F1 gain from dev to test by class (negative / neutral / positive): " + "; ".join(f"{m}: " + " / ".join(f"{cls_gap[m][cl]:+.3f}" for cl in LABELS) for m in MODES) + ".")
    w("* OOV token rate is " + ", ".join(f"{100 * (oovd[(m, 'test')]['oov_token_rate'] - oovd[(m, 'dev')]['oov_token_rate']):+.1f} percentage points ({m})" for m in MODES)
      + " on test relative to dev. More unseen words would be expected to make test harder, not easier, so OOV does not account for the higher test scores.")
    w(f"* Mean tweet length is {ls['dev']['tokens_mean']:.1f} words on dev and {ls['test']['tokens_mean']:.1f} on test; mean CMI is {R['cmi_stats']['dev']['mean']:.1f} and {R['cmi_stats']['test']['mean']:.1f}. "
      "These two properties are nearly the same in both splits.")
    w("* The models predict neutral more often on test than on dev (" + ", ".join(f"{100 * R['pred_props'][('dev', m)][1]:.1f}% to {100 * R['pred_props'][('test', m)][1]:.1f}% ({m})" for m in MODES)
      + ") although neutral is a smaller share of the gold labels on test (" + f"{100 * pr['dev'][1]:.1f}% on dev, {100 * pr['test'][1]:.1f}% on test).\n")
    w("### 4.4 What this does and does not explain\n")
    gaps = {m: float(np.mean([pt['test'][(m, s)]['macro_f1'] for s in SEEDS]) - np.mean([pt['dev'][(m, s)]['macro_f1'] for s in SEEDS])) for m in MODES}
    shifts = {m: R["rew_mode"][m][0] - float(np.mean([pt['test'][(m, s)]['macro_f1'] for s in SEEDS])) for m in MODES}
    w(f"* Re-weighting the test set to the dev class mix changes test macro F1 by {min(shifts.values()):+.4f} to {max(shifts.values()):+.4f}, against gaps of "
      f"{min(gaps.values()):+.4f} to {max(gaps.values()):+.4f}. So the class prior accounts for {'little' if max(abs(v) for v in shifts.values()) < 0.25 * min(gaps.values()) else 'part'} of the gap (see the table).\n"
      "* Length and code mixing are nearly the same in both splits and OOV is higher on test (section 4.3); none of these accounts for the higher test scores. "
      "The differences listed above describe where the gain appears (fewer polarity flips and fewer neutral-related errors) but not why.\n"
      "* **Not tested, and therefore unexplained:** whether the test tweets are intrinsically easier (annotation consistency, topic or time of collection), and whether the gap is specific to this model family. "
      "Checking the second would need test predictions from another model, which this analysis does not create.\n"
      "* Nothing in this section was used to change, select or re-run any model.\n")

    w("## 5. Limitations\n")
    w("* Three seeds per mode: standard deviations are rough, and the bootstrap does not include training-seed uncertainty.\n"
      "* One fixed test sample of 3,000 tweets; intervals are conditional on these nine trained models.\n"
      "* Three pairwise comparisons were made on one data set; only the Bonferroni intervals address multiplicity, and no hypothesis test is performed.\n"
      "* Dev scores are optimistically selected (best epoch by dev weighted F1); test scores are the unbiased estimates.\n"
      "* The test set was described by the team's EDA before the evaluation (protocol section 8). This analysis is exploratory and post hoc.\n"
      "* The dev–test gap remains partly unexplained (section 4.4).\n"
      "* Results were verified on CPU only; a GPU can differ in the last digits.\n"
      "* No claim is made that one preprocessing mode is better; the differences should be read against both the seed variation and the intervals above.\n")

    w("## 6. Reproducibility\n")
    w("* **Environment actually used for the evaluation, this analysis and the tests:** Python 3.12.5, torch 2.15.0+cpu, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, matplotlib 3.11.2, pytest 9.2.0 "
      "(listed in `requirements-bilstm.txt`; other combinations were not tested). The checkpoints record torch 2.15.0+cpu for training; the other library versions of the training runs were not recorded.\n"
      "* **Hardware:** CPU only. Reproducibility was checked on CPU; a GPU can differ in the last digits.\n"
      "* **Frozen artifacts:** the nine checkpoints (git-ignored, MD5 in `configs/bilstm_frozen_checkpoints.json`), the saved test predictions and the final-evaluation outputs. "
      "This analysis verifies all of them against their recorded hashes before it starts and again when it ends, and stops if one differs.\n"
      "* **Commands** (repository root, in the environment above; the official data in `data/raw/`):\n\n"
      "```\npython -m pytest tests/test_analyze_bilstm_test.py tests/test_evaluate_bilstm_test.py\n"
      "python -m src.analyze_bilstm_test --data-root data/raw      # needs the checkpoints; refuses to overwrite an existing results/bilstm_test_analysis\n```\n\n"
      "* The analysis uses the saved predictions, the official gold labels, the checkpoints' vocabularies and `results/cmi_per_tweet.csv`. It reads, and never changes, the frozen files, and trains and runs no model.\n"
      f"* Bootstrap: {R['B']:,} resamples, seed {BOOT_SEED}; identical inputs give identical numbers.\n")
    w("## 7. Submission checklist\n")
    w("- [ ] Nine checkpoints unchanged (hash check passes in `src.analyze_bilstm_test` and `src.evaluate_bilstm_test --validate-only`)\n"
      "- [ ] `results/bilstm_final_test/` and `configs/bilstm_frozen_checkpoints.json` unmodified since the evaluation\n"
      "- [ ] Test split used once for the pre-specified evaluation; this analysis is labelled exploratory\n"
      "- [ ] Test-set access by the EDA scripts disclosed (protocol section 8)\n"
      "- [ ] Results reported as mean ± sample sd over three seeds, with the bootstrap intervals; no winner claimed beyond what section 2 supports\n"
      "- [ ] Dev–test gap stated as partly unexplained (section 4.4)\n"
      "- [ ] README: data location, install steps and commands work from a clean clone (README is updated after the team's pending PRs merge)\n"
      "- [ ] No keys, tokens or personal data committed; checkpoints and raw data not committed unless the course requires it\n"
      "- [ ] One GitHub link submitted for the team, pointing to the default branch with all parts merged\n")
    return "\n".join(L)


def write_tables(R: dict, d: Path) -> None:
    atomic_write(d / "per_run_metrics.csv", csv_text(
        ["split", "mode", "seed", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"],
        [[c, m, s] + [f"{R['point'][c][(m, s)][k]:.6f}" for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1")]
         for c in CASES for m, s in RUNS]))
    atomic_write(d / "seed_variation.csv", csv_text(["split", "mode", "metric", "seed42", "seed43", "seed44", "mean", "sd_ddof1", "range"], R["seed_rows"]))
    atomic_write(d / "per_class_metrics.csv", csv_text(["split", "mode", "class", "metric", "mean_over_seeds", "sd_over_seeds", "support"], R["class_rows"]))
    atomic_write(d / "error_breakdown_per_run.csv", csv_text(R["err_header"], R["err_rows"]))
    rows = []
    for c in CASES:
        for met in ("macro_f1", "weighted_f1"):
            for m in MODES:
                e = R["boot_tab"][c][met]["modes"][m]
                rows.append([c, met, "mode", m, "", f"{e['point']:.6f}", f"{e['ci95'][0]:.6f}", f"{e['ci95'][1]:.6f}", "", ""])
            for p in R["boot_tab"][c][met]["pairs"]:
                rows.append([c, met, "difference", p["a"], p["b"], f"{p['point']:.6f}", f"{p['ci95'][0]:.6f}", f"{p['ci95'][1]:.6f}",
                             f"{p['ci95_bonferroni3'][0]:.6f}", f"{p['ci95_bonferroni3'][1]:.6f}"])
    atomic_write(d / "bootstrap_intervals.csv", csv_text(
        ["split", "metric", "kind", "a_or_mode", "b", "point_estimate", "ci95_low", "ci95_high", "bonferroni3_low", "bonferroni3_high"], rows))
    atomic_write(d / "bucket_metrics.csv", csv_text(
        ["split", "by", "mode", "bucket", "n", "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd"],
        [[r["split"], r["by"], r["mode"], r["bucket"], r["n"], f"{r['accuracy_mean']:.6f}", f"{r['accuracy_sd']:.6f}", f"{r['macro_f1_mean']:.6f}", f"{r['macro_f1_sd']:.6f}"] for r in R["bucket_rows"]]))
    atomic_write(d / "oov_statistics.csv", csv_text(
        ["mode", "split", "vocab_size", "tokens", "oov_token_rate", "types", "oov_type_rate", "tweets_with_oov", "oov_per_tweet"],
        [[o["mode"], o["split"], o["vocab_size"], o["tokens"], f"{o['oov_token_rate']:.6f}", o["types"], f"{o['oov_type_rate']:.6f}", f"{o['tweets_with_oov']:.6f}", f"{o['oov_per_tweet']:.6f}"] for o in R["oov"]]))
    atomic_write(d / "masked_error_examples.csv", csv_text(["example", "gold", "predicted_all_nine", "error_kind", "cmi_bucket", "tweet_masked"], R["ex_rows"]))
    atomic_write_json(d / "confusion_matrices_summed_over_seeds.json",
                      {c: {m: R["conf_sum"][(c, m)].tolist() for m in MODES} for c in CASES} | {"label_order": list(LABELS), "rows": "true", "columns": "predicted"})
    atomic_write_json(d / "gap_investigation.json", {
        "class_proportions": {k: [float(x) for x in v] for k, v in R["props"].items()},
        "predicted_proportions": {f"{c}:{m}": [float(x) for x in v] for (c, m), v in R["pred_props"].items()},
        "test_f1_reweighted_to_dev_class_mix_mean_over_seeds": {m: {"macro_f1": v[0], "weighted_f1": v[1]} for m, v in R["rew_mode"].items()},
        "length_stats": R["length_stats"], "cmi_stats": R["cmi_stats"], "checks": R["checks"]})


def make_figures(R: dict, d: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for r, c in enumerate(("dev", "test")):
        for k, m in enumerate(MODES):
            cm = R["conf_sum"][(c, m)].astype(float)
            norm = cm / cm.sum(1, keepdims=True)
            ax = axes[r, k]
            ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{int(cm[i, j])}\n({100 * norm[i, j]:.0f}%)", ha="center", va="center", fontsize=9,
                            color="white" if norm[i, j] > 0.55 else "black")
            ax.set_xticks(range(3), [x[:3] for x in LABELS])
            ax.set_yticks(range(3), [x[:3] for x in LABELS])
            ax.set_title(f"{c} - {m}", fontsize=10)
            if k == 0:
                ax.set_ylabel("true")
            if r == 1:
                ax.set_xlabel("predicted")
    fig.suptitle("Confusion matrices summed over seeds 42-44 (row-normalised %)")
    fig.tight_layout()
    fig.savefig(d / "confusion_matrices_dev_test.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    items = [(f"{p['a']} − {p['b']}", p) for p in R["boot_tab"]["test"]["macro_f1"]["pairs"]]
    for y, (name, p) in enumerate(items):
        ax.plot(p["ci95"], [y, y], color="#1f77b4", lw=3, solid_capstyle="butt")
        ax.plot(p["ci95_bonferroni3"], [y - 0.18, y - 0.18], color="#9aa5b1", lw=2, solid_capstyle="butt")
        ax.plot(p["point"], y, "o", color="black")
    ax.axvline(0, color="#444", ls="--", lw=1)
    ax.set_yticks(range(len(items)), [n for n, _ in items])
    ax.set_xlabel("test macro F1 difference (mean over seeds); blue 95%, grey Bonferroni 98.3%")
    ax.set_title("Paired bootstrap intervals (test-sampling uncertainty only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "bootstrap_differences_test.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    default_root = ROOT / "data" / "raw"
    ap.add_argument("--data-root", default=str(default_root if (default_root / FILES["train"]).exists() else DEFAULT_ROOT))
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--boot", type=int, default=BOOT_B, help="bootstrap resamples (default 10000)")
    ap.add_argument("--report", default="bilstm_final_analysis.md", help="report file name inside the output folder")
    a = ap.parse_args(argv)
    out_dir = (ROOT / a.out_dir) if not Path(a.out_dir).is_absolute() else Path(a.out_dir)
    data_root = Path(a.data_root)
    staging = out_dir.with_name(out_dir.name + ".staging")
    try:
        R = build(data_root, out_dir, a.boot)
        staging.mkdir(parents=True)
        write_tables(R, staging)
        make_figures(R, staging / "figures")
        atomic_write(staging / a.report, render_report(R))
        print("hashing frozen inputs (after) ...")
        after = hash_all(R["paths"])
        if after != R["before"]:
            raise ProtocolError("a frozen input changed while the analysis ran: " + ", ".join(k for k in after if after[k] != R["before"][k]))
        outputs = {str(p.relative_to(staging)).replace("\\", "/"): md5_file(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        atomic_write_json(staging / "run_manifest.json", {
            "utc": utc_now(), "purpose": "descriptive, exploratory analysis of the nine frozen runs; no model was run, nothing was selected",
            "bootstrap": {"resamples": a.boot, "seed": BOOT_SEED}, "frozen_inputs_md5_before_equal_after": True, "frozen_inputs_md5": after,
            "outputs_md5": outputs, "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                                                       **{k: __import__(k).__version__ for k in ("numpy", "pandas", "sklearn", "torch", "matplotlib")}}})
        staging.rename(out_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
