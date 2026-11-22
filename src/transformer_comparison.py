"""Issue #6: compare the fine-tuned mBERT and XLM-R runs on the official dev split.

    python -m src.transformer_comparison                  # runs made without --suffix
    python -m src.transformer_comparison --suffix _ep6    # the reported 6-epoch runs

Reads the runs written by src.run_transformer_experiments (lr search + final seeds) and writes:

  results/transformer_comparison.csv    one `mean` row per model (mean and sd over its three seeds) plus
                                        one `seed` row per run, and reference rows for the TF-IDF and
                                        BiLSTM baselines (dev, same metric code)
  results/transformer_pairwise.csv      mBERT - XLM-R weighted-F1 difference: per shared seed, and a
                                        paired bootstrap 95% interval on the representative predictions
  results/<model>_dev_predictions.csv   the representative run of each model (the seed with the MEDIAN dev
                                        weighted F1), uid,label,pred; all seeds stay in results/predictions/

Every metric is recomputed from the prediction files with src.metrics.compute_metrics and checked against
the scores the training run reported. Only dev predictions are read; the test split is never touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.cmi_evaluation import LABEL_INDEX, N_BOOT, file_rng, resample_indices, weighted_f1
from src.metrics import LABELS, compute_metrics
from src.run_transformer_experiments import (
    DEFAULT_MODELS,
    final_run_name,
    search_csv_name,
    search_run_name,
)
from src.train_transformer import prediction_path, scores_path
from src.transformer_utils import model_slug, resolve_model_name
from src.uid_checks import check_uid_alignment

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["weighted_f1", "macro_f1", "accuracy", "precision_macro", "recall_macro", "precision_weighted",
           "recall_weighted"]


def full_metrics(y_true, y_pred) -> dict:
    """compute_metrics plus macro and weighted precision/recall (from its per-class numbers)."""
    m = compute_metrics(y_true, y_pred)
    pc = m["per_class"]
    sup = np.array([pc[c]["support"] for c in LABELS], dtype=float)
    p = np.array([pc[c]["precision"] for c in LABELS])
    r = np.array([pc[c]["recall"] for c in LABELS])
    m.update(precision_macro=float(p.mean()), recall_macro=float(r.mean()),
             precision_weighted=float((p * sup).sum() / sup.sum()), recall_weighted=float((r * sup).sum() / sup.sum()))
    return m


def selected_lrs(out_dir: Path, suffix: str = "") -> Dict[str, float]:
    """{model: learning rate} for the rows marked selected in the lr-search CSV."""
    with open(out_dir / search_csv_name(suffix), encoding="utf-8", newline="") as f:
        return {r["model"]: float(r["lr"]) for r in csv.DictReader(f) if r["selected"] == "True"}


def collect_runs(model: str, lr: float, mode: str, out_dir: Path, suffix: str = "") -> List[dict]:
    """One dict per seed of `model` at the selected `lr`: the search run plus the final run."""
    runs = []
    for name, role in ((search_run_name(model, mode, lr, suffix), "search+final"),
                       (final_run_name(model, mode, lr, suffix), "final")):
        scores = json.loads(scores_path(out_dir, name).read_text(encoding="utf-8"))
        for r in scores["per_seed"]:
            pred_file = prediction_path(out_dir, name, r["seed"])
            df = pd.read_csv(pred_file)
            m = full_metrics(df["label"], df["pred"])
            if abs(m["weighted_f1"] - r["metrics"]["weighted_f1"]) > 1e-9:
                raise ValueError(f"{pred_file}: recomputed weighted F1 differs from the training run's report")
            runs.append({"model": model_slug(model), "checkpoint": resolve_model_name(model), "lr": lr,
                         "seed": r["seed"], "run_name": name, "used_for_lr_selection": role == "search+final",
                         "best_epoch": r["best_epoch"], "pred_file": pred_file, "pred": df, "metrics": m})
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{model}: repeated seeds {seeds}")
    return runs


def median_run(runs: List[dict]) -> dict:
    """The run with the median dev weighted F1 (lower median for an even count)."""
    order = sorted(runs, key=lambda r: (r["metrics"]["weighted_f1"], r["seed"]))
    return order[(len(order) - 1) // 2]


def paired_bootstrap(y_true, pred_a, pred_b, rng, n=N_BOOT) -> np.ndarray:
    """Weighted-F1 difference a - b on `n` resamples; both models use the SAME resampled tweets."""
    t = np.asarray([LABEL_INDEX[x] for x in y_true])
    a = np.asarray([LABEL_INDEX[x] for x in pred_a])
    b = np.asarray([LABEL_INDEX[x] for x in pred_b])
    diffs = np.empty(n)
    for i in range(n):
        idx = resample_indices(len(t), rng)
        diffs[i] = weighted_f1(t[idx], a[idx]) - weighted_f1(t[idx], b[idx])
    return diffs


def baseline_rows(out_dir: Path) -> List[dict]:
    """Reference rows from the existing dev results (not retrained here)."""
    rows = []
    tf = out_dir / "tfidf_dev_scores.json"
    if tf.exists():
        d = json.loads(tf.read_text(encoding="utf-8"))
        best = max((k for k in d if isinstance(d[k], dict) and "best_dev" in d[k]),
                   key=lambda k: d[k]["best_dev"]["weighted_f1"])
        pred = out_dir / "predictions" / f"tfidf_{best}_dev.csv"
        if pred.exists():
            df = pd.read_csv(pred)
            m = full_metrics(df["label"], df["pred"])
            rows.append({"row_type": "reference", "model": f"tfidf {best} + logistic regression", "n_seeds": 1,
                         **{f"{k}_mean": round(m[k], 4) for k in METRICS}, "notes": "baseline, single run"})
    for mode in ("raw", "clean", "normalized"):
        files = sorted((out_dir / "predictions").glob(f"bilstm_full_{mode}_s*_seed*_dev.csv"))
        if not files:
            continue
        ms = [full_metrics(*(lambda df: (df["label"], df["pred"]))(pd.read_csv(f))) for f in files]
        rows.append({"row_type": "reference", "model": f"bilstm word+char ({mode})", "n_seeds": len(ms),
                     **{f"{k}_mean": round(float(np.mean([m[k] for m in ms])), 4) for k in METRICS},
                     **{f"{k}_sd": round(float(np.std([m[k] for m in ms], ddof=1)), 4) for k in METRICS},
                     "notes": "baseline, mean over seeds"})
    return rows


def build(models: List[str], mode: str, out_dir: Path, write: bool = True, suffix: str = "") -> dict:
    """Collect each model's three seeds at its selected lr, summarise them (mean, sd, representative median run)
    and compare the two models per seed and with a paired bootstrap. Returns {"rows", "pairs", "runs"}."""
    lrs = selected_lrs(out_dir, suffix)
    all_runs = {}
    rows = []
    dev_uids = None
    for m in models:
        slug = model_slug(m)
        runs = collect_runs(m, lrs[slug], mode, out_dir, suffix)
        for r in runs:
            uids = list(r["pred"]["uid"])
            if dev_uids is None:
                dev_uids = uids
            check_uid_alignment(uids, dev_uids, name=str(r["pred_file"]), reference_name="first dev prediction file")
        all_runs[slug] = runs
        vals = {k: np.array([r["metrics"][k] for r in runs]) for k in METRICS}
        held_out = [r["metrics"]["weighted_f1"] for r in runs if not r["used_for_lr_selection"]]
        rep = median_run(runs)
        rows.append({"row_type": "mean", "model": slug, "checkpoint": resolve_model_name(m), "lr": lrs[slug],
                     "seeds": " ".join(str(r["seed"]) for r in runs), "n_seeds": len(runs),
                     **{f"{k}_mean": round(float(v.mean()), 4) for k, v in vals.items()},
                     **{f"{k}_sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0 for k, v in vals.items()},
                     "weighted_f1_mean_unselected_seeds": round(float(np.mean(held_out)), 4) if held_out else None,
                     "notes": f"representative run: seed {rep['seed']} (median dev weighted F1)"})
        for r in runs:
            rows.append({"row_type": "seed", "model": slug, "checkpoint": resolve_model_name(m), "lr": lrs[slug],
                         "seeds": str(r["seed"]), "n_seeds": 1, **{f"{k}_mean": round(r["metrics"][k], 4) for k in METRICS},
                         "notes": f"run={r['run_name']}; best_epoch={r['best_epoch']}"
                                  + ("; lr chosen with this seed" if r["used_for_lr_selection"] else "")})
    rows += baseline_rows(out_dir)

    pairs = []
    if len(models) == 2:
        a, b = (model_slug(m) for m in models)
        by_seed_b = {r["seed"]: r for r in all_runs[b]}
        for ra in all_runs[a]:
            rb = by_seed_b.get(ra["seed"])
            if rb:
                pairs.append({"comparison": f"{a} - {b}", "basis": f"seed {ra['seed']}",
                              "difference": round(ra["metrics"]["weighted_f1"] - rb["metrics"]["weighted_f1"], 4),
                              "ci_low": None, "ci_high": None, "interval_includes_zero": None})
        ra, rb = median_run(all_runs[a]), median_run(all_runs[b])
        merged = ra["pred"].merge(rb["pred"], on="uid", suffixes=("_a", "_b"), validate="one_to_one")
        if not (merged["label_a"] == merged["label_b"]).all():
            raise ValueError("gold labels differ between the two prediction files")
        diffs = paired_bootstrap(merged["label_a"], merged["pred_a"], merged["pred_b"], file_rng(f"{a}-{b}"))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pairs.append({"comparison": f"{a} - {b}",
                      "basis": f"paired bootstrap ({N_BOOT}), representative runs seed {ra['seed']} vs seed {rb['seed']}",
                      "difference": round(ra["metrics"]["weighted_f1"] - rb["metrics"]["weighted_f1"], 4),
                      "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                      "interval_includes_zero": bool(lo <= 0 <= hi)})

    if write:
        cols = ["row_type", "model", "checkpoint", "lr", "seeds", "n_seeds"] + [f"{k}_mean" for k in METRICS] + \
               [f"{k}_sd" for k in METRICS] + ["weighted_f1_mean_unselected_seeds", "notes"]
        pd.DataFrame(rows).reindex(columns=cols).to_csv(out_dir / "transformer_comparison.csv", index=False)
        if pairs:
            pd.DataFrame(pairs).to_csv(out_dir / "transformer_pairwise.csv", index=False)
        for slug, runs in all_runs.items():
            shutil.copyfile(median_run(runs)["pred_file"], out_dir / f"{slug}_dev_predictions.csv")
    return {"rows": rows, "pairs": pairs, "runs": all_runs}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="mBERT vs XLM-R comparison on dev (issue #6).")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--mode", default="clean")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--suffix", default="", help="the --suffix used with src.run_transformer_experiments")
    args = p.parse_args(argv)
    res = build(args.models, args.mode, Path(args.out_dir), suffix=args.suffix)
    for r in res["rows"]:
        if r["row_type"] != "seed":
            print(f"{r['model']:40s} wF1 {r['weighted_f1_mean']:.4f}"
                  + (f" +/- {r['weighted_f1_sd']:.4f}" if r.get("weighted_f1_sd") is not None else "")
                  + f"  macroF1 {r['macro_f1_mean']:.4f}  acc {r['accuracy_mean']:.4f}")
    for p_ in res["pairs"]:
        print(p_)
    return res


if __name__ == "__main__":
    main()
