"""Issue #10: one table with every model's dev and test results, built from the result files already in the repository.

    python -m src.final_comparison          # writes results/final_model_comparison.csv

One row per system and split. Dev metrics are recomputed from the saved dev prediction files with the same metric code as the
test evaluations (src.eval_protocol_utils.full_metrics, cross-checked against src.metrics); test metrics are read from the
frozen-evaluation outputs (results/bilstm_final_test, results/transformer_final_test). Nothing is trained or evaluated here, and
the test split is never read: a system without a finished test evaluation gets a row whose status says so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval_protocol_utils import SCALARS, full_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = "final_model_comparison.csv"
COLUMNS = (["system", "family", "variant", "split", "status", "n_runs"] + [f"{k}_mean" for k in SCALARS] + [f"{k}_sd" for k in SCALARS]
           + ["source", "notes"])


def _from_prediction_files(files) -> dict:
    """Mean and sample sd (ddof=1, only for more than one run) of every metric over the given uid,label,pred files."""
    ms = []
    for f in files:
        df = pd.read_csv(f)
        ms.append(full_metrics(df["label"], df["pred"]))
    out = {f"{k}_mean": float(np.mean([m[k] for m in ms])) for k in SCALARS}
    if len(ms) > 1:
        out |= {f"{k}_sd": float(np.std([m[k] for m in ms], ddof=1)) for k in SCALARS}
    return out | {"n_runs": len(ms)}


def _from_summary(summary: dict, key: str) -> dict:
    s = summary[key]
    return ({f"{k}_mean": s[k]["mean"] for k in SCALARS} | {f"{k}_sd": s[k]["std"] for k in SCALARS} | {"n_runs": len(s[SCALARS[0]]["values"])})


def build_rows(results: Path) -> list:
    rows = []
    pred = results / "predictions"

    def add(system, family, variant, split, status, metrics=None, source="", notes=""):
        rows.append({"system": system, "family": family, "variant": variant, "split": split, "status": status, **(metrics or {}),
                     "source": source, "notes": notes})

    # ---- baselines (dev only; TF-IDF was not run on test)
    tf = results / "tfidf_dev_scores.json"
    if tf.exists():
        d = json.loads(tf.read_text(encoding="utf-8"))
        if "majority_baseline" in d:
            mb = d["majority_baseline"]
            add("majority class", "baseline", mb.get("label", ""), "dev", "done",
                {"accuracy_mean": mb["accuracy"], "weighted_f1_mean": mb["weighted_f1"], "n_runs": 1}, tf.name, "predicts one class for every tweet")
        for name in (k for k in d if k != "majority_baseline" and isinstance(d[k], dict)):
            f = pred / f"tfidf_{name.replace('+', '_')}_dev.csv"  # word+char is saved as word_char
            if f.exists():
                add(f"TF-IDF {name} + logistic regression", "baseline", name, "dev", "done", _from_prediction_files([f]),
                    f"predictions/{f.name}", "C chosen on dev; single run")
        add("TF-IDF + logistic regression", "baseline", "all", "test", "not run", None, "",
            "not evaluated on test")

    # ---- BiLSTM (dev: 3 seeds per mode; test: frozen evaluation)
    bl_test = results / "bilstm_final_test" / "metrics_summary.json"
    bl_summary = json.loads(bl_test.read_text(encoding="utf-8")) if bl_test.exists() else None
    for mode in ("raw", "clean", "normalized"):
        files = sorted(pred.glob(f"bilstm_full_{mode}_s*_seed*_dev.csv"))
        if files:
            add("BiLSTM word+char", "BiLSTM", mode, "dev", "done", _from_prediction_files(files), f"predictions/bilstm_full_{mode}_s*_seed*_dev.csv",
                "best epoch on dev; mean and sd over seeds 42, 43, 44")
        if bl_summary and mode in bl_summary:
            add("BiLSTM word+char", "BiLSTM", mode, "test", "done", _from_summary(bl_summary, mode), "bilstm_final_test/metrics_summary.json",
                "frozen single test evaluation (docs/bilstm_final_evaluation_protocol.md)")
        elif files:
            add("BiLSTM word+char", "BiLSTM", mode, "test", "pending", None, "", "")

    # ---- fine-tuned transformers (issue #6 dev runs; test: frozen evaluation)
    comp = results / "transformer_comparison.csv"
    tr_test = results / "transformer_final_test" / "metrics_summary.json"
    tr_summary = json.loads(tr_test.read_text(encoding="utf-8")) if tr_test.exists() else None
    if comp.exists():
        c = pd.read_csv(comp)
        for model, label in (("mbert", "mBERT (bert-base-multilingual-cased)"), ("xlmr", "XLM-R (xlm-roberta-base)")):
            seeds = c[(c.row_type == "seed") & (c.model == model)]
            if seeds.empty:
                continue
            lr = float(seeds.lr.iloc[0])
            run_names = [n.split(";")[0].removeprefix("run=") for n in seeds.notes]
            files = [pred / f"transformer_{r}_seed{s}_dev.csv" for r, s in zip(run_names, seeds.seeds.astype(int))]
            add(label, "transformer", f"clean, lr {lr:g}", "dev", "done", _from_prediction_files(files), "transformer_comparison.csv",
                "lr and best epoch chosen on dev (seed 42 used for the lr); mean and sd over seeds 42, 43, 44")
            if tr_summary and model in tr_summary:
                add(label, "transformer", f"clean, lr {lr:g}", "test", "done", _from_summary(tr_summary, model),
                    "transformer_final_test/metrics_summary.json", "frozen single test evaluation (docs/transformer_final_evaluation_protocol.md)")
            else:
                add(label, "transformer", f"clean, lr {lr:g}", "test", "pending", None, "",
                    "protocol frozen (docs/transformer_final_evaluation_protocol.md); run after team approval")

    # ---- Indic transformers (issue #7; separate training loop src/indian_transformer.py, raw text, 3 epochs, lr 2e-5)
    indic = [("MuRIL (google/muril-base-cased)", "muril", [results / "muril_dev_predictions.csv",
                                                         *sorted(pred.glob("muril_s*_seed*_dev.csv"))]),
             ("IndicBERT v2 (ai4bharat/IndicBERTv2-MLM-only)", "indicbert", [results / "indicbert_dev_predictions.csv"])]
    for label, key, files in indic:
        files = [f for f in files if f.exists()]
        if not files:
            add(label, "transformer", "", "dev", "pending", None, "", "issue #7")
            continue
        add(label, "transformer", "raw, lr 2e-05", "dev", "done", _from_prediction_files(files),
            ", ".join(f.relative_to(results).as_posix() for f in files),
            f"issue #7; src/indian_transformer.py (raw text, 3 epochs, not the shared pipeline); {len(files)} seed(s); best epoch on dev")
        add(label, "transformer", "raw, lr 2e-05", "test", "not run", None, "", "no frozen test evaluation for this model")

    # ---- zero-shot LLM (issue #8): a 200-tweet dev subset, so NOT comparable with the full-dev rows
    zs, meta = results / "zero_shot_dev_predictions.csv", results / "zero_shot_run_meta.json"
    if zs.exists():
        mt = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        n = len(pd.read_csv(zs))
        add(f"Zero-shot LLM ({mt.get('model', '?')})", "LLM", f"prompt {mt.get('prompt_version', '?')}", f"dev subset ({n} tweets)", "done",
            _from_prediction_files([zs]), zs.name,
            f"issue #8; {mt.get('subset', 'dev subset')}; scores are not comparable with full-dev rows; "
            f"invalid answers {mt.get('metrics', {}).get('n_invalid', '?')} (mapped to neutral)")
    else:
        add("Zero-shot LLM", "LLM", "", "dev", "pending", None, "", "issue #8")
    add("Zero-shot LLM", "LLM", "", "test", "not run", None, "", "")
    return rows


def build(results: Path = ROOT / "results", write: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(build_rows(Path(results))).reindex(columns=COLUMNS)
    num = [c for c in COLUMNS if c.endswith(("_mean", "_sd"))]
    df[num] = df[num].astype(float).round(4)
    if write:
        df.to_csv(Path(results) / OUT_NAME, index=False)
    return df


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description="Collect every model's dev and test results into one table (issue #10).")
    p.add_argument("--results", default=str(ROOT / "results"))
    a = p.parse_args(argv)
    df = build(Path(a.results))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[["system", "variant", "split", "status", "n_runs", "macro_f1_mean", "weighted_f1_mean", "weighted_f1_sd"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
