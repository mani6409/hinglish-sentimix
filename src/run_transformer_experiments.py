"""Issue #6: learning-rate search and seeds for mBERT and XLM-R, run through the shared pipeline.

    python -m src.run_transformer_experiments                       # search, then final seeds
    python -m src.run_transformer_experiments --phase search
    python -m src.run_transformer_experiments --phase final --prune-checkpoints

    # the protocol actually reported in docs/transformer_results.md (6 epochs instead of the default 4)
    python -m src.run_transformer_experiments --suffix _ep6 --set epochs=6 --lrs 1e-5 2e-5 3e-5 --prune-checkpoints

--set KEY=VALUE overrides the config for EVERY run; pair it with --suffix, which is added to the run names
and the search CSV, so runs with different settings never share a name (and never get skipped by mistake).

Protocol (docs/transformer_results.md)
1. Search: every learning rate in --lrs with ONE seed (--search-seed, default 42), the same config otherwise.
   Each run is a normal `src.train_transformer` run named `<model>_<mode>_lr<lr>`.
2. Select: per model, the learning rate with the highest dev weighted F1 (ties -> the smaller lr).
   Written to results/transformer_lr_search<suffix>.csv.
3. Final: the selected learning rate with the remaining seeds (--final-seeds, default 43 44), run name
   `<model>_<mode>_lr<lr>_final`. Together with the search run of the selected lr this gives three seeds.
   Seed 42 took part in choosing the lr, so its score is slightly optimistic; the doc reports both
   the three-seed mean and the mean over the two seeds that played no part in the selection.

Only TRAIN and DEV are used. Each run is a separate process (clean GPU memory). A run whose scores JSON
already exists is skipped, so the driver can be restarted after an interruption.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from src.train_transformer import checkpoint_dir, scores_path
from src.transformer_utils import model_slug, resolve_model_name

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ["mbert", "xlmr"]
DEFAULT_LRS = [1e-5, 2e-5, 3e-5, 5e-5]
SEARCH_CSV = "transformer_lr_search.csv"  # without a suffix; see search_csv_name


def lr_tag(lr: float) -> str:
    """2e-05 -> '2e-5' (used in run names)."""
    mant, exp = f"{lr:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def search_run_name(model: str, mode: str, lr: float, suffix: str = "") -> str:
    """`suffix` keeps protocols with different settings apart, e.g. '_ep6' -> mbert_clean_ep6_lr1e-5."""
    return f"{model_slug(model)}_{mode}{suffix}_lr{lr_tag(lr)}"


def final_run_name(model: str, mode: str, lr: float, suffix: str = "") -> str:
    return search_run_name(model, mode, lr, suffix) + "_final"


def search_csv_name(suffix: str = "") -> str:
    return f"transformer_lr_search{suffix}.csv"


def run_training(model: str, lr: float, seeds: List[int], run_name: str, out_dir: Path, mode: str,
                 config: str, extra: List[str]) -> None:
    if scores_path(out_dir, run_name).exists():
        print(f"[skip] {run_name}: {scores_path(out_dir, run_name)} exists")
        return
    cmd = [sys.executable, "-m", "src.train_transformer", "--config", config, "--model", model, "--mode", mode,
           "--set", f"lr={lr}", "--seeds", *map(str, seeds), "--run-name", run_name, "--out-dir", str(out_dir),
           *extra]
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def select_lrs(models: List[str], lrs: List[float], mode: str, out_dir: Path, suffix: str = "") -> Dict[str, dict]:
    """Best lr per model from the search runs; writes results/transformer_lr_search.csv."""
    rows, best = [], {}
    for m in models:
        for lr in lrs:
            path = scores_path(out_dir, search_run_name(m, mode, lr, suffix))
            if not path.exists():
                raise FileNotFoundError(f"search run missing: {path}")
            s = json.loads(path.read_text(encoding="utf-8"))
            r = s["per_seed"][0]
            rows.append({"model": model_slug(m), "checkpoint": resolve_model_name(m), "mode": mode, "lr": lr,
                         "seed": r["seed"], "dev_weighted_f1": round(r["metrics"]["weighted_f1"], 4),
                         "dev_macro_f1": round(r["metrics"]["macro_f1"], 4),
                         "dev_accuracy": round(r["metrics"]["accuracy"], 4), "best_epoch": r["best_epoch"],
                         "epochs_run": r["epochs_run"], "run_name": s["run_name"]})
        own = [r for r in rows if r["model"] == model_slug(m)]
        top = max(own, key=lambda r: (r["dev_weighted_f1"], -r["lr"]))  # ties -> smaller lr
        best[m] = top
    for r in rows:
        r["selected"] = r is best[next(m for m in models if model_slug(m) == r["model"])]
    with open(out_dir / search_csv_name(suffix), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return best


def prune_search_checkpoints(models, lrs, mode, out_dir: Path, best: Dict[str, dict], seed: int,
                             suffix: str = "") -> List[str]:
    """Delete the checkpoints of search runs whose lr was NOT selected (they are never used again)."""
    removed = []
    for m in models:
        for lr in lrs:
            if lr == best[m]["lr"]:
                continue
            d = checkpoint_dir(out_dir, search_run_name(m, mode, lr, suffix), seed)
            if d.is_dir():
                shutil.rmtree(d)
                removed.append(str(d))
    return removed


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="LR search + seeds for mBERT and XLM-R (issue #6).")
    p.add_argument("--phase", choices=["search", "final", "all"], default="all")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    p.add_argument("--mode", default="clean")
    p.add_argument("--search-seed", type=int, default=42)
    p.add_argument("--final-seeds", type=int, nargs="+", default=[43, 44])
    p.add_argument("--config", default=str(ROOT / "configs" / "transformer_config.json"))
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--suffix", default="", help="added to run names and the search CSV, e.g. _ep6")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override for EVERY run, e.g. --set epochs=6 (use a --suffix with it)")
    p.add_argument("--prune-checkpoints", action="store_true",
                   help="after selection, delete checkpoints of the search runs whose lr was not selected")
    p.add_argument("extra", nargs="*", help="passed to src.train_transformer after `--`, e.g. -- --data-root X")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    extra = [x for kv in args.set for x in ("--set", kv)] + args.extra
    sfx = args.suffix

    if args.phase in ("search", "all"):
        for m in args.models:
            for lr in args.lrs:
                run_training(m, lr, [args.search_seed], search_run_name(m, args.mode, lr, sfx), out, args.mode,
                             args.config, extra)
    best = {}
    if args.phase in ("final", "all"):
        best = select_lrs(args.models, args.lrs, args.mode, out, sfx)
        for m, row in best.items():
            print(f"[select] {m}: lr={row['lr']} (dev wF1 {row['dev_weighted_f1']}, seed {row['seed']})")
        if args.prune_checkpoints:
            for d in prune_search_checkpoints(args.models, args.lrs, args.mode, out, best, args.search_seed, sfx):
                print(f"[prune] {d}")
        for m, row in best.items():
            run_training(m, row["lr"], args.final_seeds, final_run_name(m, args.mode, row["lr"], sfx), out, args.mode,
                         args.config, extra)
    return best


if __name__ == "__main__":
    main()
