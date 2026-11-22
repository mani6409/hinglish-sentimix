"""Fine-tune a multilingual transformer (mBERT, XLM-R, MuRIL, IndicBERT, ...) on the SentiMix Hinglish data.

One script for every checkpoint: the model is a setting, not a code path. Run from the repo root
(see docs/transformer_pipeline.md):

    python -m src.train_transformer --model mbert --seeds 42 43 44
    python -m src.train_transformer --model xlmr  --seeds 42 43 44 --set lr=1e-5
    python -m src.train_transformer --config configs/transformer_config.json --model muril
    python -m src.train_transformer --model mbert --limit 200 --set epochs=1     # quick debug, nothing saved

Rules this script follows (the same as src.train_bilstm)
- Train on the official TRAIN split; select the best epoch and report on the official DEV split.
- max_length is derived from TRAIN subword lengths only (src.transformer_utils.derive_max_length), per tokenizer.
- The TEST split is never loaded here. The final one-time test run is a separate step.
- Metrics come from src.metrics.compute_metrics (weighted F1 is the official score).
- Existing results are never overwritten by accident: the run stops before training if any output exists,
  writes to a staging folder and publishes only after all seeds finished (helpers shared with src.train_bilstm).

Outputs (under results/ unless --out-dir is given):
  predictions/transformer_<run>_seed<N>_dev.csv   uid,label,pred
  transformer_<run>_dev_scores.json               config, max_length, per-seed metrics and history, mean/std
  experiment_log.csv                              one appended row per seed
  checkpoints/transformer_<run>_seed<N>/          best-dev-epoch model + tokenizer + run_info.json (git-ignored)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.metrics import compute_metrics, confusion
from src.preprocess import PRESETS, load_processed_split
from src.seed import DEFAULT_SEED, set_seed
from src.train_bilstm import LOG_NAME, _parse_set, _publish, _stamp, log_has_run
from src.transformer_utils import (
    EncodedTweets,
    TransformerConfig,
    count_truncated,
    derive_max_length,
    ids_to_labels,
    load_model,
    load_tokenizer,
    make_collate,
    model_position_limit,
    model_slug,
    prediction_frame,
    predict_ids,
    resolve_device,
    resolve_model_name,
    save_checkpoint_dir,
    token_lengths,
)

ROOT = Path(__file__).resolve().parents[1]
TEXT_COL = "text_proc"  # column added by src.preprocess (the text the model reads)
DATA_VERSIONS = {
    "raw": "raw (tokens joined by spaces)",
    "clean": "clean (src.preprocess: URLs, mentions, hashtags, contractions)",
    "normalized": "normalized (src.preprocess: clean + repeated letters + Hinglish spellings)",
}


# ------------------------------------------------------------ paths ----
def prediction_path(out_dir, run_name: str, seed: int) -> Path:
    return Path(out_dir) / "predictions" / f"transformer_{run_name}_seed{seed}_dev.csv"


def scores_path(out_dir, run_name: str) -> Path:
    return Path(out_dir) / f"transformer_{run_name}_dev_scores.json"


def checkpoint_dir(out_dir, run_name: str, seed: int) -> Path:
    return Path(out_dir) / "checkpoints" / f"transformer_{run_name}_seed{seed}"


def default_run_name(cfg: TransformerConfig) -> str:
    return f"{model_slug(cfg.model_name)}_{cfg.mode}"


def find_existing_outputs(out_dir, run_name: str, seeds: List[int], save_checkpoints: bool = True) -> List[str]:
    out = Path(out_dir)
    planned = [scores_path(out, run_name)] + [prediction_path(out, run_name, s) for s in seeds]
    if save_checkpoints:
        planned += [checkpoint_dir(out, run_name, s) for s in seeds]
    found = [str(p) for p in planned if p.exists()]
    log = out / LOG_NAME
    found += [f"{log} (row of run '{run_name}', seed {s})" for s in seeds if log_has_run(log, run_name, s)]
    return found


# ------------------------------------------------------- max length ----
def resolve_max_length(tokenizer, train_texts: List[str], cfg: TransformerConfig) -> dict:
    """The max_length of a run: the config value if set, otherwise derived from TRAIN subword lengths."""
    limit = model_position_limit(tokenizer)
    lengths = token_lengths(tokenizer, train_texts)
    if cfg.max_length is not None:
        value, source = min(cfg.max_length, limit), "config"
    else:
        value = derive_max_length(lengths, cfg.length_percentile, model_limit=limit)
        source = f"train p{cfg.length_percentile:g} subword length, rounded up to a multiple of 8"
    return {"max_length": value, "source": source, "train_max": int(max(lengths)), "model_limit": limit,
            "train_truncated": count_truncated(lengths, value)}


# ------------------------------------------------------- one training ----
def _evaluate(model, loader, device, fp16, true_labels):
    pred = ids_to_labels(predict_ids(model, loader, device, fp16))
    return pred, compute_metrics(true_labels, pred)


def train_one_seed(train_df: pd.DataFrame, dev_df: pd.DataFrame, cfg: TransformerConfig, seed: int,
                   verbose: bool = True, tokenizer=None, length_info: Optional[dict] = None,
                   checkpoint_path: Optional[Path] = None) -> dict:
    """Fine-tune once with `seed` and return the best-dev-epoch result.

    Needs columns `text_proc`, `label` (and `uid` for saving predictions) in both frames.
    The epoch with the best dev weighted F1 is kept (model selection on dev); that dev score is therefore
    slightly optimistic, the unbiased number comes from the one-time test run.
    """
    cfg.validate()
    t0 = time.time()
    set_seed(seed)
    device = resolve_device(cfg.device)
    use_amp = cfg.fp16 and device.type == "cuda"

    tokenizer = tokenizer or load_tokenizer(cfg.model_name)
    train_texts, dev_texts = list(train_df[TEXT_COL]), list(dev_df[TEXT_COL])
    dev_labels = list(dev_df["label"])
    length_info = length_info or resolve_max_length(tokenizer, train_texts, cfg)
    max_len = length_info["max_length"]
    dev_trunc = count_truncated(token_lengths(tokenizer, dev_texts), max_len)

    train_ds = EncodedTweets(tokenizer, train_texts, list(train_df["label"]), max_len)
    dev_ds = EncodedTweets(tokenizer, dev_texts, dev_labels, max_len)
    collate = make_collate(tokenizer)
    gen = torch.Generator()
    gen.manual_seed(seed)  # repeatable shuffling
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, generator=gen)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate)

    set_seed(seed)  # the classifier head is initialised from the seed, independent of tokenisation work above
    model = load_model(cfg.model_name, cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    steps_per_epoch = max(1, -(-len(train_loader) // cfg.grad_accum_steps))
    total_steps = steps_per_epoch * cfg.epochs
    from transformers import get_linear_schedule_with_warmup

    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"f1": -1.0, "epoch": 0, "state": None}
    history, bad_epochs = [], 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            k = len(batch["labels"])
            total += loss.item() * k
            n += k
        _, m = _evaluate(model, dev_loader, device, cfg.fp16, dev_labels)
        history.append({"epoch": epoch, "train_loss": round(total / n, 4), "dev_weighted_f1": round(m["weighted_f1"], 4),
                        "dev_macro_f1": round(m["macro_f1"], 4), "dev_accuracy": round(m["accuracy"], 4),
                        "lr_end": float(sched.get_last_lr()[0])})
        if verbose:
            h = history[-1]
            print(f"  seed {seed} epoch {epoch:2d}  loss {h['train_loss']:.4f}  dev wF1 {h['dev_weighted_f1']:.4f}"
                  f"  macroF1 {h['dev_macro_f1']:.4f}  acc {h['dev_accuracy']:.4f}  ({time.time() - t0:.0f}s)")
        if m["weighted_f1"] > best["f1"]:
            best = {"f1": m["weighted_f1"], "epoch": epoch,
                    "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best["state"])  # back to the best epoch
    del best["state"]
    pred, metrics = _evaluate(model, dev_loader, device, cfg.fp16, dev_labels)
    saved = None
    if checkpoint_path is not None:  # only train-fitted weights, settings and dev metrics are stored
        saved = str(save_checkpoint_dir(checkpoint_path, model, tokenizer, {
            "config": asdict(cfg), "checkpoint": resolve_model_name(cfg.model_name), "seed": seed,
            "best_epoch": best["epoch"], "selected_on": "dev weighted_f1", "dev_metrics": metrics,
            "max_length": max_len, "length_source": length_info["source"],
            "preprocess_options": dict(PRESETS[cfg.mode]), "train_size": len(train_df), "dev_size": len(dev_df),
            "torch_version": str(torch.__version__), "created": date.today().isoformat()}))
    result = {
        "seed": seed, "best_epoch": best["epoch"], "epochs_run": len(history), "metrics": metrics,
        "confusion": confusion(dev_labels, pred), "history": history, "pred": pred, "n_params": n_params,
        "max_length": max_len, "truncation": {"train": length_info["train_truncated"], "dev": dev_trunc},
        "device": str(device), "fp16": use_amp, "checkpoint": saved, "seconds": round(time.time() - t0, 1),
    }
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------- experiment + logging ----
def run_experiment(train_df, dev_df, cfg: TransformerConfig, seeds: List[int], out_dir: Optional[Path] = None,
                   run_name: Optional[str] = None, who: str = "Monis", verbose: bool = True,
                   save_checkpoints: bool = True, overwrite: bool = False) -> dict:
    """Run `train_one_seed` for every seed, summarise, and (if out_dir is given) save everything.

    With out_dir=None nothing is written (tests and debugging). Output protection is the same as for the
    BiLSTM (src.train_bilstm.run_experiment): stop before training if outputs exist, stage, publish at the end,
    move old files to <out_dir>/archive/<timestamp>/ with overwrite=True, keep failed work in failed_runs/.
    """
    cfg.validate()
    run_name = run_name or default_run_name(cfg)
    out = Path(out_dir) if out_dir is not None else None
    existing = find_existing_outputs(out, run_name, seeds, save_checkpoints) if out is not None else []
    if existing and not overwrite:
        raise FileExistsError(
            f"outputs of run '{run_name}' already exist, nothing was trained or written:\n  - " + "\n  - ".join(existing)
            + "\nUse another --run-name, or --overwrite to replace them "
              "(old files are then moved to <out-dir>/archive/<timestamp>/, never deleted).")

    tokenizer = load_tokenizer(cfg.model_name)
    length_info = resolve_max_length(tokenizer, list(train_df[TEXT_COL]), cfg)
    if verbose:
        print(f"[{run_name}] {resolve_model_name(cfg.model_name)}  max_length={length_info['max_length']} "
              f"({length_info['source']}; longest train tweet {length_info['train_max']} subwords)")

    stamp = _stamp()
    stage = out / f".staging_{run_name}_{stamp}" if out is not None else None
    per_seed, log_rows, archived = [], [], []
    try:
        if stage is not None:
            (stage / "predictions").mkdir(parents=True)
        for seed in seeds:
            if verbose:
                print(f"[{run_name}] seed {seed}")
            res = train_one_seed(train_df, dev_df, cfg, seed, verbose=verbose, tokenizer=tokenizer,
                                 length_info=length_info,
                                 checkpoint_path=checkpoint_dir(stage, run_name, seed)
                                 if stage is not None and save_checkpoints else None)
            if res.get("checkpoint"):
                res["checkpoint"] = str(checkpoint_dir(out, run_name, seed))  # location after publishing
            per_seed.append(res)
            m = res["metrics"]
            if stage is not None:
                prediction_frame(dev_df["uid"].values, dev_df["label"].values, res["pred"]).to_csv(
                    prediction_path(stage, run_name, seed), index=False)
                log_rows.append(
                    [date.today().isoformat(), who, f"transformer {resolve_model_name(cfg.model_name)}",
                     DATA_VERSIONS[cfg.mode], cfg.short(), seed, f"{m['weighted_f1']:.4f}",
                     f"run={run_name}; " + ("rerun=yes; " if existing else "") +
                     f"macro_f1={m['macro_f1']:.4f}; accuracy={m['accuracy']:.4f}; best_epoch={res['best_epoch']}; "
                     f"max_length={res['max_length']}; dev_truncated_tweets={res['truncation']['dev']['tweets']}; "
                     f"params={res['n_params']}; device={res['device']}; best epoch selected on dev",
                     ""])
            if verbose:
                print(f"[{run_name}] seed {seed} best epoch {res['best_epoch']}: wF1 {m['weighted_f1']:.4f} "
                      f"macroF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}  ({res['seconds']}s)")

        summary = {}
        for key in ("weighted_f1", "macro_f1", "accuracy"):
            vals = np.array([r["metrics"][key] for r in per_seed])
            summary[key] = {"mean": round(float(vals.mean()), 4),
                            "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)}
        result = {"run_name": run_name, "checkpoint": resolve_model_name(cfg.model_name), "config": asdict(cfg),
                  "length": length_info, "seeds": list(seeds), "summary": summary,
                  "per_seed": [{k: v for k, v in r.items() if k != "pred"} for r in per_seed]}
        if stage is not None:
            scores_path(stage, run_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
            archived = _publish(stage, out, log_rows, overwrite, stamp)
    except BaseException:
        if stage is not None and stage.exists():
            failed = out / "failed_runs" / f"{run_name}_{stamp}"
            failed.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, failed)
            print(f"[{run_name}] run did not finish; its partial outputs are kept in {failed}. "
                  "Existing results were not touched.")
        raise
    result["archived"] = archived
    if verbose:
        s = summary
        print(f"[{run_name}] over seeds {list(seeds)}: wF1 {s['weighted_f1']['mean']:.4f} +/- {s['weighted_f1']['std']:.4f}, "
              f"macroF1 {s['macro_f1']['mean']:.4f} +/- {s['macro_f1']['std']:.4f}, "
              f"acc {s['accuracy']['mean']:.4f} +/- {s['accuracy']['std']:.4f}")
    return result


# -------------------------------------------------------------------- CLI ----
def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Fine-tune a transformer on the train split and evaluate on dev.")
    p.add_argument("--config", help="JSON file with TransformerConfig settings (default: built-in defaults)")
    p.add_argument("--model", help="checkpoint: mbert | xlmr | muril | indicbert | any Hub id or local folder")
    p.add_argument("--mode", choices=list(PRESETS), help="text version: raw | clean | normalized")
    p.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. --set lr=3e-5")
    p.add_argument("--run-name", help="name used in output files (default: <model>_<mode>)")
    p.add_argument("--who", default="Monis", help="name written to experiment_log.csv")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--data-root", help="folder with the SentiMix Hinglish files (default: src.data.DEFAULT_ROOT)")
    p.add_argument("--limit", type=int, help="DEBUG: use only the first N train and dev tweets; nothing is saved")
    p.add_argument("--no-checkpoints", action="store_true", help="do not save the best model of each seed")
    p.add_argument("--overwrite", action="store_true", help="replace an earlier run with the same name (old files archived)")
    args = p.parse_args(argv)

    cfg = TransformerConfig.from_json(args.config) if args.config else TransformerConfig()
    overrides = _parse_set(args.set)
    if args.model:
        overrides["model_name"] = args.model
    if args.mode:
        overrides["mode"] = args.mode
    cfg = cfg.with_overrides(overrides)

    # only train and dev are ever loaded; test is deliberately not touched here
    train_df = load_processed_split("train", mode=cfg.mode, root=args.data_root)
    dev_df = load_processed_split("dev", mode=cfg.mode, root=args.data_root)
    out_dir = Path(args.out_dir)
    if args.limit:
        train_df, dev_df = train_df.head(args.limit), dev_df.head(args.limit)
        out_dir = None
        print(f"DEBUG run on {len(train_df)} train / {len(dev_df)} dev tweets: results are NOT saved or logged.")
    return run_experiment(train_df, dev_df, cfg, args.seeds, out_dir=out_dir, run_name=args.run_name, who=args.who,
                          save_checkpoints=not args.no_checkpoints, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
