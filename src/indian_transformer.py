"""Fine-tune an Indian-language pretrained Transformer (MuRIL, IndicBERT) for Hinglish sentiment (issue #7).

This is the shared code behind src/train_muril.py and src/train_indicbert.py, so the two models use identical
training logic and differ only in the checkpoint name.

NOTE ON ISSUE #5: the issue asks to reuse the shared Transformer pipeline (src/train_transformer.py). That file
is not on `main` yet. This module therefore contains a small self-contained loop built on the same shared project
pieces (src.data, src.metrics, src.seed, src.preprocess, the `uid,label,pred` CSV format). When the shared pipeline
lands, `fine_tune` is the one function to replace by a call to it.
(Later update: src/train_transformer.py is now in the repository, but the reported MuRIL and IndicBERT dev results
were produced with the loop in this file, so it has been kept.)

Rules followed (docs/daily_plan.md): the TEST set is never loaded here; model selection (best epoch) uses the dev
set only; 3-class labels in src.metrics.LABELS order; weighted F1 from src.metrics is the reported score.
Model weights go to checkpoints/ (ignored by Git).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import LABELS, compute_metrics
from src.seed import DEFAULT_SEED, set_seed

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

DEFAULTS = dict(max_len=96, batch_size=32, lr=2e-5, epochs=3, warmup_ratio=0.1, weight_decay=0.01)


# ---- data ------------------------------------------------------------------------------------------------
def load_texts(split, text_mode="raw"):
    """DataFrame (uid, label, text_proc) for 'train' or 'dev'. The test split is refused on purpose."""
    if split not in ("train", "dev"):
        raise ValueError("only 'train' and 'dev' may be used here; the test set is run once, at the very end")
    from src.preprocess import load_processed_split

    return load_processed_split(split, mode=text_mode)[["uid", "label", "text_proc"]]


def token_length_report(tokenizer, texts, percentiles=(50, 90, 95, 99, 100)):
    """Subword lengths (without padding) and the share of tokens that are the unknown token."""
    enc = tokenizer(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
    lengths = np.array([len(x) for x in enc])
    unk = tokenizer.unk_token_id
    n_unk = sum(int(t == unk) for ids in enc for t in ids) if unk is not None else 0
    return {
        "n_texts": int(len(lengths)),
        "length_percentiles": {f"p{p}": int(np.percentile(lengths, p)) for p in percentiles},
        "unk_token_share": round(n_unk / max(int(lengths.sum()), 1), 5),
        "mean_subwords_per_whitespace_word": round(
            float(lengths.sum() - 2 * len(lengths)) / max(sum(len(t.split()) for t in texts), 1), 3),
    }


# ---- training --------------------------------------------------------------------------------------------
def _batches(enc, labels, order, batch_size):
    """Give batches of tweets and their labels, in the order given by `order`."""
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        batch = {k: v[idx] for k, v in enc.items()}
        yield batch, labels[idx]


def _encode(tokenizer, texts, max_len):
    """Turn all texts into subword ids, padded or cut to max_len. Returns (tensors, torch).

    Unlike src.transformer_utils (dynamic padding per batch), every row here has the same fixed length.
    """
    import torch

    enc = tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}, torch


def predict(model, enc, batch_size, device):
    """Predicted label ids for all tweets, in the same order. Dropout is off."""
    import torch

    model.eval()
    preds = []
    n = enc["input_ids"].shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = {k: v[i:i + batch_size].to(device) for k, v in enc.items()}
            preds.append(model(**batch).logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def fine_tune(model_name, train_df, dev_df, *, seed=DEFAULT_SEED, max_len=128, batch_size=32, lr=2e-5, epochs=3,
              warmup_ratio=0.1, weight_decay=0.01, device=None, checkpoint_dir=None, tokenizer=None, model=None,
              log=print):
    """Fine-tune `model_name` on train_df; pick the best epoch by dev weighted F1. Returns a result dict with
    the dev predictions (list of label strings in dev_df order), per-epoch scores and timing.

    `tokenizer` and `model` can be passed in (used by the tests with a tiny random model); otherwise they are
    loaded from `model_name`.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
    model = model or AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)
    model.to(device)

    enc_tr, _ = _encode(tokenizer, train_df["text_proc"], max_len)
    enc_dev, _ = _encode(tokenizer, dev_df["text_proc"], max_len)
    y_tr = torch.tensor([LABEL2ID[x] for x in train_df["label"]])
    y_dev = list(dev_df["label"])

    # same optimiser recipe as src.train_transformer: AdamW, no weight decay on biases and LayerNorm,
    # linear warm-up then linear decay, mixed precision on GPU only
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "LayerNorm.bias")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps = epochs * int(np.ceil(len(train_df) / batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # seeded generator, so the shuffling order is the same on every rerun
    rng = np.random.default_rng(seed)

    history, best = [], {"weighted_f1": -1.0}
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_df))
        total = 0.0
        for batch, y in _batches(enc_tr, y_tr, order, batch_size):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch, labels=y.to(device)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
        pred_ids = predict(model, enc_dev, batch_size, device)
        pred = [ID2LABEL[int(i)] for i in pred_ids]
        m = compute_metrics(y_dev, pred)
        history.append({"epoch": epoch, "train_loss": round(total / len(train_df), 4),
                        "dev_weighted_f1": round(m["weighted_f1"], 4), "dev_macro_f1": round(m["macro_f1"], 4),
                        "dev_accuracy": round(m["accuracy"], 4)})
        log(f"epoch {epoch}: {history[-1]}")
        # keep the predictions of the best dev epoch (there is no early stopping here: all epochs run)
        if m["weighted_f1"] > best["weighted_f1"]:
            best = {"weighted_f1": m["weighted_f1"], "epoch": epoch, "pred": pred, "metrics": m}
            if checkpoint_dir:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
    return {"best_epoch": best["epoch"], "metrics": best["metrics"], "pred": best["pred"], "history": history,
            "device": device, "train_seconds": round(time.perf_counter() - start, 1)}


# ---- outputs ---------------------------------------------------------------------------------------------
def prediction_path(short_name, seed):
    """Seed 42 -> results/<name>_dev_predictions.csv (the file named in the issue).
    Other seeds -> results/predictions/<name>_s<seed>_seed<seed>_dev.csv (picked up as the same setting by
    src.error_analysis_models and src.cmi_evaluation)."""
    if seed == DEFAULT_SEED:
        return RESULTS / f"{short_name}_dev_predictions.csv"
    return RESULTS / "predictions" / f"{short_name}_s{seed}_seed{seed}_dev.csv"


def save_run(short_name, model_name, dev_df, result, seed, config):
    """Write predictions (uid,label,pred) and a JSON with scores, history and the exact settings."""
    path = prediction_path(short_name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"uid": dev_df["uid"], "label": dev_df["label"], "pred": result["pred"]}).to_csv(path, index=False)
    scores = path.with_name(path.stem.replace("_predictions", "").replace("_dev", "") + f"_dev_scores_seed{seed}.json")
    scores.write_text(json.dumps({
        "model": model_name, "seed": seed, "config": config, "best_epoch": result["best_epoch"],
        "dev_metrics": result["metrics"], "history": result["history"], "device": result["device"],
        "train_seconds": result["train_seconds"],
    }, indent=2), encoding="utf-8")
    return path, scores


def main_for(short_name, default_checkpoint, argv=None):
    """Command line shared by train_muril.py and train_indicbert.py."""
    import argparse

    ap = argparse.ArgumentParser(description=f"Fine-tune {short_name} on SentiMix Hinglish; evaluate on dev.")
    ap.add_argument("--checkpoint", default=default_checkpoint, help="Hugging Face model name or local path")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--text-mode", default="raw", choices=["raw", "clean", "normalized"])
    ap.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--max-train-samples", type=int, default=0, help="small smoke test; 0 = all 14,000")
    ap.add_argument("--max-dev-samples", type=int, default=0, help="small smoke test; 0 = all 3,000")
    ap.add_argument("--inspect-only", action="store_true", help="only report tokenizer behaviour, no training")
    ap.add_argument("--save-checkpoint", action="store_true", help="save best weights to checkpoints/ (not in Git)")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    train, dev = load_texts("train", args.text_mode), load_texts("dev", args.text_mode)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    report = token_length_report(tok, train["text_proc"])
    print(json.dumps({"checkpoint": args.checkpoint, "tokenizer": type(tok).__name__, "vocab_size": tok.vocab_size,
                      "train_token_report": report}, indent=2))
    if args.inspect_only:
        return
    if args.max_train_samples:
        train = train.sample(n=args.max_train_samples, random_state=args.seed).reset_index(drop=True)
    if args.max_dev_samples:
        dev = dev.sample(n=args.max_dev_samples, random_state=args.seed).sort_values("uid").reset_index(drop=True)
    smoke = bool(args.max_train_samples or args.max_dev_samples)
    cfg = {k: getattr(args, k.replace("-", "_")) for k in
           ("text_mode", "max_len", "batch_size", "lr", "epochs", "max_train_samples", "max_dev_samples")}
    ckpt = ROOT / "checkpoints" / f"{short_name}_seed{args.seed}" if args.save_checkpoint else None
    result = fine_tune(args.checkpoint, train, dev, seed=args.seed, max_len=args.max_len, batch_size=args.batch_size,
                       lr=args.lr, epochs=args.epochs, checkpoint_dir=ckpt, tokenizer=tok)
    if smoke:
        print("SMOKE TEST (subset): outputs NOT written so they cannot be mistaken for real results.")
        print(json.dumps(result["history"], indent=2))
        return
    path, scores = save_run(short_name, args.checkpoint, dev, result, args.seed, cfg)
    print(f"dev weighted F1 {result['metrics']['weighted_f1']:.4f} (epoch {result['best_epoch']}); wrote {path}, {scores}")
