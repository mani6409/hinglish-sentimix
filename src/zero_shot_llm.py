"""Zero-shot LLM sentiment classification of Hinglish tweets (issue #8).

No model is trained or fine-tuned and no training label is used: the model only gets an instruction and the
tweet. The dev labels are used afterwards, only to score the answers.

Run from the repo root, for example (local open model on CPU, no account needed):
    python -m src.zero_shot_llm --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --n 200 --prompt v2
or with the Anthropic API (key is read from the environment variable ANTHROPIC_API_KEY, never from a file in
the repository):
    python -m src.zero_shot_llm --backend anthropic --model <model id> --n 200 --prompt v2

Outputs (results/):
    zero_shot_dev_predictions.csv     uid,label,pred   (common format; invalid answers get the fallback label)
    zero_shot_dev_raw_responses.csv   uid,raw_response,parsed,valid,latency_s,prompt_version
    zero_shot_run_meta.json           model name and revision, date, prompt, counts, latency, metrics
A run can be interrupted and started again with the same arguments: finished uids are skipped.
Use --out-suffix to keep several prompt variants side by side without overwriting each other.
"""
import argparse
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_split
from src.metrics import LABELS, compute_metrics
from src.seed import DEFAULT_SEED, set_seed

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FALLBACK_LABEL = "neutral"  # used only so the predictions file stays complete; counted and reported as invalid

# ---- prompts: instructions only, NO labelled examples (that would be few-shot) -----------------------------
PROMPTS = {
    "v1": (
        "Classify the sentiment of the following tweet as positive, negative or neutral.\n"
        "Reply with exactly one word: positive, negative or neutral.\n\n"
        "Tweet: {tweet}\n"
        "Sentiment:"
    ),
    "v2": (
        "You are an expert annotator of social media text. The tweet below is written in Hinglish: Hindi and "
        "English mixed in one message, with Hindi words usually typed in Roman letters (for example 'nahi' = no, "
        "'bakwas' = nonsense, 'accha' = good) and with inconsistent spelling. User names are written as @ name "
        "and links may be cut off.\n"
        "Decide the overall sentiment the author expresses: positive, negative or neutral "
        "(neutral = no clear positive or negative opinion, for example news, questions or plain statements).\n"
        "Answer with exactly one word: positive, negative or neutral. Do not explain.\n\n"
        "Tweet: {tweet}\n"
        "Answer:"
    ),
}

_LABEL_RE = re.compile(r"\b(positive|negative|neutral)\b")


def build_prompt(tweet, version="v2"):
    """Put the tweet into the prompt (v1 = short, v2 = explains Hinglish to the model)."""
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version {version!r}; choose from {sorted(PROMPTS)}")
    return PROMPTS[version].format(tweet=str(tweet).strip())


def parse_response(response):
    """Map a free-text model answer to (label, valid, reason).

    Valid only if exactly one of positive / negative / neutral appears as a word (case-insensitive), ignoring
    punctuation and quotes. Empty answers, answers with none of the words (for example a refusal or an
    explanation without a verdict) and answers that contain two different labels ("positive or negative") are
    invalid. A repeated same label ("Positive. The tweet is positive.") is valid.
    """
    if response is None or not str(response).strip():
        return FALLBACK_LABEL, False, "empty"
    found = set(_LABEL_RE.findall(str(response).lower()))
    if len(found) == 1:
        return found.pop(), True, "ok"
    if not found:
        return FALLBACK_LABEL, False, "no_label"
    return FALLBACK_LABEL, False, "ambiguous"


def classify(tweet, backend, prompt_version="v2"):
    """Reusable zero-shot classifier. `backend` is any callable prompt -> response text.
    Returns a dict with the raw response, parsed label, validity and latency in seconds."""
    prompt = build_prompt(tweet, prompt_version)
    start = time.perf_counter()
    try:
        response = backend(prompt)
    except Exception as exc:  # network / quota / model errors must not stop a long run silently
        response, error = "", f"{type(exc).__name__}: {exc}"
    else:
        error = ""
    latency = time.perf_counter() - start
    label, valid, reason = parse_response(response)
    return {"raw_response": response, "parsed": label, "valid": valid,
            "reason": error or reason, "latency_s": round(latency, 3)}


# ---- backends --------------------------------------------------------------------------------------------
def make_hf_backend(model_name, max_new_tokens=8):
    """Local open model through Hugging Face transformers. Greedy decoding (deterministic). Works on CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    revision = getattr(model.config, "_commit_hash", None)

    def backend(prompt):
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    backend.model_revision = revision
    return backend


def make_anthropic_backend(model_name, max_tokens=16):
    """Anthropic API. The key must be in the environment variable ANTHROPIC_API_KEY (never in the repo).
    Temperature 0 for repeatability. Token counts are summed in backend.usage for cost estimates."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("Set the ANTHROPIC_API_KEY environment variable (outside the repository) to use this backend.")
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    usage = {"input_tokens": 0, "output_tokens": 0}

    def backend(prompt):
        msg = client.messages.create(model=model_name, max_tokens=max_tokens, temperature=0,
                                     messages=[{"role": "user", "content": prompt}])
        usage["input_tokens"] += msg.usage.input_tokens
        usage["output_tokens"] += msg.usage.output_tokens
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    backend.usage = usage
    backend.model_revision = None
    return backend


# ---- experiment ------------------------------------------------------------------------------------------
def sample_dev(dev, n, seed=DEFAULT_SEED):
    """Random subset of n dev tweets (fixed seed, so the same tweets are used every time)."""
    if n is None or n >= len(dev):
        return dev.sort_values("uid").reset_index(drop=True)
    return dev.sample(n=n, random_state=seed).sort_values("uid").reset_index(drop=True)


def read_done(raw_path):
    """Answers saved so far (empty if none), so an interrupted run can carry on."""
    if raw_path.exists():
        return pd.read_csv(raw_path, keep_default_na=False)
    return pd.DataFrame(columns=["uid", "raw_response", "parsed", "valid", "reason", "latency_s", "prompt_version"])


def run(subset, backend, prompt_version, raw_path, save_every=10):
    """Classify every tweet of `subset` not already in raw_path (resume), append results as they come."""
    done = read_done(raw_path)
    todo = subset[~subset["uid"].isin(set(done["uid"]))]
    if len(done) and done["prompt_version"].astype(str).ne(prompt_version).any():
        raise SystemExit(f"{raw_path.name} was made with another prompt version; use --out-suffix to keep runs apart.")
    rows = []
    for i, r in enumerate(todo.itertuples(), start=1):
        res = classify(r.text, backend, prompt_version)
        rows.append({"uid": r.uid, **res, "prompt_version": prompt_version})
        if i % save_every == 0 or i == len(todo):
            done = pd.DataFrame(rows) if done.empty else pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(raw_path, index=False)
            rows = []
            print(f"  {len(done)}/{len(subset)} done", flush=True)
    return read_done(raw_path)


def evaluate(subset, raw):
    """Metrics on the tweets of `subset`. Invalid answers are counted as the fallback label in the main score;
    a second score on valid answers only is given for reference."""
    m = subset.merge(raw, on="uid", how="inner")
    parsed = m["parsed"].astype(str)
    valid = m["valid"].astype(str).str.lower().eq("true")
    all_scores = compute_metrics(m["label"], parsed)
    out = {
        "n_evaluated": int(len(m)), "n_invalid": int((~valid).sum()),
        "invalid_rate": round(float((~valid).mean()), 4),
        "invalid_reasons": m.loc[~valid, "reason"].astype(str).str.split(":").str[0].value_counts().to_dict(),
        "all_answers": all_scores,
        "valid_answers_only": compute_metrics(m.loc[valid, "label"], parsed[valid]) if valid.any() else None,
        "predicted_label_counts": parsed.value_counts().to_dict(),
        "mean_latency_s": round(float(m["latency_s"].astype(float).mean()), 3),
    }
    return out, m


def baseline_scores_on(uids, root=ROOT):
    """Weighted F1 of the existing dev predictions restricted to the same tweets (fair comparison)."""
    out = {}
    for f in sorted((Path(root) / "results" / "predictions").glob("*_dev.csv")):
        d = pd.read_csv(f)
        d = d[d["uid"].isin(uids)]
        if len(d) == len(uids):
            out[f.stem] = round(compute_metrics(d["label"], d["pred"])["weighted_f1"], 4)
    return out


def write_outputs(subset, raw, meta_extra, suffix=""):
    """Score the answers, save the predictions and the run info, and return the run info."""
    metrics, merged = evaluate(subset, raw)
    preds = merged[["uid", "label", "parsed"]].rename(columns={"parsed": "pred"})
    preds.to_csv(RESULTS / f"zero_shot_dev_predictions{suffix}.csv", index=False)
    meta = {**meta_extra, "date_run": str(date.today()), "metrics": metrics,
            "baselines_same_tweets_weighted_f1": baseline_scores_on(set(preds["uid"]))}
    (RESULTS / f"zero_shot_run_meta{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None):
    """Pick a random dev subset, ask the model about each tweet, and save scores and predictions."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["hf", "anthropic"], required=True)
    ap.add_argument("--model", required=True, help="model name or id")
    ap.add_argument("--prompt", default="v2", choices=sorted(PROMPTS))
    ap.add_argument("--n", type=int, default=200, help="size of the random dev subset (0 = all 3,000)")
    ap.add_argument("--out-suffix", default="", help="e.g. _v1 to keep a second prompt's files separate")
    args = ap.parse_args(argv)

    set_seed(DEFAULT_SEED)
    dev = load_split("dev")
    subset = sample_dev(dev, args.n or None)
    backend = (make_hf_backend if args.backend == "hf" else make_anthropic_backend)(args.model)
    raw_path = RESULTS / f"zero_shot_dev_raw_responses{args.out_suffix}.csv"
    print(f"{len(subset)} dev tweets, backend={args.backend}, model={args.model}, prompt={args.prompt}")
    wall = time.perf_counter()
    raw = run(subset, backend, args.prompt, raw_path)
    meta = write_outputs(subset, raw, {
        "backend": args.backend, "model": args.model,
        "model_revision": getattr(backend, "model_revision", None),
        "prompt_version": args.prompt, "prompt_template": PROMPTS[args.prompt],
        "decoding": "greedy (HF) / temperature 0 (API)", "subset": f"random {len(subset)} dev tweets, seed {DEFAULT_SEED}",
        "api_token_usage": getattr(backend, "usage", None), "wall_time_this_session_s": round(time.perf_counter() - wall, 1),
        "fallback_label_for_invalid": FALLBACK_LABEL,
    }, args.out_suffix)
    print(json.dumps({k: meta[k] for k in ("metrics", "baselines_same_tweets_weighted_f1")}, indent=2)[:3000])


if __name__ == "__main__":
    main()
