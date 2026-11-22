"""Categorized error analysis of all dev predictions (issue #9).

Run from the repo root:  python -m src.error_analysis_models

Inputs (read only, nothing is retrained):
  results/predictions/*_dev.csv      one file per model run, columns uid,label,pred
  results/muril_dev_predictions.csv, results/indicbert_dev_predictions.csv, results/zero_shot_dev_predictions.csv
                                     (issues #7 and #8; used automatically if they exist)
  results/cmi_per_tweet.csv          CMI per tweet (test rows are dropped, as in src/cmi_evaluation.py)
  data/raw/...                       dev tweets (src/data.py)

Outputs (results/):
  error_analysis_models.csv          one row per dev tweet that at least one setting got wrong, with flags
  error_model_summary.csv            per setting: scores, errors, confusion pairs
  error_flag_rates_models.csv        per flag: error rate with vs without the flag (all settings pooled), chi-square p
  error_agreement.csv                pairwise overlap of the wrong tweets of two settings
  error_cmi_rates.csv                error rate per setting and CMI bucket (95% Wilson interval)
  error_manual_sample.csv            stratified sample of errors with a column for hand labels; labels already in
                                     the file are kept on a rerun (matched by uid), never blanked

Important honesty note: the "flags" are automatic keyword/pattern heuristics. They mark candidates for a
category; they are NOT the category itself. "context_dependent" and the final "annotation error" decision
need a human reader, which is what error_manual_sample.csv is for.

A "setting" is a model configuration. Runs that differ only in seed (names ending in _s42_seed42 etc.) are
merged into one setting by majority vote over the seeds (ties go to the lowest seed). This is an ensemble of
the seeds, so its score can differ from the mean single-seed score; the summary reports both.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.cmi_evaluation import assign_bucket, bucket_edges, read_cmi_without_test
from src.data import load_split
from src.metrics import LABELS, compute_metrics
from src.preprocess import normalize_hinglish
from src.seed import DEFAULT_SEED, set_seed
from src.uid_checks import check_uid_alignment

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXTRA_PREDICTION_FILES = [  # issue-named files outside results/predictions/
    RESULTS / "muril_dev_predictions.csv",
    RESULTS / "indicbert_dev_predictions.csv",
    RESULTS / "zero_shot_dev_predictions.csv",
]

# ---- error categories (definitions used in the report) ---------------------------------------------------
CATEGORIES = {
    "sarcasm_irony": "Positive-sounding words or laughing emoji, but the gold label is negative (or the reverse).",
    "slang_informal": "Slang, abuse words, shouting in capitals, or stretched letters (soooo).",
    "code_mixing": "Tweet is in the high code-mixing group (CMI above the training 2/3 quantile).",
    "negation": "Contains a negation word (nahi, na, mat, not, don't ...); the model may ignore the flip.",
    "transliteration_spelling": "Contains a Romanised-Hindi spelling variant that src.preprocess would rewrite.",
    "context_dependent": "Needs outside knowledge (event, person, news). Cannot be detected automatically: hand label only.",
    "annotation_error_candidate": "Most settings agree on the same wrong label, so the gold label itself may be wrong.",
    "very_short": "15 tokens or fewer (about 8% of tweets; the shortest dev tweet has 8 tokens, the mean is 26).",
}
FLAG_COLUMNS = [
    "negation", "slang_informal", "sarcasm_cue", "spelling_variant", "very_short", "high_cmi",
    "label_noise_candidate",
]
MANUAL_CATEGORIES = [c for c in CATEGORIES] + ["other"]

NEGATION_WORDS = {
    "nahi", "nahin", "nhi", "nahii", "na", "mat", "never", "not", "no", "dont", "don't", "doesnt", "doesn't",
    "didnt", "didn't", "isnt", "isn't", "cant", "can't", "cannot", "wont", "won't", "wasnt", "wasn't",
    "nai", "bina", "nothing", "n't",
}
SLANG_WORDS = {
    "bc", "mc", "bsdk", "chutiya", "chutiye", "saala", "sala", "kutta", "kamina", "harami", "bakwas", "bhai", "yaar",
    "bro", "lol", "lmao", "rofl", "wtf", "omg", "fuck", "fucking", "shit", "bloody", "dude", "abe", "oye", "ghanta",
}
POSITIVE_CUES = {
    "wah", "waah", "wow", "great", "nice", "thanks", "thank", "bravo", "lovely", "best", "superb", "mast",
    "zabardast", "shandar", "congrats", "congratulations", "good",
}
LAUGH_EMOJI = set("😂😅🤣😏🙄😆😜😝")
_ELONGATED_RE = re.compile(r"(.)\1{2,}")
_MENTION_RE = re.compile(r"@\s*\w+(?:\s+_+\s+\w+)*")
SHORT_MAX_TOKENS = 15
NOISE_AGREEMENT = 0.75  # share of settings that must give the same wrong label
NOISE_MIN_SETTINGS = 4
BONFERRONI_TESTS = len(FLAG_COLUMNS)


# ---- loading ---------------------------------------------------------------------------------------------
def setting_name(file_stem):
    """results/predictions/bilstm_full_raw_s42_seed42_dev.csv -> 'bilstm_full_raw'."""
    name = re.sub(r"_dev(_predictions)?$", "", file_stem)
    name = re.sub(r"_s\d+_seed\d+$", "", name)
    return name


def seed_of(file_stem):
    m = re.search(r"seed(\d+)", file_stem)
    return int(m.group(1)) if m else 0


def prediction_files(root=ROOT):
    files = sorted((Path(root) / "results" / "predictions").glob("*_dev.csv"))
    files += [p for p in (Path(root) / "results" / p.name for p in EXTRA_PREDICTION_FILES) if p.exists()]
    return files


def anonymise(text):
    return _MENTION_RE.sub("@USER", text)


def load_predictions(files, dev):
    """Return {setting: DataFrame(uid, label, pred, n_runs, run_scores)} with seeds merged by majority vote.

    Every file's uids are checked against the dev uids (no duplicates / unexpected). A file may cover only a
    SUBSET of dev (the LLM run may be limited by cost); then the setting just covers those uids.
    """
    gold = dict(zip(dev["uid"], dev["label"]))
    runs = {}
    for f in files:
        df = pd.read_csv(f)
        if list(df.columns[:3]) != ["uid", "label", "pred"]:
            raise ValueError(f"{f.name}: expected columns uid,label,pred, got {list(df.columns)}")
        if df["uid"].duplicated().any():
            raise ValueError(f"{f.name}: duplicate uids")
        unknown = set(df["uid"]) - set(gold)
        if unknown:
            raise ValueError(f"{f.name}: {len(unknown)} uids are not dev uids, e.g. {sorted(unknown)[:3]}")
        if len(df) == len(dev):
            check_uid_alignment(df["uid"], dev["uid"], name=f.name, reference_name="dev")
        wrong_gold = (df["uid"].map(gold) != df["label"]).sum()
        if wrong_gold:
            raise ValueError(f"{f.name}: {wrong_gold} gold labels differ from the dev file")
        bad = set(df["pred"]) - set(LABELS)
        if bad:
            raise ValueError(f"{f.name}: unknown predicted labels {sorted(bad)}")
        runs.setdefault(setting_name(f.stem), []).append((seed_of(f.stem), df[["uid", "label", "pred"]]))
    out = {}
    for name, items in runs.items():
        items.sort(key=lambda t: t[0])
        out[name] = merge_seeds([d for _, d in items])
    return out


def merge_seeds(frames):
    """Majority vote over several runs of the same setting. Ties go to the first (lowest-seed) run."""
    base = frames[0].set_index("uid")
    if len(frames) == 1:
        return base.reset_index().assign(n_runs=1)
    votes = pd.DataFrame({i: f.set_index("uid")["pred"] for i, f in enumerate(frames)})
    votes = votes.loc[base.index]

    def vote(row):
        counts = row.value_counts()
        top = counts[counts == counts.max()].index
        return row.iloc[0] if row.iloc[0] in top else top[0]

    merged = base.copy()
    merged["pred"] = votes.apply(vote, axis=1)
    return merged.reset_index().assign(n_runs=len(frames))


# ---- flags -----------------------------------------------------------------------------------------------
def _words(text):
    return re.findall(r"[a-z']+", text.lower())


def flag_row(text, tokens, label, pred, normalised_changed, high_cmi):
    """Automatic candidate flags for one tweet. See the module docstring: these are heuristics."""
    words = _words(text)
    wset = set(words)
    caps = any(len(w) >= 3 and w.isupper() and w.isalpha() for w in text.split())
    laugh = any(ch in LAUGH_EMOJI for ch in text)
    slang = bool(_ELONGATED_RE.search(text.lower())) or caps or bool(wset & SLANG_WORDS)
    positive_looking = bool(wset & POSITIVE_CUES) or laugh
    return {
        "negation": bool(wset & NEGATION_WORDS) or "n't" in text.lower(),
        "slang_informal": slang,
        # gold negative but the text looks positive: the typical shape of sarcasm.
        "sarcasm_cue": label == "negative" and positive_looking,
        "spelling_variant": bool(normalised_changed),
        "very_short": len(tokens) <= SHORT_MAX_TOKENS,
        "high_cmi": bool(high_cmi),
    }


def normalised_changed(text):
    """True if the Hinglish spelling normaliser rewrites anything beyond case and spacing."""
    plain = re.sub(r"\s+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", normalize_hinglish(text).lower()).strip() != plain


def bucket_lookup(cmi_table, edges):
    """{uid: 'low'|'medium'|'high'} for every tweet in the CMI table."""
    return dict(zip(cmi_table["uid"], assign_bucket(cmi_table["cmi"], edges).astype(str)))


def build_error_table(dev, settings, cmi_table, edges):
    """One row per dev tweet misclassified by >= 1 setting. Returns (table, wrong_matrix)."""
    cmi = cmi_table.set_index("uid")["cmi"]
    buckets = bucket_lookup(cmi_table, edges)
    base = dev.set_index("uid")
    wrong = pd.DataFrame(index=base.index)
    preds = pd.DataFrame(index=base.index)
    for name, df in settings.items():
        d = df.set_index("uid")
        wrong[name] = (d["pred"] != d["label"]).reindex(base.index)  # NaN where setting has no prediction
        preds[name] = d["pred"].reindex(base.index)
    covered = wrong.notna().sum(axis=1)
    n_wrong = wrong.fillna(False).astype(bool).sum(axis=1)
    keep = n_wrong > 0
    rows = []
    for uid in base.index[keep]:
        label = base.at[uid, "label"]
        wrong_preds = preds.loc[uid][wrong.loc[uid].fillna(False).astype(bool)]
        top_pred, top_n = wrong_preds.value_counts().index[0], int(wrong_preds.value_counts().iloc[0])
        text = base.at[uid, "text"]
        c = cmi.get(uid, np.nan)
        bucket = buckets.get(uid, "")
        flags = flag_row(text, base.at[uid, "tokens"], label, top_pred, normalised_changed(text), bucket == "high")
        share = top_n / covered[uid]
        flags["label_noise_candidate"] = bool(covered[uid] >= NOISE_MIN_SETTINGS and share >= NOISE_AGREEMENT)
        rows.append({
            "uid": uid, "text": anonymise(text), "label": label, "cmi": c,
            "cmi_bucket": bucket,
            "n_settings_covering": int(covered[uid]), "n_settings_wrong": int(n_wrong[uid]),
            "most_common_wrong_pred": top_pred, "wrong_settings": ";".join(wrong_preds.index),
            **flags,
        })
    return pd.DataFrame(rows), wrong


# ---- summaries -------------------------------------------------------------------------------------------
def model_summary(settings, file_scores):
    rows = []
    for name, df in settings.items():
        m = compute_metrics(df["label"], df["pred"])
        wrong = df[df["pred"] != df["label"]]
        pairs = wrong.groupby(["label", "pred"]).size().sort_values(ascending=False)
        neutral = int(((wrong["label"] == "neutral") | (wrong["pred"] == "neutral")).sum())
        row = {
            "setting": name, "n_runs": int(df["n_runs"].iloc[0]), "n_tweets": len(df), "n_errors": len(wrong),
            "weighted_f1": round(m["weighted_f1"], 4), "macro_f1": round(m["macro_f1"], 4),
            "accuracy": round(m["accuracy"], 4),
            "mean_single_run_weighted_f1": round(file_scores.get(name, np.nan), 4),
            "share_errors_involving_neutral": round(neutral / max(len(wrong), 1), 3),
            "top_confusion": f"{pairs.index[0][0]}->{pairs.index[0][1]} ({int(pairs.iloc[0])})" if len(pairs) else "",
        }
        for lab in LABELS:
            row[f"recall_{lab}"] = round(m["per_class"][lab]["recall"], 3)
            row[f"precision_{lab}"] = round(m["per_class"][lab]["precision"], 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)


def flag_rates(dev, settings, cmi_table, edges):
    """Error rate with vs without each flag, pooled over (setting, tweet) pairs. Flags are computed for ALL dev
    tweets so the comparison group (correct tweets) is fair. chi-square test of independence; the Bonferroni
    threshold is 0.05 / number of flags. Pooled pairs are not independent (the same tweet appears once per
    setting), so p-values are only a rough guide."""
    buckets = bucket_lookup(cmi_table, edges)
    flags = {}
    for r in dev.itertuples():
        text = r.text
        f = flag_row(text, r.tokens, r.label, None, normalised_changed(text), buckets.get(r.uid) == "high")
        f.pop("sarcasm_cue")  # depends on gold label + errors; compared separately below
        flags[r.uid] = f
    flags = pd.DataFrame.from_dict(flags, orient="index")
    out = []
    for flag in [c for c in FLAG_COLUMNS if c in flags.columns]:
        wrong_with = n_with = wrong_without = n_without = 0
        for df in settings.values():
            f = flags.loc[df["uid"], flag].to_numpy()
            w = (df["pred"] != df["label"]).to_numpy()
            wrong_with += int((w & f).sum()); n_with += int(f.sum())
            wrong_without += int((w & ~f).sum()); n_without += int((~f).sum())
        table = [[wrong_with, n_with - wrong_with], [wrong_without, n_without - wrong_without]]
        p = chi2_contingency(table)[1] if min(n_with, n_without) > 0 else np.nan
        out.append({
            "flag": flag, "n_pairs_with": n_with, "error_rate_with": round(wrong_with / max(n_with, 1), 4),
            "n_pairs_without": n_without, "error_rate_without": round(wrong_without / max(n_without, 1), 4),
            "chi2_p": p, "bonferroni_threshold": 0.05 / BONFERRONI_TESTS,
            "significant_after_bonferroni": bool(p < 0.05 / BONFERRONI_TESTS) if not np.isnan(p) else False,
        })
    return pd.DataFrame(out)


def agreement_table(wrong):
    """Pairwise overlap of wrong tweets (only tweets covered by both settings). jaccard = |both wrong| / |either wrong|."""
    names = list(wrong.columns)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = wrong[[a, b]].dropna()
            if len(both) == 0:
                continue
            wa, wb = both[a].astype(bool), both[b].astype(bool)
            either = int((wa | wb).sum())
            rows.append({
                "setting_a": a, "setting_b": b, "n_common_tweets": len(both),
                "wrong_a": int(wa.sum()), "wrong_b": int(wb.sum()), "wrong_both": int((wa & wb).sum()),
                "jaccard": round(float((wa & wb).sum()) / either, 3) if either else np.nan,
            })
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def cmi_error_rates(settings, cmi_table, edges):
    bucket = pd.Series(assign_bucket(cmi_table["cmi"], edges).astype(str).to_numpy(), index=cmi_table["uid"].to_numpy())
    rows = []
    for name, df in settings.items():
        b = bucket.reindex(df["uid"]).to_numpy()
        w = (df["pred"] != df["label"]).to_numpy()
        for bk in ["low", "medium", "high"]:
            mask = b == bk
            k, n = int(w[mask].sum()), int(mask.sum())
            lo, hi = wilson(k, n)
            rows.append({"setting": name, "cmi_bucket": bk, "n": n, "errors": k,
                         "error_rate": round(k / n, 4) if n else np.nan,
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    return pd.DataFrame(rows)


def manual_sample(errors, summary, n=60, seed=DEFAULT_SEED):
    """Stratified random sample of errors of the best setting, covering every (gold -> predicted) pair in
    proportion (at least 3 per pair when available). The hand-label columns are left empty on purpose;
    keep_existing_labels() copies earlier labels back in before the file is written."""
    best = summary.iloc[0]["setting"]
    pool = errors[errors["wrong_settings"].str.split(";").apply(lambda s: best in s)].copy()
    pool["pair"] = pool["label"] + "->" + pool["most_common_wrong_pred"]
    rng = np.random.default_rng(seed)
    parts = []
    for pair, g in pool.groupby("pair"):
        k = min(len(g), max(3, round(n * len(g) / len(pool))))
        parts.append(g.sample(k, random_state=int(rng.integers(1 << 31))))
    sample = pd.concat(parts).head(max(n, len(parts) * 3)).sort_values("uid")
    sample = sample[["uid", "text", "label", "most_common_wrong_pred", "cmi_bucket", "n_settings_wrong"]].copy()
    sample.insert(0, "best_setting", best)
    sample["manual_category"] = ""  # to be filled by hand with one of MANUAL_CATEGORIES
    sample["notes"] = ""
    return sample


def keep_existing_labels(sample, path):
    """Carry hand labels over from an existing manual-sample file, matched by uid.

    Without this, rerunning the script would blank the labels that a person has filled in. Rows of the old file
    that have a label are kept: if the uid is in the new sample its label and notes are copied over; if it is not
    (the best setting or the predictions changed), the whole labelled row is appended, so no label is ever lost.
    Rows without a label are not kept. Returns (new_frame, number_of_labels_carried_over)."""
    path = Path(path)
    if not path.exists():
        return sample, 0
    old = pd.read_csv(path, keep_default_na=False)
    if not {"uid", "manual_category"}.issubset(old.columns):
        return sample, 0
    old = old[old["manual_category"].astype(str).str.strip() != ""].drop_duplicates("uid", keep="last")
    if old.empty:
        return sample, 0
    sample = sample.copy()
    old_notes = old.set_index("uid")["notes"] if "notes" in old.columns else pd.Series(dtype=object)
    labels = old.set_index("uid")["manual_category"]
    in_new = sample["uid"].isin(labels.index)
    sample.loc[in_new, "manual_category"] = sample.loc[in_new, "uid"].map(labels).to_numpy()
    if len(old_notes):
        sample.loc[in_new, "notes"] = sample.loc[in_new, "uid"].map(old_notes).fillna("").to_numpy()
    extra = old[~old["uid"].isin(sample["uid"])]
    if len(extra):
        extra = extra.reindex(columns=sample.columns, fill_value="")
        sample = pd.concat([sample, extra], ignore_index=True).sort_values("uid")
    return sample, len(old)


def file_level_scores(files, dev):
    """Mean single-run weighted F1 per setting (for reference next to the majority-vote score)."""
    per = {}
    for f in files:
        d = pd.read_csv(f)
        per.setdefault(setting_name(f.stem), []).append(compute_metrics(d["label"], d["pred"])["weighted_f1"])
    return {k: float(np.mean(v)) for k, v in per.items()}


def main():
    set_seed(DEFAULT_SEED)
    dev = load_split("dev")
    files = prediction_files()
    if not files:
        raise SystemExit("No dev prediction files found in results/predictions/")
    settings = load_predictions(files, dev)
    cmi_all = read_cmi_without_test(RESULTS / "cmi_per_tweet.csv")
    edges = bucket_edges(cmi_all.loc[cmi_all["split"] == "train", "cmi"])
    cmi_dev = cmi_all[cmi_all["split"] == "dev"]
    check_uid_alignment(cmi_dev["uid"], dev["uid"], name="cmi dev rows", reference_name="dev")

    errors, wrong = build_error_table(dev, settings, cmi_dev, edges)
    summary = model_summary(settings, file_level_scores(files, dev))
    rates = flag_rates(dev, settings, cmi_dev, edges)
    errors.to_csv(RESULTS / "error_analysis_models.csv", index=False)
    summary.to_csv(RESULTS / "error_model_summary.csv", index=False)
    rates.to_csv(RESULTS / "error_flag_rates_models.csv", index=False)
    agreement_table(wrong).to_csv(RESULTS / "error_agreement.csv", index=False)
    cmi_error_rates(settings, cmi_dev, edges).to_csv(RESULTS / "error_cmi_rates.csv", index=False)
    manual_path = RESULTS / "error_manual_sample.csv"
    sample, n_kept = keep_existing_labels(manual_sample(errors, summary), manual_path)
    sample.to_csv(manual_path, index=False)
    print(f"manual sample: {len(sample)} rows, {n_kept} existing hand labels kept")

    print(f"settings: {list(settings)}")
    print(f"dev tweets wrong in >= 1 setting: {len(errors)} of {len(dev)}")
    print(summary[["setting", "n_runs", "n_tweets", "weighted_f1", "mean_single_run_weighted_f1", "top_confusion"]]
          .to_string(index=False))
    print(rates.to_string(index=False))


if __name__ == "__main__":
    main()
