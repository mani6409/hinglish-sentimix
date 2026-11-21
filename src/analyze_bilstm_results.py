"""Development-set analysis of the nine BiLSTM runs (raw / clean / normalized x seeds 42, 43, 44).

Read-only on every existing output. Nothing is trained, the test split and test labels are never used, and no
existing result file is modified. Run from the repo root (needs only pandas, numpy, scikit-learn, matplotlib):

    python -m src.analyze_bilstm_results [--data-root data/raw]

Reads:  results/bilstm_full_<mode>_s<seed>_dev_scores.json, results/predictions/bilstm_full_<mode>_s<seed>_seed<seed>_dev.csv,
        results/cmi_per_tweet.csv (train and dev rows only), results/eda_part2_stats.json, results/tfidf_dev_scores.json,
        and the official DEV file (for tweet text and gold labels).
Writes: reports/bilstm_results_analysis.md and results/figures/bilstm_*.png (refuses to overwrite unless --overwrite).

Everything is deterministic (the bootstrap uses a fixed seed). The error-analysis examples are fixed UIDs that were chosen
after reading rule-based candidate pools; their gold labels, predictions and text are always taken from the data here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

from src.data import load_split  # noqa: E402
from src.preprocess import load_processed_split, normalize_repeated_chars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODES = ("raw", "clean", "normalized")
SEEDS = (42, 43, 44)
LABELS = ["negative", "neutral", "positive"]
BUCKETS = ("low", "mid", "high")
BOOT_B, BOOT_SEED = 4000, 20261116

REPORT_PATH = ROOT / "reports" / "bilstm_results_analysis.md"
FIG_DIR = ROOT / "results" / "figures"
FIG_NAMES = {**{m: f"bilstm_confusion_{m}.png" for m in MODES}, "all": "bilstm_confusion_all_modes.png",
             "cmi": "bilstm_cmi_bucket_metrics.png"}

# ---------------------------------------------------------------- palette ----
# validated reference palette (dataviz skill): categorical slots 1-3 for the three modes, one-hue blue ramp for heatmaps
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODE_COLOR = {"raw": "#2a78d6", "clean": "#eb6834", "normalized": "#1baf7a"}
MODE_MARKER = {"raw": "o", "clean": "s", "normalized": "D"}  # secondary encoding besides color
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

# Reviewed examples: (uid, my reading). Gold labels, predictions, CMI and text are always read from the data.
EXAMPLES = {
    "Negation": [
        (1678, "'nahi' negates an insulting verb ('gali nahi dega'); the abusive word itself looks negative on the surface"),
        (24, "negative-sounding words ('nonesense', 'baymani') together with 'nahi ho sakti' (cannot happen); polarity depends on the writer's stance"),
        (439, "'nhi' in a tweet that mixes negative ('atankistan') and positive ('achi') words"),
        (1378, "'nothing great' negates positive-sounding words ('great', 'excited'); predicted positive"),
        (1701, "'not a fan of ...' followed by a positive-sounding 'good luck'"),
    ],
    "Possible sarcasm / irony (my reading)": [
        (81, "positive phrases ('Accha kiya', 'Good politics', 'Well done') next to 'Most corrupted party'; reads as ironic criticism"),
        (252, "ironic thanks ('thanks for reminding everyone of your OTHER clusterfuck'); positive-sounding 'haha', 'thanks'"),
        (14775, "a well-wishing phrase ('All the best') with a partisan jab; reads as mocking"),
        (1656, "'Congratulations on your marvellous victory' followed by 'but ...'; reads as ironic criticism"),
        (28203, "laughter emojis with a mocking question; the emoji signals amusement, not positive sentiment"),
        (1969, "'Sublime ... I am pretty sure ... you will repeat the same'; reads as ironic"),
    ],
    "Spelling variation": [
        (1553, "variant 'nhi' (mapped to 'nahi' in normalized mode) but wrong in all nine runs, so normalizing the spelling did not rescue it"),
        (3249, "variant 'nhi' again; accusatory wording but a neutral gold label"),
        (35946, "normalization HELPED: 'nhi'/'krna' -> 'nahi'/'karna'; raw predicted the wrong class in 3/3 seeds, normalized was right in 3/3"),
        (13602, "normalization HELPED: 'Bakvas' -> 'Bakwas' (a more frequent spelling in train, see the counts below); raw wrong in 3/3 seeds, normalized right in 3/3"),
        (17582, "normalization HURT: the elongated 'Yeeeee' became 'Yee'; raw right in 3/3 seeds, normalized wrong in 3/3 (elongation may carry emphasis)"),
        (28346, "normalization HURT: 'Yehhh', 'looggg' shortened; raw right in 3/3 seeds, normalized wrong in 3/3"),
    ],
    "Ambiguous sentiment / possible label inconsistency (my reading)": [
        (715, "gold neutral, yet explicit positive wording ('I am so happy'); all nine runs predicted positive"),
        (1338, "gold neutral, yet 'Thank you ... such an awesome interview'"),
        (1425, "gold neutral, yet 'HOW MUCH WE LOVE THIS MAN'"),
        (605, "gold neutral, yet 'Happy Sunday' with heart and smile emojis"),
        (64, "short, factual-sounding remark labelled positive; predicted neutral"),
        (1334, "gold neutral, but predicted negative: a pejorative implication without explicit sentiment words"),
    ],
    "Code-mixing": [
        (101, "CMI 50 (the maximum); several words look like neither Hindi nor English, plus laughter emojis. The language tags behind CMI are noisy"),
        (469, "high CMI; English sentences with Hindi/Urdu invocations ('Allah khair', 'Bismillah', 'Amitbhai ki Jai'); positive-sounding phrases, negative gold"),
        (497, "high CMI; English mixed with Hindi/Urdu mocking remarks about a sports defeat; predicted neutral"),
        (1415, "low CMI (Hindi/Urdu-dominant romanized text) with a tongue-out emoji; gold positive, predicted neutral"),
        (1969, "low CMI (English-dominant) sarcasm; shows that hard tweets are not only highly mixed ones"),
    ],
}
NEG_TOKENS = {"nahi", "nhi", "nahin", "nahim", "nhin", "na", "naa", "mat", "nai", "no", "not", "never", "dont", "cannot",
              "nope", "nothing", "neither", "nor"}


# ================================================================ helpers ====
def counts_text(counter, order=None) -> str:
    keys = order or sorted(counter)
    return ", ".join(f"{k} {counter.get(k, 0):,}" for k in keys)


def md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


def ms(values, d: int = 4) -> str:
    v = np.asarray(values, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def table(header, rows) -> list:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def metrics_of(y_true, y_pred) -> dict:
    """Accuracy, macro / weighted precision-recall-F1 and per-class P/R/F1/support for string label arrays."""
    out = {"accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean())}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average=avg, zero_division=0)
        out.update({f"{avg}_precision": float(p), f"{avg}_recall": float(r), f"{avg}_f1": float(f)})
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out["per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                        for i, c in enumerate(LABELS)}
    return out


def confusion_counts(y_true, y_pred) -> np.ndarray:
    idx = {c: i for i, c in enumerate(LABELS)}
    m = np.zeros((3, 3), int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


# ============================================================ data loading ====
def load_everything(data_root: Path):
    checks = []  # (description, passed, detail)

    def check(desc, ok, detail=""):
        checks.append((desc, bool(ok), detail))
        return ok

    dev = load_split("dev", root=data_root)  # the official DEV file only
    uids = dev["uid"].tolist()
    gold = dict(zip(dev["uid"], dev["label"]))
    y = dev["label"].to_numpy()
    check("Official dev file: 3,000 unique uids", len(uids) == 3000 and len(set(uids)) == 3000, f"{len(uids)} rows")

    # load the stored scores and the predictions of all nine runs
    scores, preds = {}, {}
    json_diffs = []
    for m in MODES:
        for s in SEEDS:
            name = f"full_{m}_s{s}"
            d = json.loads((ROOT / "results" / f"bilstm_{name}_dev_scores.json").read_text(encoding="utf-8"))
            scores[(m, s)] = d
            check(f"{name}: scores JSON metadata (mode, seed, limits 56/18, single seed)",
                  d["config"]["mode"] == m and d["seeds"] == [s] and (d["limits"]["max_words"], d["limits"]["max_chars"]) == (56, 18))
            # read one run's prediction file and check it before using it
            df = pd.read_csv(ROOT / "results" / "predictions" / f"bilstm_{name}_seed{s}_dev.csv")
            ok_cols = list(df.columns) == ["uid", "label", "pred"]
            ok_rows = len(df) == 3000 and df["uid"].is_unique
            ok_set = set(df["uid"]) == set(uids)
            ok_lab = ok_set and bool((df["label"].to_numpy() == df["uid"].map(gold).to_numpy()).all())
            ok_vals = set(df["pred"]) <= set(LABELS)
            same_order = df["uid"].tolist() == uids
            check(f"{name}: prediction file has columns uid,label,pred; 3,000 unique uids; uid set == official dev; "
                  f"gold labels agree by uid; valid predictions",
                  ok_cols and ok_rows and ok_set and ok_lab and ok_vals, f"row order equals dev file order: {same_order}")
            preds[(m, s)] = df.set_index("uid").loc[uids, "pred"].to_numpy()  # aligned BY UID, never by row position
            # recompute the stored metrics from the predictions
            mm = metrics_of(y, preds[(m, s)])
            st = d["per_seed"][0]["metrics"]
            diff = max(abs(mm["accuracy"] - st["accuracy"]), abs(mm["macro_f1"] - st["macro_f1"]),
                       abs(mm["weighted_f1"] - st["weighted_f1"]),
                       *[abs(mm["per_class"][c][k] - st["per_class"][c][k]) for c in LABELS for k in ("precision", "recall", "f1")])
            json_diffs.append(diff)
    check("Metrics recomputed from the nine prediction files equal the stored score JSONs "
          "(accuracy, macro F1, weighted F1, per-class precision/recall/F1)", max(json_diffs) < 1e-9,
          f"max absolute difference {max(json_diffs):.2e}")

    # CMI: read ONLY train and dev rows; test rows are skipped without parsing their label or cmi value
    rows = {"train": [], "dev": []}
    skipped = 0
    with open(ROOT / "results" / "cmi_per_tweet.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] in rows:
                rows[r["split"]].append((int(r["uid"]), r["label"], float(r["cmi"])))
            else:
                skipped += 1
    check("CMI file: test rows skipped (only their split field was inspected; their label and CMI values were not parsed or used)", skipped == 3000, f"{skipped} rows skipped")
    cdev = pd.DataFrame(rows["dev"], columns=["uid", "label", "cmi"])
    check("CMI dev rows: 3,000, unique uids, uid set == official dev uids (no missing, no duplicate)",
          len(cdev) == 3000 and cdev["uid"].is_unique and set(cdev["uid"]) == set(uids))
    # put the CMI rows in the same order as the dev tweets (matched by uid)
    cdev = cdev.set_index("uid").loc[uids]
    check("CMI dev label column equals the official dev gold label (join by uid)", bool((cdev["label"].to_numpy() == y).all()))
    check("CMI values are within [0, 50]", bool(cdev["cmi"].between(0, 50).all()))
    train_cmi = np.array([c for _, _, c in rows["train"]])
    check("CMI train rows: 14,000", len(train_cmi) == 14000)
    _, edges = pd.qcut(train_cmi, 3, retbins=True)  # Mani's definition: equal-count terciles of TRAIN (src/eda_part2.py)
    tr_bucket = pd.cut(train_cmi, bins=edges, include_lowest=True, labels=list(BUCKETS))
    sizes = [int((tr_bucket == b).sum()) for b in BUCKETS]
    rec = json.loads((ROOT / "results" / "eda_part2_stats.json").read_text(encoding="utf-8"))["train_label_share_by_cmi_tercile_%"]
    rec_sizes = [rec[b]["tweets"] for b in BUCKETS]
    # Mani recorded the ranges rounded to one decimal, so compare at that precision
    rng_ok = all(round(float(train_cmi[tr_bucket == b].min()), 1) == rec[b]["cmi_range"][0]
                 and round(float(train_cmi[tr_bucket == b].max()), 1) == rec[b]["cmi_range"][1] for b in BUCKETS)
    check("Train tercile edges reproduce Mani's recorded bucket sizes and ranges (eda_part2_stats.json)",
          sizes == rec_sizes and rng_ok, f"train bucket sizes {sizes} vs recorded {rec_sizes}; edges {np.round(edges, 3).tolist()}")
    bucket = pd.cut(cdev["cmi"].to_numpy(), bins=edges, include_lowest=True, labels=list(BUCKETS)).astype(str)
    check("Every dev tweet falls in exactly one CMI bucket", set(bucket) == set(BUCKETS) and len(bucket) == 3000,
          f"dev bucket sizes {dict(Counter(bucket))}")
    return dict(dev=dev, uids=uids, y=y, scores=scores, preds=preds, cmi=cdev["cmi"].to_numpy(), bucket=bucket,
                edges=edges, train_bucket_sizes=sizes, checks=checks)


# ============================================================== bootstrap =====
def macro_f1_from_conf(conf):  # conf: (B, 3, 3) rows = true
    # true positives per class for every resample (the diagonal of each confusion matrix)
    tp = np.einsum("bii->bi", conf).astype(float)
    prec = np.divide(tp, conf.sum(1), out=np.zeros_like(tp), where=conf.sum(1) > 0)
    rec = np.divide(tp, conf.sum(2), out=np.zeros_like(tp), where=conf.sum(2) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return f1.mean(1)


def bootstrap_all(y, preds, B=BOOT_B, seed=BOOT_SEED):
    """Paired bootstrap over dev tweets (same resample for every mode and seed). Returns, per mode, arrays of the
    seed-averaged accuracy and macro F1 for each resample."""
    rng = np.random.default_rng(seed)
    n = len(y)
    # each row is one bootstrap sample: n tweets drawn with replacement; the same rows are used for every mode and seed
    idx = rng.integers(0, n, (B, n))
    code = {c: i for i, c in enumerate(LABELS)}
    ycode = np.array([code[v] for v in y])
    out = {}
    # offsets let one bincount call build a 3x3 confusion matrix for every resample at once
    offs = 9 * np.arange(B)[:, None]
    for m in MODES:
        accs, f1s = [], []
        for s in SEEDS:
            pc = np.array([code[v] for v in preds[(m, s)]])
            t, p = ycode[idx], pc[idx]
            # accuracy of this run on every resample
            accs.append((t == p).mean(1))
            conf = np.bincount((t * 3 + p + offs).ravel(), minlength=9 * B).reshape(B, 3, 3)
            f1s.append(macro_f1_from_conf(conf))
        # average the three seeds, so each mode has one score per resample
        out[m] = {"accuracy": np.mean(accs, 0), "macro_f1": np.mean(f1s, 0)}
    return out


def ci(arr):
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def boot_subset_diff(corr_a, corr_b, mask, B=BOOT_B, seed=BOOT_SEED):
    """Bootstrap the difference in seed-averaged accuracy between two modes on the tweets in `mask`."""
    # per-tweet difference in correctness, only for the tweets in the chosen group
    d = (corr_a - corr_b)[mask]
    rng = np.random.default_rng(seed)
    # bootstrap the mean difference inside the group
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), ci(means)


# ================================================================= figures ====
def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "font.size": 10,
    })


def draw_heatmap(ax, conf, annotate_counts=True, title=None, fontsize=12):
    rowpct = 100 * conf / conf.sum(1, keepdims=True)
    im = ax.imshow(rowpct, cmap=CMAP, vmin=0, vmax=100, aspect="equal", interpolation="nearest")
    for i in range(3):
        for j in range(3):
            dark = rowpct[i, j] > 48
            ax.text(j, i - (0.12 if annotate_counts else 0), f"{rowpct[i, j]:.1f}%", ha="center", va="center",
                    fontsize=fontsize, fontweight="bold" if i == j else "normal", color="#ffffff" if dark else INK)
            if annotate_counts:
                ax.text(j, i + 0.22, f"n = {conf[i, j]:,}", ha="center", va="center", fontsize=fontsize - 3.5,
                        color="#e6eefb" if dark else INK2)
    ax.set_xticks(range(3), [c.capitalize() for c in LABELS])
    ax.set_yticks(range(3), [c.capitalize() for c in LABELS])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k in (0.5, 1.5):  # 2px surface gaps between cells (inner lines only, so the outer edge stays clean)
        ax.axhline(k, color=SURFACE, linewidth=3, solid_capstyle="butt")  # butt caps: no stub past the edge
        ax.axvline(k, color=SURFACE, linewidth=3, solid_capstyle="butt")
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
    return im


def fig_confusion_single(mode, conf, pooled, path):
    fig = plt.figure(figsize=(9.6, 4.9), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.30, left=0.085, right=0.975, top=0.80, bottom=0.14)
    fig.text(0.085, 0.945, f"{mode.capitalize()}: BiLSTM confusion matrix on the development set", fontsize=13, fontweight="bold", va="center")
    fig.text(0.085, 0.895, "Row-normalized (% of each true class); counts are summed over seeds 42, 43, 44 (9,000 predictions)",
             fontsize=8.8, color=INK2, va="center")
    ax = fig.add_subplot(gs[0])
    draw_heatmap(ax, conf)
    tx = fig.add_subplot(gs[1])
    tx.axis("off")
    tx.set_xlim(0, 1)
    tx.set_ylim(0, 1)
    tx.text(0, 0.97, "Class-wise precision, recall and F1", fontsize=10.5, fontweight="bold", va="top")
    right_edge = {"Precision": 0.50, "Recall": 0.68, "F1": 0.82, "Support*": 1.0}  # numeric columns are right-aligned
    y0 = 0.82
    tx.text(0, y0, "Class", fontsize=9, color=INK2, va="center", ha="left")
    for h, x in right_edge.items():
        tx.text(x, y0, h, fontsize=9, color=INK2, va="center", ha="right")
    tx.plot([0, 1], [y0 - 0.05, y0 - 0.05], color=AXIS, lw=0.8)
    for k, c in enumerate(LABELS):
        yy = y0 - 0.14 - 0.105 * k
        pc = pooled["per_class"][c]
        tx.text(0, yy, c.capitalize(), fontsize=10, va="center", ha="left")
        for v, x in zip([pc["precision"], pc["recall"], pc["f1"]], list(right_edge.values())[:3]):
            tx.text(x, yy, f"{v:.3f}", fontsize=10, va="center", ha="right")
        tx.text(1.0, yy, f"{pc['support']:,}", fontsize=10, va="center", ha="right", color=INK2)
    ysep = y0 - 0.14 - 0.105 * 2 - 0.07
    tx.plot([0, 1], [ysep, ysep], color=GRID, lw=0.8)
    summary = [("Macro avg", pooled["macro_precision"], pooled["macro_recall"], pooled["macro_f1"]),
               ("Weighted avg", pooled["weighted_precision"], pooled["weighted_recall"], pooled["weighted_f1"])]
    for k, (name, p, r, f) in enumerate(summary):
        yk = ysep - 0.08 - 0.10 * k
        tx.text(0, yk, name, fontsize=10, va="center", ha="left", color=INK2)
        for v, x in zip([p, r, f], list(right_edge.values())[:3]):
            tx.text(x, yk, f"{v:.3f}", fontsize=10, va="center", ha="right", color=INK2)
    tx.text(0, ysep - 0.08 - 0.10 * 2, "Accuracy", fontsize=10, va="center", ha="left", fontweight="bold")
    tx.text(0.50, ysep - 0.08 - 0.10 * 2, f"{pooled['accuracy']:.3f}", fontsize=10, va="center", ha="right", fontweight="bold")
    tx.text(0, 0.0, "*Support = true tweets summed over the 3 seeds (3 x the dev class size).\nPooled over the three seeds; the per-seed mean ± std is in the report.",
            fontsize=7.8, color=INK2, va="bottom")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_confusion_all(confs, summaries, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2), dpi=300)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.78, bottom=0.24, wspace=0.30)
    im = None
    for ax, m in zip(axes, MODES):
        im = draw_heatmap(ax, confs[m], title=m.capitalize(), fontsize=11)
        ax.text(0.5, -0.30, f"macro F1 {summaries[m]['macro_f1']:.3f}  ·  weighted F1 {summaries[m]['weighted_f1']:.3f}  ·  accuracy {summaries[m]['accuracy']:.3f}",
                transform=ax.transAxes, fontsize=9, color=INK2, ha="center", va="top")
    cax = fig.add_axes([0.915, 0.30, 0.011, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("% of the true class", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.text(0.07, 0.94, "BiLSTM confusion matrices on the development set, by preprocessing mode", fontsize=13, fontweight="bold", va="center")
    fig.text(0.07, 0.895, "Row-normalized; counts summed over seeds 42, 43, 44 (9,000 predictions per mode); one shared color scale. "
             "Scores under each panel are computed on the pooled predictions.", fontsize=9, color=INK2, va="center")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_cmi(bucket_stats, bucket_n, bucket_labels, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.17, wspace=0.17)
    offs = {"raw": -0.22, "clean": 0.0, "normalized": 0.22}
    for ax, key, ttl in ((axes[0], "accuracy", "Accuracy"), (axes[1], "macro_f1", "Macro F1")):
        lo, hi = 1, 0
        for m in MODES:
            for b in BUCKETS:
                v = np.array(bucket_stats[(m, b)][key])
                lo, hi = min(lo, v.min()), max(hi, v.max())
        pad = 0.012
        ax.set_ylim(np.floor((lo - pad) * 100) / 100, np.ceil((hi + 2.2 * pad) * 100) / 100)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for m in MODES:
            for bi, b in enumerate(BUCKETS):
                v = np.array(bucket_stats[(m, b)][key])
                x = bi + offs[m]
                ax.plot([x, x], [v.min(), v.max()], color=MODE_COLOR[m], lw=2, solid_capstyle="round", zorder=2)
                ax.scatter([x] * 3, v, s=14, color=MODE_COLOR[m], alpha=0.55, zorder=3, marker=MODE_MARKER[m], linewidths=0)
                ax.scatter([x], [v.mean()], s=64, color=MODE_COLOR[m], edgecolor=SURFACE, linewidth=1.6, zorder=4, marker=MODE_MARKER[m])
        for m in MODES:  # direct labels in the first bucket, in ink (not series color); 'normalized' goes below its range
            v = np.array(bucket_stats[(m, "low")][key])
            if m == "normalized":
                ax.text(0 + offs[m], v.min() - 0.0035, m, fontsize=8.8, color=INK2, va="top", ha="center")
            else:
                ax.text(0 + offs[m], v.max() + 0.0035, m, fontsize=8.8, color=INK2, va="bottom", ha="center")
        ax.set_xticks(range(3), [f"{bucket_labels[b]}\nn = {bucket_n[b]:,}" for b in BUCKETS])
        ax.set_xlim(-0.55, 2.55)
        ax.set_title(ttl, loc="left", fontsize=11.5, fontweight="bold", pad=8)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Score on the dev tweets of the bucket\n(y-axis does not start at 0)")
    fig.text(0.075, 0.945, "BiLSTM performance by code-mixing (CMI) bucket on the development set", fontsize=13, fontweight="bold", va="center")
    fig.text(0.075, 0.895, "Large marker = mean of seeds 42, 43, 44; small markers = the three seeds; line = seed range. Buckets are the train CMI terciles.",
             fontsize=9, color=INK2, va="center")
    handles = [plt.Line2D([0], [0], marker=MODE_MARKER[m], color=MODE_COLOR[m], lw=2, markersize=8, label=m) for m in MODES]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.985, 0.985), fontsize=9.5)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ================================================================= report =====
def build(data_root: Path, overwrite: bool):
    outputs = [REPORT_PATH] + [FIG_DIR / n for n in FIG_NAMES.values()]
    # do not overwrite an earlier report or figures unless asked to
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not overwrite:
        sys.exit("refusing to overwrite existing outputs (use --overwrite):\n  " + "\n  ".join(existing))
    D = load_everything(data_root)
    dev, uids, y, preds, scores, cmi, bucket = D["dev"], D["uids"], D["y"], D["preds"], D["scores"], D["cmi"], D["bucket"]
    n = len(y)
    txt = dev["text"].to_numpy()
    L = []  # report lines
    add = L.append

    # ----- per-seed, pooled metrics
    seed_m = {(m, s): metrics_of(y, preds[(m, s)]) for m in MODES for s in SEEDS}
    pooled_true = np.concatenate([y] * 3)
    pooled_m, conf_pooled, conf_seed = {}, {}, {}
    for m in MODES:
        pooled_pred = np.concatenate([preds[(m, s)] for s in SEEDS])
        pooled_m[m] = metrics_of(pooled_true, pooled_pred)
        conf_pooled[m] = confusion_counts(pooled_true, pooled_pred)
        for s in SEEDS:
            conf_seed[(m, s)] = confusion_counts(y, preds[(m, s)])
    # cross-check the confusion matrices with the matrices stored in the JSONs
    stored_ok = all((np.array(scores[(m, s)]["per_seed"][0]["confusion"]) == conf_seed[(m, s)]).all() for m in MODES for s in SEEDS)
    D["checks"].append(("Confusion matrices recomputed from predictions equal those stored in the score JSONs", stored_ok, ""))

    # for every mode and seed: was each dev tweet predicted correctly?
    correct = {m: np.stack([preds[(m, s)] == y for s in SEEDS]) for m in MODES}  # (3 seeds, n)
    corr_mean = {m: correct[m].mean(0) for m in MODES}  # seed-averaged per-tweet correctness
    # how many of the nine runs got each tweet wrong (0 to 9)
    errs9 = sum((~correct[m]).sum(0) for m in MODES)

    def agg(key, m):
        return [seed_m[(m, s)][key] for s in SEEDS]

    def agg_pc(m, c, k):
        return [seed_m[(m, s)]["per_class"][c][k] for s in SEEDS]

    # paired bootstrap over the dev tweets (see BOOT_B and BOOT_SEED at the top of the file)
    boot = bootstrap_all(y, preds)

    # ================= title / summary placeholder (filled at the end)
    add("# BiLSTM word + character model: development-set results analysis")
    add("")
    add("*Raw vs clean vs normalized preprocessing, seeds 42 / 43 / 44. Development set only. Generated by "
        "`python -m src.analyze_bilstm_results`; every number below is computed from the existing experiment outputs.*")
    add("")
    add("> **Content note.** The quoted tweets come from real social-media data and include political arguments and abusive language.")
    add("")
    summary_slot = len(L)
    add("")  # placeholder for the summary block

    # ================= 1. scope
    add("## 1. Scope, sources and restrictions")
    add("")
    add("**Inputs (all pre-existing, read-only):** nine score files `results/bilstm_full_<mode>_s<seed>_dev_scores.json`, nine prediction files "
        "`results/predictions/bilstm_full_<mode>_s<seed>_seed<seed>_dev.csv` (`uid,label,pred`), Mani's per-tweet CMI file `results/cmi_per_tweet.csv` "
        "(**train and dev rows only**; for its 3,000 test rows only the `split` field was inspected so that they could be skipped, and their labels and CMI values were not parsed or used), `results/eda_part2_stats.json`, `results/tfidf_dev_scores.json`, "
        "and the official dev file for tweet text and gold labels.")
    add("")
    add("**Not done:** no model was trained or re-trained; the test split and test labels were not used; preprocessing, model code, hyper-parameters, "
        "CMI values and the CMI methodology were not changed; no existing result or prediction file was modified.")
    add("")
    add("**Missing field, handled explicitly:** the score JSONs store accuracy, macro/weighted F1 and per-class precision/recall/F1/support, but **not** macro or "
        "weighted precision and recall. These are derived here from the prediction files (and the derivation is checked: recomputed accuracy, F1 and per-class "
        "values equal the stored ones to 1e-9, see section 2). Weighted recall equals accuracy by definition.")
    add("")

    # ================= 2. verification
    add("## 2. Verification of inputs (Task 5)")
    add("")
    nfail = sum(1 for _, ok, _ in D["checks"] if not ok)
    add(f"**{len(D['checks']) - nfail} of {len(D['checks'])} checks passed" + (f"; {nfail} FAILED, see below.**" if nfail else ".**"))
    add("")
    # collapse the per-run checks into groups for readability
    groups = {}
    for desc, ok, detail in D["checks"]:
        key = desc.split(":", 1)[1].strip() if desc.startswith("full_") else desc
        groups.setdefault(key, []).append((ok, detail))
    rows = []
    for key, items in groups.items():
        okall = all(o for o, _ in items)
        det = "; ".join(sorted({d for _, d in items if d}))[:230]
        rows.append(("PASS" if okall else "**FAIL**", f"{key}" + (f" (x{len(items)} runs)" if len(items) > 1 else ""), md_cell(det)))
    for line in table(["Result", "Check", "Detail"], rows):
        add(line)
    add("")
    add("All joins between predictions, gold labels, tweet text and CMI values use the **uid**, never row position (the prediction files happen to be in dev-file "
        "order, but nothing relies on it).")
    add("")

    # ================= 3. consolidated comparison
    add("## 3. Consolidated comparison (Task 1)")
    add("")
    add("### 3.1 Mean ± sample standard deviation across the three seeds")
    add("")
    add("For each seed the metric is computed on the 3,000 dev tweets; the table shows mean ± sample std (n = 3, ddof = 1) of those three per-seed values. "
        "**Weighted recall equals accuracy.**")
    add("")
    hdr = ["Mode", "Accuracy", "Macro precision", "Macro recall", "**Macro F1**", "Weighted precision", "**Weighted F1**"]
    rows = [[m, ms(agg("accuracy", m)), ms(agg("macro_precision", m)), ms(agg("macro_recall", m)), "**" + ms(agg("macro_f1", m)) + "**",
             ms(agg("weighted_precision", m)), "**" + ms(agg("weighted_f1", m)) + "**"] for m in MODES]
    for line in table(hdr, rows):
        add(line)
    add("")
    add("### 3.2 Metrics on pooled predictions")
    add("")
    add("Here the three seeds' predictions are **concatenated** (3 x 3,000 = 9,000 predictions per mode) and each metric is computed **once** on the pooled set. "
        "This is a different quantity from section 3.1: it has no seed-to-seed spread, and the 9,000 predictions are **not independent** (the same 3,000 tweets "
        "appear three times), so it must not be used to compute confidence intervals.")
    add("")
    rows = [[m, fmt(pooled_m[m]["accuracy"]), fmt(pooled_m[m]["macro_precision"]), fmt(pooled_m[m]["macro_recall"]), fmt(pooled_m[m]["macro_f1"]),
             fmt(pooled_m[m]["weighted_precision"]), fmt(pooled_m[m]["weighted_f1"])] for m in MODES]
    for line in table(["Mode", "Accuracy", "Macro precision", "Macro recall", "Macro F1", "Weighted precision", "Weighted F1"], rows):
        add(line)
    add("")
    add("### 3.3 How the two aggregations differ")
    add("")
    rows = []
    for m in MODES:
        for key, name in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")):
            a, p = float(np.mean(agg(key, m))), pooled_m[m][key]
            rows.append([m, name, fmt(a), fmt(p), f"{p - a:+.5f}"])
    for line in table(["Mode", "Metric", "Mean of per-seed values", "Pooled predictions", "Pooled - mean"], rows):
        add(line)
    add("")
    add("Accuracy is identical in both ways (every seed has the same number of tweets). F1 is not linear in the predictions, so pooled and averaged F1 differ "
        "slightly (here by at most "
        f"{max(abs(pooled_m[m][k] - np.mean(agg(k, m))) for m in MODES for k in ('macro_f1', 'weighted_f1')):.5f}). The headline numbers in this report are the "
        "**mean of per-seed metrics** (3.1), because they come with a seed spread.")
    add("")
    add("### 3.4 Class-wise precision, recall and F1 (mean ± std over seeds)")
    add("")
    rows = []
    for m in MODES:
        for c in LABELS:
            rows.append([m, c, scores[(m, 42)]["per_seed"][0]["metrics"]["per_class"][c]["support"], ms(agg_pc(m, c, "precision")),
                         ms(agg_pc(m, c, "recall")), ms(agg_pc(m, c, "f1"))])
    for line in table(["Mode", "Class", "Dev support", "Precision", "Recall", "F1"], rows):
        add(line)
    add("")
    add("### 3.5 Every run")
    add("")
    rows = []
    for m in MODES:
        for s in SEEDS:
            r = scores[(m, s)]["per_seed"][0]
            mm = seed_m[(m, s)]
            rows.append([f"full_{m}_s{s}", fmt(mm["accuracy"]), fmt(mm["macro_precision"]), fmt(mm["macro_recall"]), fmt(mm["macro_f1"]),
                         fmt(mm["weighted_precision"]), fmt(mm["weighted_f1"]), f"{r['best_epoch']} / {r['epochs_run']}", f"{r['seconds'] / 60:.0f}"])
    for line in table(["Run", "Accuracy", "Macro P", "Macro R", "Macro F1", "Weighted P", "Weighted F1", "Best / run epochs", "Train min"], rows):
        add(line)
    add("")
    add("### 3.6 Are the differences between modes larger than the noise?")
    add("")
    add("Two sources of uncertainty, kept apart:")
    add("")
    add("1. **Seed-to-seed variation** (section 3.1 standard deviations; only 3 seeds, so the std itself is rough).")
    add(f"2. **Sampling variation of the 3,000 dev tweets.** A paired bootstrap ({BOOT_B:,} resamples of the dev tweets, fixed seed; the same resample is applied to "
        "every mode and seed) gives a 95% interval for the difference between modes of the seed-averaged metric. It reflects which tweets happen to be in the dev "
        "set, **not** seed variation.")
    add("")
    rows = []
    pairs = [("clean", "raw"), ("normalized", "raw"), ("normalized", "clean")]
    for a, b in pairs:
        for key, name in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1")):
            d = boot[a][key] - boot[b][key]
            pt = float(np.mean(agg(key, a)) - np.mean(agg(key, b)))
            lo, hi = ci(d)
            per_seed = [seed_m[(a, s)][key] - seed_m[(b, s)][key] for s in SEEDS]
            sign = sum(1 for v in per_seed if v > 0)
            rows.append([f"{a} - {b}", name, f"{pt:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]", "includes 0" if lo <= 0 <= hi else "excludes 0",
                         " / ".join(f"{v:+.4f}" for v in per_seed), f"{sign} of 3"])
    for line in table(["Difference", "Metric", "Mean difference", "95% bootstrap interval (dev sampling)", "Interval vs 0", "Per seed (42 / 43 / 44)", "Seeds where first mode is higher"], rows):
        add(line)
    add("")
    nc_acc = [seed_m[("normalized", s)]["accuracy"] - seed_m[("clean", s)]["accuracy"] for s in SEEDS]
    lo_nc, hi_nc = ci(boot["normalized"]["accuracy"] - boot["clean"]["accuracy"])
    add(f"**Reading the table.** " + ("Every interval includes 0, so no difference between modes is established. " if all(r[4] == "includes 0" for r in rows) else "") +
        f"The most consistent pattern is that **normalized is lower than clean in {sum(1 for v in nc_acc if v < 0)} of 3 seeds** (accuracy {np.mean(nc_acc):+.4f}, "
        f"95% interval [{lo_nc:+.4f}, {hi_nc:+.4f}]); clean vs raw changes sign between seeds. The differences are of the same size as the seed-to-seed spread of a single mode.")
    add("")
    add("The seeds share the dev set and the split but **not** the random initialisation (the vocabularies, hence the embedding tables, differ between modes), so the "
        "per-seed differences are descriptive and not a formal paired test.")
    add("")
    tf = json.loads((ROOT / "results" / "tfidf_dev_scores.json").read_text(encoding="utf-8"))
    add("### 3.7 Reference: the TF-IDF baselines (existing results, dev set)")
    add("")
    rows = []
    for key, name in (("word_1-2", "TF-IDF word 1-2"), ("char_2-5", "TF-IDF char 2-5"), ("word+char", "TF-IDF word+char")):
        b = tf[key]["best_dev"]
        rows.append([name, fmt(b["accuracy"]), fmt(b["macro_f1"]), fmt(b["weighted_f1"])])
    rows.append(["Majority class (neutral)", fmt(tf["majority_baseline"]["accuracy"]), "-", fmt(tf["majority_baseline"]["weighted_f1"])])
    for line in table(["Model", "Accuracy", "Macro F1", "Weighted F1"], rows):
        add(line)
    add("")
    add("These are single deterministic runs whose regularisation strength was chosen on dev (a mild optimistic bias, as for the BiLSTM's best epoch).")
    add("")

    # ================= 4. confusion matrices
    add("## 4. Confusion matrices (Task 2)")
    add("")
    add("**Aggregation method.** For each mode the three seeds' predictions are pooled, and the matrix is the **sum of the three per-seed 3x3 count matrices** "
        "(9,000 predictions; each row sums to 3 x the dev class size). Cells are shown as the percentage of the true class (row-normalised) with the pooled count. "
        "The class-wise precision, recall and F1 printed in each figure are computed on the same pooled predictions; section 3.4 gives the per-seed mean ± std. "
        "Per-seed matrices are in Appendix A. Figures use one shared color scale (0-100% of the true class).")
    add("")
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for m in MODES:
        fig_confusion_single(m, conf_pooled[m], pooled_m[m], FIG_DIR / FIG_NAMES[m])
    fig_confusion_all(conf_pooled, pooled_m, FIG_DIR / FIG_NAMES["all"])
    add("![All three modes](../results/figures/" + FIG_NAMES["all"] + ")")
    add("")
    for m in MODES:
        add(f"**{m.capitalize()}** (`results/figures/{FIG_NAMES[m]}`)")
        add("")
        add(f"![{m} confusion matrix](../results/figures/{FIG_NAMES[m]})")
        add("")
        rows = []
        for i, c in enumerate(LABELS):
            tot = conf_pooled[m][i].sum()
            rows.append([f"true {c}", *[f"{conf_pooled[m][i, j]:,} ({100 * conf_pooled[m][i, j] / tot:.1f}%)" for j in range(3)], f"{tot:,}"])
        for line in table(["", "pred negative", "pred neutral", "pred positive", "Row total"], rows):
            add(line)
        add("")
    pair_rows = {}
    for m in MODES:
        cm = conf_pooled[m]
        err = cm.sum() - np.trace(cm)
        neu = cm[1, 0] + cm[1, 2] + cm[0, 1] + cm[2, 1]
        pair_rows[m] = (err, neu, cm[0, 2] + cm[2, 0])
    add("**What the matrices show** (pooled over seeds):")
    add("")
    for m in MODES:
        err, neu, flip = pair_rows[m]
        cm = conf_pooled[m]
        add(f"- **{m}:** {err:,} of 9,000 predictions are wrong; {100 * neu / err:.0f}% of the errors involve the neutral class and only "
            f"{100 * flip / err:.0f}% are direct negative <-> positive flips. Recall by class: negative {100 * cm[0, 0] / cm[0].sum():.1f}%, "
            f"neutral {100 * cm[1, 1] / cm[1].sum():.1f}%, positive {100 * cm[2, 2] / cm[2].sum():.1f}%.")
    neu_low = all(all(seed_m[(m, s)]["per_class"]["neutral"][k] < min(seed_m[(m, s)]["per_class"][c][k] for c in ("negative", "positive")) for s in SEEDS) for m in MODES for k in ("precision", "recall", "f1"))
    add("- " + ("In every run of every mode **neutral has the lowest precision, recall and F1**" if neu_low else "**Neutral is the hardest class** in most runs")
        + ": polar tweets are often read as neutral (about a third of the negative tweets and 25-30% of the positive tweets), and neutral tweets are spread over both polar classes.")
    add("")

    # ================= 5. error analysis
    add("## 5. Error analysis (Task 3, BiLSTM-specific and preliminary)")
    add("")
    add("*This is a BiLSTM-specific preliminary analysis. Akansha is responsible for the team's broader categorised error analysis.*")
    add("")
    add("**Method and caveats.**")
    add("")
    add("- Predictions come from the nine real dev prediction files. A tweet is an *error* for a run if its predicted label differs from the gold label.")
    add("- Categories are **heuristic and transparent**: negation = the cleaned tweet contains one of "
        + ", ".join(f"`{t}`" for t in sorted(NEG_TOKENS)) + " or a word ending in `n't` (a crude list; `na` and `no` are ambiguous, so this over-counts); "
        "spelling variation = the tweet text is changed by the normalization step of `src.preprocess`; code-mixing = the CMI bucket of section 6.")
    add("- **Sarcasm, ambiguity and possible label inconsistency cannot be detected automatically.** For these the *tweets* are real and selected by simple "
        "rules (e.g. gold negative but positive-sounding words and predicted positive), but the *interpretation* (\"reads as sarcastic\") is **my reading**, "
        "not an annotation, and the gold labels themselves are noisy (the SemEval annotators agreed only 55% of the time, see `docs/problem_statement.md`).")
    add("- Every example below is a real dev tweet; its uid, gold label, predictions and CMI are read from the data. The tweet text is the dev file's tokenised text "
        "(tokens joined by spaces).")
    add("")
    # 5.2 error structure
    add("### 5.1 Overall error structure (pooled over seeds)")
    add("")
    rows = []
    for m in MODES:
        cm = conf_pooled[m]
        err = int(cm.sum() - np.trace(cm))
        pairs_ = sorted([((LABELS[i], LABELS[j]), int(cm[i, j])) for i in range(3) for j in range(3) if i != j], key=lambda t: -t[1])
        rows.append([m, f"{err:,}"] + [f"{a}->{b}: {c:,} ({100 * c / err:.0f}%)" for (a, b), c in pairs_])
    for line in table(["Mode", "Errors", "1st", "2nd", "3rd", "4th", "5th", "6th"], rows):
        add(line)
    add("")
    add("Recurring pattern 1: **errors are dominated by the neutral boundary.** Polar tweets read as neutral (negative->neutral, positive->neutral) and neutral tweets "
        "read as polar account for about nine of ten errors in every mode, while direct negative <-> positive flips are rare.")
    add("")
    # 5.2 consistency
    gold_hard = Counter(y[errs9 == 9])
    easy = errs9 == 0
    add("### 5.2 Consistently hard vs unstable tweets")
    add("")
    add("Each tweet has 9 predictions (3 modes x 3 seeds).")
    add("")
    dist = Counter(errs9.tolist())
    add(f"- **Right in all 9 runs: {int(easy.sum()):,} tweets ({100 * easy.mean():.0f}%).**")
    add(f"- **Wrong in all 9 runs: {int((errs9 == 9).sum()):,} tweets ({100 * (errs9 == 9).mean():.0f}%)** - gold labels: {counts_text(gold_hard, LABELS)}. These are the "
        "tweets no run can get right: hard cases, or cases where the gold label is debatable.")
    add(f"- **Wrong in 1-8 runs: {int(((errs9 > 0) & (errs9 < 9)).sum()):,} tweets ({100 * ((errs9 > 0) & (errs9 < 9)).mean():.0f}%)**: the runs disagree with "
        "each other, which is a measure of how borderline the sentiment is. Their number of wrong runs: "
        + ", ".join(f"{k}: {dist[k]}" for k in range(1, 9)) + ".")
    add("")
    add("Recurring pattern 2: **a large share of the errors are shared by all nine runs**, so they are not caused by seed noise or by the choice of preprocessing "
        "mode; they reflect genuinely hard tweets, limits of this model, or debatable gold labels (this analysis cannot tell which). A further large group flips between runs.")
    add("")
    # 5.3 categories
    clean_txt = load_processed_split("dev", mode="clean", root=data_root)["text_proc"].to_numpy()
    norm_txt = load_processed_split("dev", mode="normalized", root=data_root)["text_proc"].to_numpy()
    rep_txt = np.array([normalize_repeated_chars(t) for t in clean_txt])
    neg_flag = np.array([any(w.lower() in NEG_TOKENS or w.lower().endswith("n't") for w in t.split()) for t in clean_txt])
    changed = clean_txt != norm_txt
    rep_changed = clean_txt != rep_txt
    dict_changed = rep_txt != norm_txt
    cut_flag = np.array([t.rstrip().endswith("…") for t in txt])
    add("### 5.3 Accuracy by tweet property")
    add("")
    add("Accuracy averaged over the three seeds of each mode (n = number of dev tweets). Gold-label mixes differ between subsets, so differences are descriptive.")
    add("")
    subsets = [("All dev tweets", np.ones(n, bool)), ("Contains a negation token", neg_flag), ("No negation token", ~neg_flag),
               ("Text changed by normalization", changed), ("Text not changed by normalization", ~changed), ("Tweet cut off (ends with ...)", cut_flag),
               ("CMI low", bucket == "low"), ("CMI mid", bucket == "mid"), ("CMI high", bucket == "high")]
    rows = [[name, f"{int(mask.sum()):,}", *[fmt(correct[m][:, mask].mean()) for m in MODES]] for name, mask in subsets]
    for line in table(["Subset", "n", "raw", "clean", "normalized"], rows):
        add(line)
    add("")
    neg_gap = np.mean([correct[m][:, neg_flag].mean() - correct[m][:, ~neg_flag].mean() for m in MODES])
    add(f"Recurring pattern 3: **tweets with a negation token are harder** (about {100 * abs(neg_gap):.1f} accuracy points lower than tweets without one, on average "
        f"over the modes). Part of this gap may come from the different label mix of these tweets (negation tweets: {counts_text(Counter(y[neg_flag]), LABELS)}), so it is "
        "not proof that negation is mishandled by itself.")
    add("")
    # 5.4 spelling analysis
    add("### 5.4 Spelling variation: does normalization help where it changes the text?")
    add("")
    add("Normalization changes the text of some dev tweets in two ways: the **repeated-letter** step (`yaaar` -> `yaar`) and the **Hinglish-dictionary** step "
        "(`nhi` -> `nahi`). Tweets whose text is identical in all three modes act as a control. Accuracy is averaged over seeds; the bootstrap interval is for "
        "*normalized minus raw* on that subset (tweet sampling only).")
    add("")
    unchanged = ~changed
    rows = []
    sub_boot = {}
    for name, mask in (("Hinglish-dictionary step changed the text", dict_changed), ("Repeated-letter step changed the text", rep_changed),
                       ("Any change (dictionary or repeats)", changed), ("Text identical in all three modes (control)", unchanged)):
        d, (lo, hi) = boot_subset_diff(corr_mean["normalized"], corr_mean["raw"], mask)
        sub_boot[name] = (d, lo, hi)
        rows.append([name, f"{int(mask.sum()):,}", *[fmt(correct[m][:, mask].mean()) for m in MODES], f"{d:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"])
    for line in table(["Subset", "n", "raw", "clean", "normalized", "normalized - raw", "95% interval"], rows):
        add(line)
    add("")
    vc = Counter()
    for a, b in zip(rep_txt, norm_txt):
        for x, z in zip(a.split(), b.split()):
            if x != z:
                vc[(x.lower(), z.lower())] += 1
    add("Most frequent dictionary rewrites in dev: " + ", ".join(f"`{a}`->`{b}` ({c})" for (a, b), c in vc.most_common(8)) + ".")
    add("")
    helped = changed & (~correct["raw"]).all(0) & correct["normalized"].all(0)
    hurt = changed & correct["raw"].all(0) & (~correct["normalized"]).all(0)
    flips = []
    for si, s in enumerate(SEEDS):
        a, b = correct["raw"][si], correct["normalized"][si]
        flips.append((int((a & ~b).sum()), int((~a & b).sum())))
    add(f"Tweet-level flips between raw and normalized over **all** 3,000 tweets, per seed (raw right & normalized wrong / raw wrong & normalized right): "
        + "; ".join(f"seed {s}: {a} / {b}" for s, (a, b) in zip(SEEDS, flips)) +
        f". Among the changed tweets, normalization turned a wrong prediction into a right one in all 3 seeds for {int(helped.sum())} tweets and the reverse for "
        f"{int(hurt.sum())} tweets.")
    add("")
    tr_raw = load_split("train", root=data_root)["text"].str.lower()
    cnt = {w: int(tr_raw.str.split().apply(lambda toks, w=w: toks.count(w)).sum()) for w in ("bakvas", "bakwas")}
    add(f"Spelling frequency evidence for one example below: in the **train** text `bakwas` occurs {cnt['bakwas']} times and `bakvas` {cnt['bakvas']} times.")
    add("")
    dd = sub_boot["Hinglish-dictionary step changed the text"]
    rr = sub_boot["Repeated-letter step changed the text"]
    uu = sub_boot["Text identical in all three modes (control)"]
    add("Recurring pattern 4 (spelling):")
    add("")
    add(f"- On tweets where the **dictionary** step changed the text, normalized is {dd[0] * 100:+.1f} points vs raw (interval [{dd[1] * 100:+.1f}, {dd[2] * 100:+.1f}]).")
    add(f"- On tweets where the **repeated-letter** step changed the text, normalized is {rr[0] * 100:+.1f} points vs raw (interval [{rr[1] * 100:+.1f}, {rr[2] * 100:+.1f}]); "
        "elongation such as `Yeeeee` may carry emphasis, which this step removes (a hypothesis, not tested here).")
    add(f"- On the **control** tweets (identical text in all modes) normalized still differs from raw by {uu[0] * 100:+.1f} points (interval [{uu[1] * 100:+.1f}, {uu[2] * 100:+.1f}]). "
        "Because the text is the same, that difference comes from the different **training** (vocabulary, initialisation), not from the text change, and it sets the "
        "scale of differences that should *not* be interpreted.")
    add("")
    # 5.5 examples
    add("### 5.5 Real examples")
    add("")
    add("Predictions are the **majority label over the three seeds** of each mode, with the vote count in brackets (3/3 = all seeds agree). "
        "\"Wrong in x of 9\" counts all nine runs. 'My reading' is an interpretation, not an annotation.")
    add("")
    uid_index = {u: i for i, u in enumerate(uids)}
    for cat, items in EXAMPLES.items():
        add(f"#### {cat}")
        add("")
        rows = []
        for uid, note in items:
            i = uid_index[uid]
            votes = []
            for m in MODES:
                cc = Counter(preds[(m, s)][i] for s in SEEDS)
                top = cc.most_common(1)[0]
                votes.append(f"{top[0]} ({top[1]}/3)" if top[1] >= 2 else "split: " + "/".join(sorted(cc.elements())))
            rows.append([uid, y[i], " / ".join(votes), f"{int(errs9[i])} of 9", f"{cmi[i]:.0f} ({bucket[i]})", md_cell(txt[i]), md_cell(note)])
        for line in table(["uid", "Gold", "Predicted raw / clean / normalized", "Wrong in", "CMI (bucket)", "Tweet (dev text)", "Why it is hard (my reading)"], rows):
            add(line)
        add("")
    add("Recurring pattern 5: **positive-sounding words in negative tweets** (irony or sarcasm) lead to positive predictions; **laughter emojis** are not reliable positive "
        "evidence; **negated polar words** are often misread; and several **neutral-gold tweets contain explicit positive wording**, which the models (reasonably) "
        "call positive, a sign of annotation inconsistency or an unclear neutral convention.")
    add("")

    # ================= 6. CMI analysis
    add("## 6. CMI analysis (Task 4, BiLSTM-specific supporting analysis for Mani)")
    add("")
    add("**Bucket definition (existing, not changed).** The low / mid / high buckets are the equal-count **terciles of the train CMI** from Mani's `src/eda_part2.py` "
        "(`pd.qcut(train cmi, 3)`; `docs/eda_part2.md` calls them *candidate buckets*; no later definition exists in the repository). Their edges, computed from the existing "
        f"per-tweet train CMI values, are **{D['edges'][1]:.3f}** and **{D['edges'][2]:.3f}**, giving train sizes {D['train_bucket_sizes']} - exactly the sizes and ranges "
        "Mani recorded (low 0-23.8, mid 24.0-34.8, high 35.0-50.0). The same edges were applied to the dev CMI values; CMI itself was **not** recomputed, and the dev "
        "tweets were joined to predictions by uid.")
    add("")
    bn = {b: int((bucket == b).sum()) for b in BUCKETS}
    blab = {"low": f"low (CMI ≤ {D['edges'][1]:.1f})", "mid": "mid", "high": f"high (CMI > {D['edges'][2]:.1f})"}
    add("Dev bucket sizes: " + ", ".join(f"{b} {bn[b]:,}" for b in BUCKETS) + ". Dev gold-label mix per bucket (negative / neutral / positive): "
        + "; ".join(f"{b}: " + " / ".join(str(int(((bucket == b) & (y == c)).sum())) for c in LABELS) for b in BUCKETS) + ".")
    add("")
    bstat = {}
    rows = []
    for m in MODES:
        for b in BUCKETS:
            mask = bucket == b
            per = [metrics_of(y[mask], preds[(m, s)][mask]) for s in SEEDS]
            pooled = metrics_of(np.concatenate([y[mask]] * 3), np.concatenate([preds[(m, s)][mask] for s in SEEDS]))
            bstat[(m, b)] = {"accuracy": [p["accuracy"] for p in per], "macro_f1": [p["macro_f1"] for p in per]}
            acc_mean = float(np.mean([p["accuracy"] for p in per]))
            se = float(np.sqrt(acc_mean * (1 - acc_mean) / mask.sum()))
            rows.append([m, b, f"{int(mask.sum()):,}", ms([p["accuracy"] for p in per]), ms([p["macro_f1"] for p in per]),
                         ms([p["weighted_f1"] for p in per]), ms([p["macro_precision"] for p in per]), ms([p["macro_recall"] for p in per]),
                         fmt(pooled["accuracy"]), f"±{1.96 * se:.3f}"])
    add("### 6.1 Metrics per bucket and mode (mean ± std over the three seeds)")
    add("")
    for line in table(["Mode", "Bucket", "n", "Accuracy", "Macro F1", "Weighted F1", "Macro precision", "Macro recall", "Pooled accuracy", "95% sampling margin (accuracy)"], rows):
        add(line)
    add("")
    add("The last column is the binomial margin 1.96 x sqrt(p(1-p)/n) for the bucket's accuracy: with about 1,000 tweets per bucket, differences smaller than roughly "
        "0.02-0.03 between buckets or modes cannot be told apart from sampling noise. Macro metrics also depend on each bucket's class mix.")
    add("")
    fig_cmi(bstat, bn, blab, FIG_DIR / FIG_NAMES["cmi"])
    add(f"![CMI buckets](../results/figures/{FIG_NAMES['cmi']})")
    add("")
    add("### 6.2 Within-mode trend and between-mode differences")
    add("")
    rows = []
    within_flags = []
    for m in MODES:
        for hi_b, lo_b in (("high", "low"),):
            ch, cl = corr_mean[m][bucket == hi_b], corr_mean[m][bucket == lo_b]
            rng = np.random.default_rng(BOOT_SEED)
            bh = ch[rng.integers(0, len(ch), (BOOT_B, len(ch)))].mean(1)
            bl = cl[rng.integers(0, len(cl), (BOOT_B, len(cl)))].mean(1)
            lo, hi = ci(bh - bl)
            within_flags.append(lo <= 0 <= hi)
            rows.append([m, f"{hi_b} - {lo_b}", f"{ch.mean() - cl.mean():+.4f}", f"[{lo:+.4f}, {hi:+.4f}]", "includes 0" if lo <= 0 <= hi else "excludes 0"])
    for line in table(["Mode", "Difference in accuracy", "Mean difference", "95% bootstrap interval", "Interval vs 0"], rows):
        add(line)
    add("")
    rows = []
    for b in BUCKETS:
        mask = bucket == b
        for a_, b_ in pairs:
            d, (lo, hi) = boot_subset_diff(corr_mean[a_], corr_mean[b_], mask)
            rows.append([b, f"{a_} - {b_}", f"{d:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]", "includes 0" if lo <= 0 <= hi else "excludes 0"])
    for line in table(["Bucket", "Modes", "Accuracy difference", "95% bootstrap interval", "Interval vs 0"], rows):
        add(line)
    add("")
    hard_by_b = {b: int(((errs9 == 9) & (bucket == b)).sum()) for b in BUCKETS}
    excl = []
    for b in BUCKETS:
        for a_, b_ in pairs:
            d, (lo, hi) = boot_subset_diff(corr_mean[a_], corr_mean[b_], bucket == b)
            if not (lo <= 0 <= hi):
                excl.append((b, a_, b_, d, lo, hi))
    low_norm_below_clean = max(bstat[("normalized", "low")]["accuracy"]) < min(bstat[("clean", "low")]["accuracy"])
    add("**Within-mode trend:** " + ("no mode shows a clear change with code-mixing; every high-minus-low interval above includes 0 (the point estimates differ in sign: "
        + ", ".join(f"{m} {float(corr_mean[m][bucket == 'high'].mean() - corr_mean[m][bucket == 'low'].mean()):+.3f}" for m in MODES) + ")."
        if all(within_flags) else "at least one mode shows a high-minus-low difference whose interval excludes 0 (see the table)."))
    add("")
    if excl:
        add(f"**{len(excl)} of the nine bucket-level comparisons {'excludes' if len(excl) == 1 else 'exclude'} 0:** " + "; ".join(
            f"in the {b} bucket, {a_} is {d * 100:+.1f} points vs {b_} (95% interval [{lo * 100:+.1f}, {hi * 100:+.1f}])" for b, a_, b_, d, lo, hi in excl)
            + (". In that bucket all three normalized seeds are below all three clean seeds" if low_norm_below_clean else "")
            + ". With nine comparisons, intervals that cover tweet sampling but not seed variation, and no correction, this is a lead to re-check with more seeds, not an established effect.")
        add("")
    mean_cmi_hard, mean_cmi_easy = float(cmi[errs9 == 9].mean()), float(cmi[errs9 == 0].mean())
    add(f"Tweets wrong in all nine runs by bucket: {counts_text(hard_by_b, list(BUCKETS))} (mean CMI {mean_cmi_hard:.1f}, against {mean_cmi_easy:.1f} for tweets right in all nine runs).")
    add("")

    # ================= 7 / 8
    add("## 7. Limitations")
    add("")
    for t in ["**Many comparisons, no correction.** This report compares three modes on the whole dev set, on several tweet subsets and in three CMI buckets, each with a 95% "
              "interval and no multiple-comparison correction, so a few intervals excluding 0 are expected by chance; treat subset and bucket results as exploratory leads.",
              "**Three seeds only.** The standard deviations are rough, and the seeds are not a formal paired design (different vocabularies give different initialisations).",
              "**Development-set model selection.** Each run's best epoch was chosen by dev weighted F1, so all dev scores are slightly optimistic. Final, unbiased "
              "numbers need the single test run, which has **not** been done; nothing here used the test split.",
              "**Dev sampling noise.** With 3,000 tweets (about 1,000 per CMI bucket) small differences are within noise; the bootstrap intervals cover tweet sampling only.",
              "**Pooled predictions are not independent** (the same tweets three times); use them for descriptive pooling, not for intervals.",
              "**Heuristic categories.** Negation and spelling subsets are rule-based and imperfect; sarcasm, ambiguity and label inconsistency are my reading of real tweets, "
              "not annotations, and the gold labels are noisy (annotator agreement was 55%).",
              "**CMI caveats.** Mani's CMI values do not match the paper's averages (documented in `docs/eda_part2.md`), the underlying language tags are noisy, the buckets are "
              "the *candidate* train terciles, and almost every tweet is genuinely mixed, so the 'low' bucket is still a mixed-language bucket.",
              "**Model- and data-specific.** Everything describes this BiLSTM, this preprocessing and the Hinglish dev split; it does not transfer to the transformer or LLM runs."]:
        add(f"- {t}")
    add("")
    add("## 8. Reproducibility and files")
    add("")
    add("- Reproduce with `python -m src.analyze_bilstm_results` from the repository root (pandas, numpy, scikit-learn, matplotlib; no PyTorch needed). The script refuses "
        f"to overwrite existing outputs unless `--overwrite` is given. The bootstrap uses a fixed seed ({BOOT_SEED}, {BOOT_B:,} resamples); everything else is deterministic.")
    add("- Report: `reports/bilstm_results_analysis.md`. Figures: " + ", ".join(f"`results/figures/{v}`" for v in FIG_NAMES.values()) + ".")
    add("- Script: `src/analyze_bilstm_results.py` (new). No other file was created or changed.")
    add("")
    add("## Appendix A. Per-seed confusion matrices (rows = true, columns = predicted: negative, neutral, positive)")
    add("")
    for m in MODES:
        rows = []
        for s in SEEDS:
            cm = conf_seed[(m, s)]
            rows.append([f"seed {s}", *[" / ".join(str(int(v)) for v in cm[i]) for i in range(3)]])
        add(f"**{m}**")
        add("")
        for line in table(["Run", "true negative", "true neutral", "true positive"], rows):
            add(line)
        add("")

    # ----- summary block (computed)
    best_wf = max(MODES, key=lambda m: np.mean(agg("weighted_f1", m)))
    spread = max(np.mean(agg("weighted_f1", m)) for m in MODES) - min(np.mean(agg("weighted_f1", m)) for m in MODES)
    diffs_incl = all(lo <= 0 <= hi for a_, b_ in pairs for lo, hi in [ci(boot[a_]["accuracy"] - boot[b_]["accuracy"])])
    summ = ["## Main findings", ""]
    summ.append(f"1. **The three preprocessing modes are not clearly distinguishable on the dev set.** Mean weighted F1 over three seeds: "
                + ", ".join(f"{m} {np.mean(agg('weighted_f1', m)):.4f} (± {np.std(agg('weighted_f1', m), ddof=1):.4f})" for m in MODES) +
                f"; the spread between modes ({spread:.4f}) is comparable to a single mode's seed std and below the dev sampling error. "
                + ("All pairwise bootstrap intervals for the accuracy difference include 0. " if diffs_incl else "")
                + f"The most consistent pattern is **normalized below clean in {sum(1 for v in nc_acc if v < 0)} of 3 seeds** (accuracy {np.mean(nc_acc):+.4f}, "
                f"95% interval [{lo_nc:+.4f}, {hi_nc:+.4f}]). Normalization did **not** improve performance; no mode should be called the winner.")
    summ.append(f"2. **Errors are about the neutral boundary.** {np.mean([100 * pair_rows[m][1] / pair_rows[m][0] for m in MODES]):.0f}% of errors involve the neutral class; direct "
                f"negative <-> positive flips are only {np.mean([100 * pair_rows[m][2] / pair_rows[m][0] for m in MODES]):.0f}%. Neutral has the lowest F1 in every mode.")
    summ.append(f"3. **Many errors are shared by all runs:** {int((errs9 == 9).sum())} of 3,000 dev tweets ({100 * (errs9 == 9).mean():.0f}%) are wrong in all nine runs, "
                f"{int(easy.sum())} ({100 * easy.mean():.0f}%) are right in all nine, and the rest flip between runs.")
    summ.append(f"4. **Spelling normalization has opposite effects by step (exploratory, small subsets):** on tweets the dictionary step changed (n = {int(dict_changed.sum())}) normalized is "
                f"{dd[0] * 100:+.1f} points vs raw (interval [{dd[1] * 100:+.1f}, {dd[2] * 100:+.1f}], includes 0); on tweets the repeated-letter step changed (n = {int(rep_changed.sum())}) "
                f"it is {rr[0] * 100:+.1f} points ([{rr[1] * 100:+.1f}, {rr[2] * 100:+.1f}], excludes 0); and on control tweets with identical text in all modes it still differs by "
                f"{uu[0] * 100:+.1f} points ([{uu[1] * 100:+.1f}, {uu[2] * 100:+.1f}]) purely because of different training, which is the scale of difference not to over-interpret (section 5.4).")
    summ.append("5. **Code-mixing (CMI buckets): " + ("no clear trend within any mode" if all(within_flags) else "see section 6")
                + f"** (high-minus-low intervals include 0). One bucket-level pairwise difference stands out: "
                + ("; ".join(f"in the {b} bucket {a_} is {d * 100:+.1f} points vs {b_} ([{lo * 100:+.1f}, {hi * 100:+.1f}])" for b, a_, b_, d, lo, hi in excl) if excl else "none excludes 0")
                + ("; all three normalized seeds are below all three clean seeds there" if (excl and low_norm_below_clean) else "")
                + f". With nine comparisons and no correction this is a lead to re-check, not an established effect. Hard tweets are spread almost evenly over the buckets ({counts_text(hard_by_b, list(BUCKETS))}).")
    summ.append("6. **Limits:** three seeds, best epoch selected on dev, heuristic categories, no test evaluation (section 7).")
    summ.append("")
    L[summary_slot:summary_slot + 1] = summ

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    return D


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dev-set analysis of the nine BiLSTM runs (read-only on existing outputs).")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--overwrite", action="store_true", help="replace the report and figures this script wrote earlier")
    args = ap.parse_args(argv)
    D = build(Path(args.data_root), args.overwrite)
    # every verification check must pass, otherwise the script exits with an error
    bad = [d for d, ok, _ in D["checks"] if not ok]
    print(f"wrote {REPORT_PATH.relative_to(ROOT)} and {len(FIG_NAMES)} figures; verification checks: {len(D['checks']) - len(bad)}/{len(D['checks'])} passed")
    if bad:
        print("FAILED CHECKS:", *bad, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
