"""Add dev weighted F1, macro F1 and accuracy columns to results/experiment_log.csv (issue #4).

Run from the repo root:  python -m src.log_metrics
- Each log row is matched to its dev prediction file and the three scores are recomputed with src/metrics.py.
- The existing `dev_score` column (weighted F1, the official metric) is left unchanged and must agree with the
  recomputed weighted F1; for BiLSTM rows the macro F1 and accuracy written in `notes` must agree too.
  The script stops if anything disagrees.
- Rows without a value in the new columns are filled; rows that already have one are left as they are, so the
  script can be run again after new experiments are logged.
- Training scripts (src/baselines_tfidf.py, src/train_bilstm.py) still append rows with the ORIGINAL 9 columns.
  Until this script is rerun, such a row has empty values in the three new columns. Line endings stay CRLF.
"""
import re
from pathlib import Path

import pandas as pd

from src.data import load_split
from src.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "experiment_log.csv"
NEW = ["dev_weighted_f1", "dev_macro_f1", "dev_accuracy"]
TOL = 5e-5  # the log stores four decimals


def prediction_file(row):
    """Path of the dev prediction file for a log row, or None for the majority-class floor."""
    model, notes = str(row["model"]), str(row["notes"])
    if model.startswith("majority"):
        return None
    if model.startswith("tfidf"):
        name = model.split()[1].replace("+", "_")
        return ROOT / "results" / "predictions" / f"tfidf_{name}_dev.csv"
    run = re.search(r"run=(\w+)", notes)
    if model.startswith("bilstm") and run:
        seed = re.search(r"s(\d+)$", run.group(1)).group(1)
        return ROOT / "results" / "predictions" / f"bilstm_{run.group(1)}_seed{seed}_dev.csv"
    raise ValueError(f"cannot find the prediction file for log row: {model!r}, {notes[:60]!r}")


def scores_for(row, dev):
    path = prediction_file(row)
    if path is None:  # always predict the label named in `settings`
        label = re.search(r"'(\w+)'", str(row["settings"])).group(1)
        y_true, y_pred = dev["label"], [label] * len(dev)
    else:
        d = pd.read_csv(path)
        m = dev[["uid", "label"]].merge(d[["uid", "pred"]], on="uid", how="inner", validate="one_to_one")
        if len(m) != len(dev) or len(d) != len(dev):
            raise ValueError(f"{path.name}: uids do not match the dev split")
        y_true, y_pred = m["label"], m["pred"]
    m = compute_metrics(y_true, y_pred)
    return m["weighted_f1"], m["macro_f1"], m["accuracy"]


def main():
    dev = load_split("dev")
    log = pd.read_csv(LOG)
    for col in NEW:
        if col not in log.columns:
            log[col] = float("nan")
    for i, row in log.iterrows():
        if not pd.isna(row["dev_weighted_f1"]):
            continue
        w, mac, acc = scores_for(row, dev)
        if abs(w - float(row["dev_score"])) > TOL:
            raise AssertionError(f"row {i}: dev_score {row['dev_score']} != recomputed weighted F1 {w:.4f}")
        notes = str(row["notes"])
        for key, value in (("macro_f1", mac), ("accuracy", acc)):
            found = re.search(rf"{key}=([0-9.]+)", notes)
            if found and abs(float(found.group(1)) - value) > TOL:
                raise AssertionError(f"row {i}: notes say {key}={found.group(1)} but recomputed {value:.4f}")
        log.loc[i, NEW] = [round(w, 4), round(mac, 4), round(acc, 4)]
    log.to_csv(LOG, index=False, lineterminator="\r\n")  # CRLF, like the csv.writer calls that append rows
    print(log[["who", "model", "seed", "dev_score"] + NEW].to_string(index=False, max_colwidth=26))


if __name__ == "__main__":
    main()
