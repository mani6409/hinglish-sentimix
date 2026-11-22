"""Agreement between two readers' hand categories for the same tweets (issue #4).

Usage:
  python -m src.label_agreement --make-template     writes results/error_categories_reviewer2_template.csv
                                                    (the 60 tweets, no categories) for the second reader to fill
  python -m src.label_agreement SECOND_FILE         compares SECOND_FILE (uid,category) with the first reader's
                                                    original results/error_categories_manual.csv

Reports raw agreement, Cohen's kappa and a table of the disagreements. The first reader's file is never modified.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.data import load_split
from src.uid_checks import check_uid_alignment

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "results" / "error_categories_manual.csv"
TEMPLATE = ROOT / "results" / "error_categories_reviewer2_template.csv"


def agreement(first, second):
    """Raw agreement and Cohen's kappa of two (uid, category) frames over exactly the same uids."""
    check_uid_alignment(second["uid"], first["uid"], name="second reader's file", reference_name="first reader's file")
    m = first.merge(second, on="uid", suffixes=("_first", "_second"), validate="one_to_one")
    a, b = m["category_first"], m["category_second"]
    raw = float((a == b).mean())
    kappa = float(cohen_kappa_score(a, b)) if (a.nunique() > 1 or b.nunique() > 1) else float("nan")
    return {"tweets": len(m), "raw_agreement": raw, "cohen_kappa": kappa}, m[a != b]


def make_template():
    first = pd.read_csv(FIRST)
    dev = load_split("dev")[["uid", "text"]]
    t = first[["uid"]].merge(dev, on="uid", how="left", validate="one_to_one")
    if t["text"].isna().any():
        raise ValueError("some uids of the first reader's file are not in dev")
    t["category"] = ""
    t.to_csv(TEMPLATE, index=False)
    print(f"wrote {TEMPLATE} ({len(t)} tweets)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("second", nargs="?")
    ap.add_argument("--make-template", action="store_true")
    args = ap.parse_args(argv)
    if args.make_template:
        return make_template()
    if not args.second:
        ap.error("give the second reader's file, or use --make-template")
    result, diffs = agreement(pd.read_csv(FIRST), pd.read_csv(args.second))
    print(result)
    print(diffs.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
