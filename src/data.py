"""Load the SentiMix Hinglish CoNLL files into one row per tweet.

File format (see docs/data_format.md):
    meta<TAB><tweet id>[<TAB><sentiment>]     <- header line (test file has no sentiment)
    <token><TAB><language tag>                <- one line per token
    (blank line between tweets)
Language tags: Hin, Eng, O (other), EMT (emoticon).
"""
import csv
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "zenodo_download" / "extracted"
    / "Semeval_2020_task9_data" / "Hinglish"
)

FILES = {
    "train": "Hinglish_train_14k_split_conll.txt",
    "dev": "Hinglish_dev_3k_split_conll.txt",
    "test": "Hinglish_test_unalbelled_conll_updated.txt",  # no labels inside
}
TEST_LABELS_FILE = "Hinglish_test_labels.txt"


def parse_conll(path):
    """Parse one CoNLL file into a list of tweet dicts:
    {uid, label (or None), tokens, langs}. Raises ValueError on malformed lines
    so data problems are never silently ignored."""
    tweets, cur = [], None
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if line.startswith("meta\t"):
                if cur is not None:
                    tweets.append(cur)
                parts = line.split("\t")
                if len(parts) < 2 or not parts[1].strip():
                    raise ValueError(f"{path}:{n}: bad header line {line!r}")
                label = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
                cur = {"uid": int(parts[1]), "label": label, "tokens": [], "langs": []}
            elif line.strip() == "":
                continue  # separator between tweets
            else:
                if cur is None:
                    raise ValueError(f"{path}:{n}: token before any header")
                if "\t" not in line:
                    raise ValueError(f"{path}:{n}: token line without a tab {line!r}")
                token, lang = line.rsplit("\t", 1)
                cur["tokens"].append(token)
                cur["langs"].append(lang.strip())
    if cur is not None:
        tweets.append(cur)
    return tweets


def to_frame(tweets):
    """Turn the list of tweet dicts from `parse_conll` into a DataFrame.

    Adds `text` (the tokens joined with single spaces, which is what every model reads)
    and `n_tokens`. The token and language-tag lists are kept for the CMI analysis.
    """
    df = pd.DataFrame(tweets)
    df["text"] = df["tokens"].apply(lambda t: " ".join(t))
    df["n_tokens"] = df["tokens"].apply(len)
    return df[["uid", "label", "text", "tokens", "langs", "n_tokens"]]


def load_test_labels(path):
    """Read Uid,Sentiment CSV into {uid: label}."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["Uid"])] = row["Sentiment"].strip()
    return out


def load_split(split, root=None):
    """Return a DataFrame for 'train', 'dev' or 'test'.

    Columns: uid, label, text, tokens (list), langs (list), n_tokens.
    For 'test' the labels are filled in from the separate labels file.
    IMPORTANT: use the test labels only for the single final test run.
    """
    if split not in FILES:
        raise ValueError(f"split must be one of {list(FILES)}")
    root = Path(root) if root else DEFAULT_ROOT
    df = to_frame(parse_conll(root / FILES[split]))
    if split == "test":
        labels = load_test_labels(root / TEST_LABELS_FILE)
        missing = set(df["uid"]) - set(labels)
        if missing:
            raise ValueError(f"{len(missing)} test tweets have no label, e.g. {sorted(missing)[:5]}")
        df["label"] = df["uid"].map(labels)
    return df
