"""Load the SentiMix Hinglish CoNLL files into one table row per tweet.

Purpose:
    Turns the three official data files (train, dev, test) and the separate test-label file into pandas
    DataFrames with one row per tweet. Every other module in the project gets its tweets through
    ``load_split``; none of them parse the raw files themselves.

Why this file exists:
    The official files use a CoNLL-style layout (one token per line) that no analysis or model can use
    directly. Keeping the parsing in one place means every experiment sees exactly the same tweets, in the
    same order, with the same labels. The parser raises an error on malformed input instead of guessing, so a
    damaged data file stops the run and cannot silently change results.

File format (source, licence and examples: docs/data_format.md):
    meta<TAB><tweet id>[<TAB><sentiment>]     <- header line (the test file has no sentiment)
    <token><TAB><language tag>                <- one line per token
    (blank line between tweets)
    Language tags: Hin (Hindi), Eng (English), O (other: mentions, URLs, punctuation), EMT (emoticon).

Place in the pipeline:
    data/raw/.../Hinglish/*.txt  ->  parse_conll  ->  to_frame  ->  load_split  ->  every other module

Inputs:
    The files named in ``FILES`` and ``TEST_LABELS_FILE``, looked up under ``DEFAULT_ROOT`` (inside the
    repository: data/raw/zenodo_download/extracted/Semeval_2020_task9_data/Hinglish) or under the ``root``
    argument of ``load_split``.

Output:
    A DataFrame with the columns uid, label, text, tokens, langs, n_tokens (described in ``load_split``).

Used by (importers found on main when this was written):
    src/baselines_tfidf.py, src/eda_part1.py, src/eda_part2.py, src/error_analysis.py,
    src/error_analysis_models.py, src/eval_protocol_utils.py, src/evaluate_transformer_test.py,
    src/label_agreement.py, src/log_metrics.py, src/preprocess.py, src/tfidf_error_analysis.py,
    src/zero_shot_llm.py, tests/test_data_seed.py, tests/test_error_analysis.py,
    tests/test_preprocess.py, notebooks/bilstm.ipynb.

Depends on:
    pandas and the Python standard library.

Design decisions:
    * Strict parsing: every problem raises ValueError with the file name and line number.
    * The test file contains no labels. They live in a separate file and are attached by uid (a dictionary
      lookup, never by row position), and a test tweet without a label is an error.
    * ``text`` is the tokens joined by one space. That is the project's "raw" text version: it is already
      tokenised (URLs and punctuation are split into separate tokens, for example ``https // t . co / x``).

Assumptions and limits:
    * Files are UTF-8, and a header line starts with ``meta`` followed by a TAB. A token that is exactly
      ``meta`` followed by a TAB would be misread as a header. No line of the three official files starts
      with ``meta<TAB>`` followed by a non-numeric field (checked with grep), so this does not occur in this
      data; new data should be checked.
    * All 20,000 tweets are kept in memory, which is small.
    * Project rule: the test labels are for the single final test run only (see
      docs/bilstm_final_evaluation_protocol.md). Loading them for descriptive statistics has happened and is
      disclosed there.
"""
import csv
from pathlib import Path

import pandas as pd

# Folder that holds the official files. It lives inside the repository because the course asked for the data
# to be included; the path mirrors how the Zenodo archive unpacks.
DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "zenodo_download" / "extracted"
    / "Semeval_2020_task9_data" / "Hinglish"
)

# Split name -> file name. The test file name is spelled "unalbelled" in the official release; keep it as is.
FILES = {
    "train": "Hinglish_train_14k_split_conll.txt",
    "dev": "Hinglish_dev_3k_split_conll.txt",
    "test": "Hinglish_test_unalbelled_conll_updated.txt",  # no labels inside
}
TEST_LABELS_FILE = "Hinglish_test_labels.txt"  # CSV with the columns Uid,Sentiment


def parse_conll(path):
    """Parse one CoNLL file into a list of tweet dictionaries.

    Purpose:
        First step of loading: converts the one-token-per-line file into Python objects, one per tweet.

    How it works:
        A header line (``meta<TAB>uid[<TAB>label]``) closes the tweet being built, if any, and starts a new
        one. Token lines are appended to the current tweet. Blank lines are skipped, so a tweet ends when
        the next header or the end of the file is reached and not at the blank line; a missing blank line
        therefore cannot merge two tweets.

    Args:
        path: Path (str or Path) of a train, dev or test CoNLL file.

    Returns:
        A list with one dict per tweet: ``{"uid": int, "label": str or None, "tokens": list of str,
        "langs": list of str}``. ``tokens`` and ``langs`` always have the same length and the same order.
        ``label`` is None for the test file, which has no sentiment in its header lines.

    Raises:
        ValueError: A header line has no tweet id, the id is not an integer, a token line appears before the
            first header, or a token line has no TAB (these messages contain ``path:line``). A tweet id that
            is not an integer raises ``int()``'s own ValueError, without path or line.
        OSError: The file cannot be opened (for example FileNotFoundError).

    Edge cases:
        An empty file returns an empty list. A header with no token lines gives a tweet with empty lists. An
        empty label field after the uid is treated as no label (None).

    Example:
        parse_conll(DEFAULT_ROOT / FILES["dev"])[0]["uid"]  ->  the id of the first dev tweet (an int)
    """
    tweets, cur = [], None
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, start=1):
            # Remove only the newline: spaces inside a line belong to the data and must not be stripped.
            line = raw.rstrip("\n")
            if line.startswith("meta\t"):
                # A new header means the previous tweet is complete.
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
                # Split at the LAST tab so that a token that itself contains a tab stays in one piece.
                token, lang = line.rsplit("\t", 1)
                cur["tokens"].append(token)
                cur["langs"].append(lang.strip())
    # The last tweet has no following header, so it is added here.
    if cur is not None:
        tweets.append(cur)
    return tweets


def to_frame(tweets):
    """Convert the output of ``parse_conll`` into the project's standard table.

    Purpose:
        Adds the two derived columns that every analysis needs and fixes the column order.

    Args:
        tweets: List of tweet dicts as returned by ``parse_conll``. It must not be empty.

    Returns:
        A DataFrame with columns ``uid`` (int), ``label`` (str or None), ``text`` (tokens joined by one space),
        ``tokens`` (list of str), ``langs`` (list of str) and ``n_tokens`` (number of tokens). Row order equals
        the order in the file.

    Raises:
        KeyError: ``tweets`` is empty (the DataFrame then has no ``tokens`` column). ``load_split`` never
            triggers this because the official files are not empty.
    """
    df = pd.DataFrame(tweets)
    df["text"] = df["tokens"].apply(lambda t: " ".join(t))
    df["n_tokens"] = df["tokens"].apply(len)
    return df[["uid", "label", "text", "tokens", "langs", "n_tokens"]]


def load_test_labels(path):
    """Read the test-label file into a dictionary.

    Purpose:
        The test tweets come without labels; the organisers published them in a separate CSV file
        (columns ``Uid,Sentiment``). This function reads that file.

    Args:
        path: Path of ``Hinglish_test_labels.txt``.

    Returns:
        Dict mapping tweet id (int) to the sentiment label (str, whitespace removed).

    Raises:
        KeyError: A column ``Uid`` or ``Sentiment`` is missing from the header.
        ValueError: A ``Uid`` is not an integer.

    Edge cases:
        If a uid appeared twice, the later row would silently replace the earlier one. The official file has
        3,000 rows with 3,000 distinct uids, so this does not happen here.
    """
    out = {}
    # newline="" is what the csv module requires so that it handles line endings itself.
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["Uid"])] = row["Sentiment"].strip()
    return out


def load_split(split, root=None):
    """Return the tweets of one split as a DataFrame. This is the entry point for all other modules.

    Purpose:
        One call that gives train, dev or test in the project's standard shape, so every experiment starts
        from identical data.

    Args:
        split: ``"train"`` (14,000 tweets), ``"dev"`` (3,000) or ``"test"`` (3,000).
        root: Optional folder that holds the official files. Default is ``DEFAULT_ROOT``.

    Returns:
        A DataFrame with the columns uid, label, text, tokens (list), langs (list) and n_tokens, in file
        order. For ``"test"`` the labels are filled in from the separate labels file, joined by uid.

    Raises:
        ValueError: ``split`` is not one of the three names, a test tweet has no label in the labels file, or
            the file is malformed (see ``parse_conll``).
        OSError: A data file is missing.

    Edge cases:
        A uid that is in the labels file but not in the test file is ignored. A test tweet without a label
        raises, so a test row never carries an empty label.

    IMPORTANT:
        Use the test labels only for the single final test run (project rule).

    Example:
        dev = load_split("dev"); dev.shape  ->  (3000, 6)
    """
    if split not in FILES:
        raise ValueError(f"split must be one of {list(FILES)}")
    root = Path(root) if root else DEFAULT_ROOT
    df = to_frame(parse_conll(root / FILES[split]))
    if split == "test":
        labels = load_test_labels(root / TEST_LABELS_FILE)
        # Fail loudly if any test tweet cannot be matched: an unlabelled row would otherwise become NaN and
        # distort every metric computed later.
        missing = set(df["uid"]) - set(labels)
        if missing:
            raise ValueError(f"{len(missing)} test tweets have no label, e.g. {sorted(missing)[:5]}")
        df["label"] = df["uid"].map(labels)
    return df
