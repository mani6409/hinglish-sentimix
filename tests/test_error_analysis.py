from pathlib import Path

import pandas as pd

from src.error_analysis import flags, plot_confusion

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"implicit_context", "label_questionable", "sarcasm_irony", "negation", "slang_informal",
              "transliteration_variants", "clear_sentiment_missed", "code_mixing"}


def test_flags_detect_negation_variant_truncation_and_repeats():
    df = pd.DataFrame({"text": ["yeh acha nahi hai …", "Sooo good 😍", "nhi yaar https // t . co / x"]})
    f = flags(df, {"acha"})
    assert list(f["negation_word"]) == [True, False, True]
    assert list(f["spelling_variant"]) == [True, False, False]
    assert list(f["truncated"]) == [True, False, False]
    assert list(f["repeated_letters"]) == [False, True, False]
    assert list(f["has_emoji"]) == [False, True, False]
    assert list(f["url"]) == [False, False, True]


def test_negation_matching_is_whole_word():
    f = flags(pd.DataFrame({"text": ["nahin", "nothing", "cannot"]}), set())
    assert list(f["negation_word"]) == [True, False, True]


def test_confusion_figure_is_written(tmp_path):
    out = tmp_path / "cm.png"
    plot_confusion([[5, 1, 0], [1, 5, 1], [0, 1, 5]], "t", out)
    assert out.stat().st_size > 0


def test_manual_category_file_is_valid():
    m = pd.read_csv(ROOT / "results" / "error_categories_manual.csv")
    assert len(m) == 60 and m["uid"].is_unique
    assert set(m["category"]) <= CATEGORIES


# ---- review additions ----------------------------------------------------------------------------------------
import numpy as np
import pytest

from src.error_analysis import EMOJI_PATTERN, URL_PATTERN, neutral_error_counts


@pytest.mark.parametrize("text", [
    "see https://example.com/page", "see http://example.com", "https // t . co / abc", "HTTPS://T.CO/x",
    "visit www.example.com", "visit www . example . com",
])
def test_url_detector_finds_ordinary_and_tokenised_urls(text):
    assert URL_PATTERN.search(text)


@pytest.mark.parametrize("text", ["no link here", "the http protocol", "a // comment", "show wwwwing"])
def test_url_detector_ignores_non_urls(text):
    assert not URL_PATTERN.search(text)


def test_flags_url_column_uses_the_corrected_detector():
    f = flags(pd.DataFrame({"text": ["x https://a.com y", "x https // t . co / z", "plain"]}), set())
    assert list(f["url"]) == [True, True, False]


def test_emoji_detector_is_a_heuristic_with_known_gaps():
    assert EMOJI_PATTERN.search("so good 😍")
    assert EMOJI_PATTERN.search("❤")
    assert EMOJI_PATTERN.search("🤦🏽‍♂️")        # multi-codepoint sequence: matched through its first symbol
    assert not EMOJI_PATTERN.search("🇮🇳")        # regional-indicator flag: documented gap
    assert not EMOJI_PATTERN.search("plain text")


def test_neutral_error_counts_from_confusion_matrix():
    cm = [[571, 265, 54], [261, 623, 244], [88, 251, 643]]   # best baseline on dev
    c = neutral_error_counts(cm)
    assert c["errors"] == 1163
    assert c["errors_involving_neutral"] == 1021       # (261 + 244) true neutral wrong + (265 + 251) predicted neutral wrong
    assert c["negative_positive_swaps"] == 142         # 54 + 88
    assert c["share_involving_neutral"] == pytest.approx(1021 / 1163)


def test_neutral_error_counts_agree_with_row_count():
    rng = np.random.default_rng(3)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    wrong = y != p
    rows = int(((y == 1) | (p == 1))[wrong].sum())
    assert neutral_error_counts(cm)["errors_involving_neutral"] == rows


def test_bonferroni_threshold_for_six_flags():
    assert 0.05 / 6 == pytest.approx(0.008333, abs=1e-6)


def test_reading_sample_reproduces_the_saved_hand_labelled_uids():
    from src.data import load_split
    from src.error_analysis import BEST, reading_sample_uids
    dev = load_split("dev")
    preds = pd.read_csv(ROOT / "results" / "predictions" / f"{BEST}_dev.csv")
    saved = pd.read_csv(ROOT / "results" / "error_categories_manual.csv")
    uids = reading_sample_uids(dev, preds)
    assert len(uids) == 60 and set(uids) == set(saved["uid"])


def test_experiment_log_keeps_crlf_line_endings():
    raw = (ROOT / "results" / "experiment_log.csv").read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") > 1
