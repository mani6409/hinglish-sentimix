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
