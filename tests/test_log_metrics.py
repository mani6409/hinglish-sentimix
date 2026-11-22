import pandas as pd
import pytest

from src.log_metrics import NEW, prediction_file


def row(model, notes="", settings="", seed=42):
    return pd.Series({"model": model, "notes": notes, "settings": settings, "seed": seed})


def test_tfidf_rows_map_to_prediction_files():
    assert prediction_file(row("tfidf word+char + logistic regression")).name == "tfidf_word_char_dev.csv"
    assert prediction_file(row("tfidf char_2-5 + logistic regression")).name == "tfidf_char_2-5_dev.csv"
    assert prediction_file(row("tfidf word_1-2 + logistic regression")).name == "tfidf_word_1-2_dev.csv"


def test_bilstm_rows_map_via_the_run_name():
    p = prediction_file(row("bilstm word+char (cnn)", "run=full_normalized_s43; macro_f1=0.61", seed=43))
    assert p.name == "bilstm_full_normalized_s43_seed43_dev.csv"


def test_word_only_runs_use_the_seed_column():
    p = prediction_file(row("bilstm word-only", "run=bilstm_normalized_wordonly; macro_f1=0.61", seed=44))
    assert p.name == "bilstm_bilstm_normalized_wordonly_seed44_dev.csv"


def test_majority_row_has_no_file_and_unknown_rows_raise():
    assert prediction_file(row("majority-class")) is None
    with pytest.raises(ValueError):
        prediction_file(row("something else"))


def test_new_columns():
    assert NEW == ["dev_weighted_f1", "dev_macro_f1", "dev_accuracy"]
