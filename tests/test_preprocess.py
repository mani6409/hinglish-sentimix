import json

import pandas as pd
import pytest

from src.data import DEFAULT_ROOT
from src.preprocess import (
    DEFAULT_VARIANTS_PATH,
    PRESETS,
    HinglishNormalizer,
    build_variant_map,
    clean_text,
    load_processed_split,
    normalize_hinglish,
    normalize_repeated_chars,
    preprocess_frame,
    preprocess_text,
    tokenize,
)


# ------------------------------------------------------------------ URLs ----
def test_tokenised_tco_url_becomes_placeholder():
    assert clean_text("bahut bura https // t . co / j2PLFDjUFE") == "bahut bura URL"


def test_tco_url_with_lost_dot():
    assert clean_text("dekho https // t co / KRth25D2G6 yaar") == "dekho URL yaar"


def test_untokenised_url():
    assert clean_text("see https://example.com/a/b?x=1 now") == "see URL now"
    assert clean_text("see www.example.com now") == "see URL now"


def test_url_remove_and_keep_options():
    t = "wow https // t . co / abc"
    assert clean_text(t, url="remove") == "wow"
    assert "https" in clean_text(t, url="keep")


def test_truncation_marker_before_url_is_dropped():
    assert clean_text("Dusre ko pareshan krke … https // t co / j2PLFDjUFE") == "Dusre ko pareshan krke URL"
    assert clean_text("sentence cut off …") == "sentence cut off"
    assert clean_text("wait... what") == "wait... what"  # ordinary ellipsis is kept


# -------------------------------------------------------------- mentions ----
def test_mention_becomes_placeholder_and_rt_is_dropped():
    assert clean_text("RT @ ashok Modi ji accha bol rahe") == "USER Modi ji accha bol rahe"


def test_consecutive_mentions_collapse():
    assert clean_text("@ a @ b @ c Teri ma bahan hai kya") == "USER Teri ma bahan hai kya"
    assert clean_text("@ a @ b hi", collapse_mentions=False) == "USER USER hi"


def test_mention_with_split_underscore():
    assert clean_text("@ Khushal _ Khattak Hahaha afghani") == "USER Hahaha afghani"
    assert clean_text("@ say __ green Baat srf dance") == "USER Baat srf dance"


def test_mention_remove_and_keep():
    assert clean_text("@ bob hello", mention="remove") == "hello"
    assert clean_text("@ bob hello", mention="keep") == "@ bob hello"


def test_rt_only_removed_at_start():
    assert clean_text("kal RT hua") == "kal RT hua"
    assert clean_text("RT hua", strip_rt=False) == "RT hua"


# -------------------------------------------------------------- hashtags ----
def test_hashtag_word_is_kept():
    assert clean_text("# NewIndia aa gaya") == "NewIndia aa gaya"
    assert clean_text("#Biggboss shuru") == "Biggboss shuru"


def test_hashtag_remove_keep_and_camel_split():
    assert clean_text("#Biggboss shuru", hashtag="remove") == "shuru"
    assert clean_text("#Biggboss shuru", hashtag="keep") == "#Biggboss shuru"
    assert clean_text("#NewIndia", split_camel_hashtags=True) == "New India"


# ---------------------------------------------------------- whitespace ------
def test_whitespace_and_invisible_characters():
    assert clean_text("  bahut \t  accha\n\n hai ​ ") == "bahut accha hai"


def test_unknown_option_value_raises():
    with pytest.raises(ValueError):
        clean_text("x", url="bogus")


# ------------------------------------------------- negation and emoji --------
@pytest.mark.parametrize("neg,expected", [
    ("nahi", "nahi"), ("nhi", "nahi"), ("nahin", "nahi"),  # spelling variants map to one negation
    ("not", "not"), ("never", "never"), ("na", "na"), ("mat", "mat"), ("NOT", "NOT"),  # others untouched
])
def test_negation_words_survive_full_pipeline(neg, expected):
    assert preprocess_text(f"mujhe yeh {neg} pasand") == f"mujhe yeh {expected} pasand"


def test_nhi_maps_to_nahi_not_deleted():
    assert preprocess_text("yeh nhi chalega") == "yeh nahi chalega"


def test_split_contractions_are_rejoined_so_negation_is_kept():
    assert clean_text("I didn ’ t like it") == "I didn't like it"
    assert clean_text("Wasn ’ t what I ’ d hoped") == "Wasn't what I'd hoped"
    assert clean_text("they don ’ t") == "they don't"
    # a closing quote followed by a normal word is not a contraction
    assert clean_text("we “ tried ’ to race") == "we “ tried ’ to race"


def test_emojis_are_preserved():
    t = "bahut accha 😂😂😂 😟 ❤️ #proud"
    out = clean_text(t)
    for e in ("😂😂😂", "😟", "❤️"):
        assert e in out
    assert "😂😂😂" in preprocess_text(t)  # repeated emojis are not shortened


def test_punctuation_emphasis_kept():
    assert clean_text("kya baat hai !!! ?? ") == "kya baat hai !!! ??"


# -------------------------------------------------------------- repeats -----
def test_repeated_chars_examples_from_spec():
    assert normalize_repeated_chars("goooood") == "good"
    assert normalize_repeated_chars("soooo") == "soo"


def test_repeats_conservative():
    assert normalize_repeated_chars("good yaar aaj") == "good yaar aaj"  # runs of 2 untouched
    assert normalize_repeated_chars("EEEE") == "EE"
    assert normalize_repeated_chars("Ooooo") == "Oo"
    assert normalize_repeated_chars("yaaar ilaaaj") == "yaar ilaaj"
    assert normalize_repeated_chars("!!!! 1000000 ...") == "!!!! 1000000 ..."  # only letters


def test_repeats_custom_max():
    assert normalize_repeated_chars("goooood", max_repeat=1) == "god"
    with pytest.raises(ValueError):
        normalize_repeated_chars("x", max_repeat=0)


# ------------------------------------------------------------- hinglish -----
@pytest.mark.parametrize("src,dst", [
    ("accha", "achha"), ("acha", "achha"), ("achhaa", "achha"),
    ("bohot", "bahut"), ("bahot", "bahut"),
])
def test_spec_examples(src, dst):
    assert normalize_hinglish(src) == dst


def test_sentence_level_and_case_preserved():
    assert normalize_hinglish("Acha hai, ACHA lagta hai, bohot sahi") == "Achha hai, ACHHA lagta hai, bahut sahi"


def test_whole_word_only():
    # variants must not be replaced inside longer words
    assert normalize_hinglish("achaar bohotsara kyunhi") == "achaar bohotsara kyunhi"
    assert normalize_hinglish("don't nhi'x") == "don't nhi'x"


def test_english_and_unknown_words_untouched():
    t = "he said to me per the plan hay kia is good"
    assert normalize_hinglish(t) == t


def test_default_dictionary_has_no_common_english_words():
    english = {"he", "me", "to", "is", "per", "par", "hay", "kia", "bi", "ho", "ha", "hi", "no", "so", "do",
               "the", "and", "or", "at", "in", "on", "it", "my", "we", "be", "by", "as", "an", "are", "was",
               "who", "why", "yes", "all", "can", "not", "but", "ok", "oh", "sir", "kar", "mat", "bas", "na"}
    data = json.loads(DEFAULT_VARIANTS_PATH.read_text(encoding="utf-8"))
    flat = build_variant_map(data)
    assert not (set(flat) & english)
    assert not (set(flat.values()) & english)


def test_default_dictionary_is_valid_and_lowercase():
    flat = build_variant_map(json.loads(DEFAULT_VARIANTS_PATH.read_text(encoding="utf-8")))
    assert flat and all(k == k.lower() and v == v.lower() for k, v in flat.items())
    assert not (set(flat) & set(flat.values()))


def test_custom_dictionary_is_configurable():
    custom = {"zindagi": ["jindagi"]}
    assert normalize_hinglish("jindagi acha", variants=custom) == "zindagi acha"  # 'acha' not in custom
    n = HinglishNormalizer({"sab": ["sabhi"]})
    assert n("sabhi log") == "sab log"
    assert HinglishNormalizer({})("acha") == "acha"


def test_flat_dictionary_format_and_underscore_keys_ignored():
    assert build_variant_map({"_comment": "x", "bohot": "bahut"}) == {"bohot": "bahut"}  # flat {variant: canonical}


def test_dictionary_conflicts_are_rejected():
    with pytest.raises(ValueError):
        build_variant_map({"a": ["x"], "b": ["x"]})  # one variant, two canonicals
    with pytest.raises(ValueError):
        build_variant_map({"a": ["b"], "b": ["c"]})  # canonical used as variant


# ---------------------------------------------------- input robustness ------
@pytest.mark.parametrize("bad", [None, float("nan"), "", "   ", "\n\t"])
def test_empty_and_missing_inputs_give_empty_string(bad):
    for f in (clean_text, normalize_repeated_chars, normalize_hinglish, preprocess_text):
        assert f(bad).strip() == ""


def test_non_string_input_is_coerced():
    assert preprocess_text(123) == "123"
    assert tokenize(None) == []


def test_text_made_only_of_noise():
    assert clean_text("https // t . co / abc") == "URL"
    assert clean_text("@ a @ b", mention="remove") == ""


# ------------------------------------------- originals and on/off switches ---
def test_original_text_is_not_modified_and_frame_keeps_it():
    df = pd.DataFrame({"uid": [1, 2], "label": ["positive", "negative"],
                       "text": ["RT @ x Yaaar bohot accha https // t . co / q", "nhi chahiye"]})
    before = df.copy(deep=True)
    out = preprocess_frame(df, mode="normalized")
    pd.testing.assert_frame_equal(df, before)  # input untouched
    assert list(out["text"]) == list(df["text"])  # original column kept verbatim
    assert list(out["uid"]) == [1, 2] and list(out["label"]) == ["positive", "negative"]
    assert out["text_proc"].tolist() == ["USER Yaar bahut achha URL", "nahi chahiye"]


def test_flags_switch_each_stage():
    t = "RT @ x Yaaar bohot accha !!"
    assert preprocess_text(t, clean=False, repeats=False, hinglish=False) == t
    assert preprocess_text(t, repeats=False, hinglish=False) == "USER Yaaar bohot accha !!"
    assert preprocess_text(t, hinglish=False) == "USER Yaar bohot accha !!"
    assert preprocess_text(t) == "USER Yaar bahut achha !!"
    assert preprocess_text(t, lowercase=True) == "user yaar bahut achha !!"


def test_presets():
    t = "bohot accha https // t . co / q"
    assert preprocess_text(t, **PRESETS["raw"]) == t
    assert preprocess_text(t, **PRESETS["clean"]) == "bohot accha URL"
    assert preprocess_text(t, **PRESETS["normalized"]) == "bahut achha URL"


def test_normalised_text_differs_only_when_dictionary_applies():
    t = "kal match hai"
    assert preprocess_text(t) == t


def test_preprocessing_is_idempotent():
    t = "RT @ x # NewIndia Yaaar bohot accha didn ’ t 😂 … https // t . co / q"
    once = preprocess_text(t)
    assert preprocess_text(once) == once


def test_tokenize_gives_whitespace_tokens():
    assert tokenize(preprocess_text("bohot  accha USER")) == ["bahut", "achha", "USER"]


def test_frame_bad_mode_raises():
    with pytest.raises(ValueError):
        preprocess_frame(pd.DataFrame({"text": ["x"]}), mode="nope")


# -------------------------- real data (skipped if the dataset is missing) ----
real = pytest.mark.skipif(not DEFAULT_ROOT.exists(), reason="data not downloaded")


@real
def test_real_splits_keep_rows_uids_labels():
    from src.data import load_split
    for split in ("train", "dev", "test"):
        raw = load_split(split)
        proc = load_processed_split(split)
        assert len(raw) == len(proc)
        assert (raw["uid"].values == proc["uid"].values).all()
        assert (raw["label"].values == proc["label"].values).all()
        assert (raw["text"].values == proc["text"].values).all()  # original text retained
        assert not (proc["text_proc"].str.strip() == "").any()
