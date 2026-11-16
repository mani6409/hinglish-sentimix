"""Tests for vocabularies, encoding, padding and truncation (no model, no training)."""
import pytest

from src.bilstm_data import (
    LABEL2ID,
    PAD_ID,
    PLACEHOLDER_WORDS,
    UNK_ID,
    SentimentDataset,
    TextEncoder,
    Vocab,
    derive_length,
    make_batch,
)
from src.preprocess import PRESETS, preprocess_text


# ----------------------------------------------------------------- Vocab ----
def test_vocab_reserved_ids_and_order():
    v = Vocab.build([["b", "a", "a"], ["c", "a", "b"]])
    assert v.itos[:2] == ["<pad>", "<unk>"] and PAD_ID == 0 and UNK_ID == 1
    assert v.itos[2:] == ["a", "b", "c"]  # by frequency (a=3, b=2, c=1)
    assert len(v) == 5


def test_vocab_ties_are_alphabetical_so_runs_are_identical():
    a = Vocab.build([["z", "y", "x"]])
    b = Vocab.build([["x", "y", "z"]])
    assert a.itos == b.itos == ["<pad>", "<unk>", "x", "y", "z"]


def test_vocab_min_freq_and_max_size():
    seqs = [["a", "a", "a", "b", "b", "c"]]
    assert Vocab.build(seqs, min_freq=2).itos[2:] == ["a", "b"]
    assert Vocab.build(seqs, max_size=1).itos[2:] == ["a"]


def test_unknown_tokens_become_unk():
    v = Vocab.build([["hello", "world"]])
    assert v.encode(["hello", "never_seen", "world"]) == [v.stoi["hello"], UNK_ID, v.stoi["world"]]
    assert "never_seen" not in v


def test_literal_pad_and_unk_strings_in_text_do_not_break_the_vocab():
    v = Vocab.build([["<pad>", "<unk>", "x"]])
    assert v.itos == ["<pad>", "<unk>", "x"]


def test_vocab_rejects_bad_tables():
    with pytest.raises(ValueError):
        Vocab(["a", "b"])
    with pytest.raises(ValueError):
        Vocab(["<pad>", "<unk>", "a", "a"])


def test_derive_length():
    assert derive_length([3, 5, 9]) == 9
    assert derive_length(list(range(0, 101)), 90) == 90  # exact 90th percentile
    assert derive_length([1, 2, 3, 4], 50) == 3  # 2.5 is rounded up
    assert derive_length([], 100) == 1


# ------------------------------------------------------------ TextEncoder ----
TRAIN = ["yaar bahut achha hai", "yaar kya bakwas hai", "bahut bura hai"]


def test_encoder_is_fitted_on_train_only():
    enc = TextEncoder.fit(TRAIN, word_min_freq=1, char_min_freq=1)
    assert "yaar" in enc.word_vocab and "bakwas" in enc.word_vocab
    dev_only = "mast zabardast"  # appears only in dev/test
    assert "mast" not in enc.word_vocab
    words, chars = enc.encode(dev_only)
    assert words == [UNK_ID, UNK_ID]  # unseen words -> <unk>, vocabulary not extended
    assert "mast" not in enc.word_vocab and len(enc.word_vocab) == len(TextEncoder.fit(TRAIN, word_min_freq=1, char_min_freq=1).word_vocab)
    assert all(len(c) >= 1 for c in chars)


def test_word_min_freq_sends_rare_train_words_to_unk():
    enc = TextEncoder.fit(TRAIN, word_min_freq=2, char_min_freq=1)
    assert "yaar" in enc.word_vocab and "bahut" in enc.word_vocab and "hai" in enc.word_vocab
    assert "bakwas" not in enc.word_vocab and "bura" not in enc.word_vocab


def test_unseen_characters_become_unk():
    enc = TextEncoder.fit(["abc"], word_min_freq=1, char_min_freq=1)
    words, chars = enc.encode("abz")
    assert chars[0][2] == UNK_ID and chars[0][0] != UNK_ID


def test_lengths_are_derived_from_train_data():
    enc = TextEncoder.fit(["a b c d e", "a b", "abcdefghij k"], word_min_freq=1, char_min_freq=1,
                          chars_percentile=100)
    assert enc.max_words == 5  # longest train tweet
    assert enc.max_chars == 10  # longest train token


def test_explicit_limits_override_derived_ones():
    enc = TextEncoder.fit(TRAIN, max_words=7, max_chars=9)
    assert (enc.max_words, enc.max_chars) == (7, 9)


def test_truncation_of_words_and_characters():
    enc = TextEncoder.fit(["aaaa bbbb cccc dddd"], word_min_freq=1, char_min_freq=1, max_words=3, max_chars=2)
    words, chars = enc.encode("aaaa bbbb cccc dddd eeee")
    assert len(words) == 3 and len(chars) == 3  # kept the first 3 words
    assert all(len(c) == 2 for c in chars)  # kept the first 2 characters of each


def test_character_encoding_matches_vocab():
    enc = TextEncoder.fit(["ab ba"], word_min_freq=1, char_min_freq=1)
    words, chars = enc.encode("ab")
    cv = enc.char_vocab
    assert chars == [[cv.stoi["a"], cv.stoi["b"]]]


def test_lowercasing_options():
    enc = TextEncoder.fit(["Good good"], word_min_freq=1, char_min_freq=1)  # default: lowercase words, keep char case
    (w1, w2), (c1, c2) = enc.encode("Good good")
    assert w1 == w2 != UNK_ID  # one word entry
    assert c1[0] != c2[0]  # 'G' and 'g' stay different characters
    enc2 = TextEncoder.fit(["Good good"], word_min_freq=1, char_min_freq=1, lowercase_words=False, lowercase_chars=True)
    (w1, w2), (c1, c2) = enc2.encode("Good good")
    assert w1 != w2 and c1 == c2


def test_empty_or_blank_tweet_has_length_one():
    enc = TextEncoder.fit(TRAIN)
    assert enc.encode("") == ([UNK_ID], [[UNK_ID]])
    assert enc.encode("   ") == ([UNK_ID], [[UNK_ID]])


def test_count_truncated_reports_what_encode_cuts():
    enc = TextEncoder.fit(["aaaa bbbb"], word_min_freq=1, char_min_freq=1, max_words=3, max_chars=4)
    assert enc.count_truncated(["a b", "a b c d", "abcdefgh b"]) == {"tweets": 1, "words": 1}
    assert enc.count_truncated([]) == {"tweets": 0, "words": 0}


def test_encoder_round_trips_through_plain_python_dict():
    import json

    enc = TextEncoder.fit(TRAIN + ["Mast YAAR 😂"], word_min_freq=1, char_min_freq=1, max_words=6, max_chars=7,
                          lowercase_words=False, lowercase_chars=True)
    d = enc.to_dict()
    json.dumps(d)  # only plain lists / ints / bools, so it can sit inside a checkpoint
    enc2 = TextEncoder.from_dict(d)
    assert enc2.to_dict() == d
    for t in ("yaar bahut achha", "Mast YAAR 😂 unseenword", "", "x" * 30):
        assert enc2.encode(t) == enc.encode(t)


# ---- review issue I4: the placeholders USER / URL must not collide with the ordinary words user / url ----
SPECIAL_IDS = {"USER": 2, "URL": 3}  # reserved right after <pad>=0 and <unk>=1


def test_placeholder_words_are_exactly_what_preprocessing_writes():
    out = preprocess_text("RT @ bob see https // t . co / abc")
    assert out.split() == ["USER", "see", "URL"]
    assert PLACEHOLDER_WORDS == ("USER", "URL")


@pytest.mark.parametrize("texts,min_freq", [
    (["a b c"], 1),                       # data with no placeholder at all (like raw mode)
    (["USER URL x x"], 1),                # data with both
    (["USER x x y y"], 5),                # placeholder rarer than min_freq: still reserved
    ([], 1),                              # no data
])
def test_placeholders_have_reserved_ids_whatever_the_data(texts, min_freq):
    enc = TextEncoder.fit(texts, word_min_freq=min_freq, char_min_freq=1)
    assert enc.word_vocab.itos[:4] == ["<pad>", "<unk>", "USER", "URL"]
    assert {w: enc.word_vocab.stoi[w] for w in PLACEHOLDER_WORDS} == SPECIAL_IDS


def test_placeholder_and_ordinary_word_get_different_ids():
    enc = TextEncoder.fit(["USER user URL url User Url the"], word_min_freq=1, char_min_freq=1)
    (w_user, w_user2, w_url, w_url2, w_User, w_Url, w_the), _ = enc.encode("USER user URL url User Url the")
    assert (w_user, w_url) == (2, 3)                       # placeholders
    assert w_user2 == w_User and w_url2 == w_Url            # ordinary words are lowercased together ...
    assert len({w_user, w_user2, w_url, w_url2, w_the}) == 5  # ... and every one has its own id
    assert min(w_user2, w_url2, w_the) >= 4                 # ordinary words come after the reserved ids


def test_a_real_user_word_never_maps_to_the_placeholder():
    """Before the fix, lowercasing made the placeholder `USER` and a dev word `user` share one id."""
    enc = TextEncoder.fit(["USER hello USER hello URL"], word_min_freq=1, char_min_freq=1)
    assert "user" not in enc.word_vocab  # the placeholder was not lowercased into the word `user`
    assert enc.encode("user")[0] == [UNK_ID]  # an unseen ordinary word is unknown, not USER
    assert enc.encode("USER")[0] == [2] and enc.encode("url")[0] == [UNK_ID] and enc.encode("URL")[0] == [3]


def test_placeholders_are_not_lowercased_for_characters_either():
    enc = TextEncoder.fit(["USER user"], word_min_freq=1, char_min_freq=1, lowercase_chars=True)
    (_, _), (c_placeholder, c_word) = enc.encode("USER user")
    assert c_placeholder != c_word  # upper-case characters are kept for the placeholder
    assert c_word == enc.encode("User")[1][0]  # an ordinary word is lowercased as before


def test_placeholders_stay_distinct_when_word_lowercasing_is_off():
    enc = TextEncoder.fit(["USER user"], word_min_freq=1, char_min_freq=1, lowercase_words=False)
    assert enc.word_vocab.itos[:4] == ["<pad>", "<unk>", "USER", "URL"]
    assert enc.encode("USER user")[0][0] == 2 and enc.encode("USER user")[0][1] > 3


def test_special_words_are_exempt_from_min_freq_and_max_size_and_validated():
    v = Vocab.build([["USER", "a", "a", "b"]], min_freq=2, max_size=1, specials=["USER", "URL"])
    assert v.itos == ["<pad>", "<unk>", "USER", "URL", "a"]
    with pytest.raises(ValueError):
        Vocab.build([["a"]], specials=["x", "x"])
    with pytest.raises(ValueError):
        Vocab.build([["a"]], specials=["<unk>"])


RAW_TWEETS = ["RT @ bob the user guide is good https // t . co / abc", "see the url and the User page https // t . co / x",
              "kya haal hai @ amit bhai"]


@pytest.mark.parametrize("mode", list(PRESETS))
def test_placeholder_handling_is_identical_in_raw_clean_and_normalized(mode):
    texts = [preprocess_text(t, **PRESETS[mode]) for t in RAW_TWEETS]
    enc = TextEncoder.fit(texts, word_min_freq=1, char_min_freq=1)
    assert {w: enc.word_vocab.stoi[w] for w in PLACEHOLDER_WORDS} == SPECIAL_IDS  # same ids in every mode
    user, url = enc.word_vocab.stoi["user"], enc.word_vocab.stoi["url"]  # the ordinary words are in the vocabulary ...
    assert user not in (2, 3) and url not in (2, 3) and user != url        # ... with their own ids
    if mode == "raw":
        assert "USER" not in " ".join(texts).split()  # raw text has no placeholders; the ids are simply unused
    else:
        words = enc.encode(texts[0])[0]
        assert words[0] == 2 and words[-1] == 3  # "USER ... URL": mention and link map to the reserved ids
        assert user in words and 2 not in words[1:] and 3 not in words[:-1]  # the word "user" kept its own id


def test_encoder_serialization_keeps_placeholder_mapping():
    enc = TextEncoder.fit(["USER user URL url the"], word_min_freq=1, char_min_freq=1)
    d = enc.to_dict()
    assert d["special_words"] == ["USER", "URL"] and d["word_itos"][:4] == ["<pad>", "<unk>", "USER", "URL"]
    enc2 = TextEncoder.from_dict(d)
    assert enc2.special_words == ("USER", "URL") and enc2.to_dict() == d
    for t in ("USER user URL url", "User Url the unseen"):
        assert enc2.encode(t) == enc.encode(t)


# ----------------------------------------------------- Dataset and batches ----
def _enc():
    return TextEncoder.fit(["ab cde f", "ab"], word_min_freq=1, char_min_freq=1)


def test_dataset_labels_and_validation():
    enc = _enc()
    ds = SentimentDataset(enc, ["ab", "cde"], ["negative", "positive"])
    assert len(ds) == 2 and ds[0][2] == LABEL2ID["negative"] == 0 and ds[1][2] == LABEL2ID["positive"] == 2
    assert SentimentDataset(enc, ["ab"])[0][2] == -1  # unlabeled
    with pytest.raises(ValueError):
        SentimentDataset(enc, ["ab"], ["happy"])
    with pytest.raises(ValueError):
        SentimentDataset(enc, ["ab", "cde"], ["negative"])


def test_make_batch_padding_shapes_and_values():
    torch = pytest.importorskip("torch")
    enc = _enc()
    ds = SentimentDataset(enc, ["ab cde f", "ab"], ["neutral", "positive"])
    b = make_batch([ds[0], ds[1]])
    assert b["words"].shape == (2, 3)  # 3 = longest tweet in the batch
    assert b["chars"].shape == (2, 3, 3)  # 3 = longest word ("cde")
    assert b["lengths"].tolist() == [3, 1]
    assert b["labels"].tolist() == [1, 2]
    assert b["words"][1, 1:].tolist() == [PAD_ID, PAD_ID]  # padded words
    assert b["chars"][1, 1:].sum().item() == 0  # padded words have only pad characters
    assert b["chars"][0, 2].tolist() == [enc.char_vocab.stoi["f"], PAD_ID, PAD_ID]  # short word padded
    assert b["words"].dtype == torch.long


def test_make_batch_is_only_as_wide_as_the_batch():
    pytest.importorskip("torch")
    enc = _enc()
    ds = SentimentDataset(enc, ["ab", "ab"], ["neutral", "neutral"])
    b = make_batch([ds[0], ds[1]])
    assert b["words"].shape == (2, 1) and b["chars"].shape == (2, 1, 2)
