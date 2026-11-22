"""Text preprocessing and transliteration normalisation for Hinglish tweets.

Pipeline (each step can be switched off; see docs/preprocessing.md):

    original text
      -> clean_text()                 URLs, mentions, hashtags, RT, truncation, whitespace
      -> normalize_repeated_chars()   "goooood" -> "good"   (runs of 3+ letters -> 2)
      -> normalize_hinglish()         "bohot" -> "bahut"     (dictionary, whole words only)

Design rules
- Nothing here is learned from data: no vocabulary is fitted, no statistics are
  computed, and labels are never looked at. The same function is applied to
  train, dev and test, so it cannot leak information between splits.
- Sentiment-bearing material is kept: negations, emojis, exclamation marks,
  hashtag words and capitalisation (unless `lowercase=True` is requested).
- Output is a plain space-separated string, so it can be fed to a word-level
  BiLSTM (`tokenize`) or to any subword tokenizer (mBERT, XLM-R, ...) unchanged.
- The input in this project is already tokenised and re-joined with spaces
  (e.g. ``https // t . co / ID``, ``@ name``, ``didn ’ t``); the regexes below
  handle that form and also ordinary untokenised tweets.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS_PATH = ROOT / "configs" / "hinglish_variants.json"

URL_PLACEHOLDER = "URL"
MENTION_PLACEHOLDER = "USER"

# private-use characters used only inside clean_text() as temporary markers
_URL_MARK = ""
_MENTION_MARK = ""

# ---------------------------------------------------------------- patterns --
# Link forms seen in the data: "https // t . co / ID", "https // t co / ID"
# (the dot is sometimes lost) and ordinary "https://example.com/a/b".
_URL_RE = re.compile(
    r"""
    (?:https?\s*:?\s*//|www\s*\.)\s*
    (?:
        t\s*\.?\s*co\s*/\s*[\w-]*                       # twitter short link, spaced or not
      | [\w-]+(?:\.[\w-]+)*(?:\s*/\s*[\w\-.%?=&#~+]*)*  # host (+ path segments)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
# "@ name" (tokenised) or "@name"; also "@ Khushal _ Khattak" where '_' got split off.
_MENTION_RE = re.compile(r"@\s*\w+(?:\s+_+\s+\w+)*")
_HASHTAG_RE = re.compile(r"#\s*(\w+)")
_LEADING_RT_RE = re.compile(r"^\s*RT\b[\s:]*")
# a truncation marker right before the (placeholder) URL or at the very end
_TRUNCATION_RE = re.compile(r"\s*…\s*(?=" + _URL_MARK + r"|\s*$)")  # only the Unicode '…', not "..."
# "didn ’ t" / "don ’ t" / "I ’ m" / "that ’ s" -> "didn't" ... (keeps negation intact)
_CONTRACTION_RE = re.compile(r"(?<=\w)\s*[’‘`']\s*(?=(?:t|s|m|re|ve|ll|d)\b)", re.IGNORECASE)
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_INVISIBLE_RE = re.compile(r"[​‌﻿\x00-\x08\x0b-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


def _as_text(value) -> str:
    """Coerce an arbitrary input to str. None and NaN become ''."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value if isinstance(value, str) else str(value)


# -------------------------------------------------------------- clean_text --
def clean_text(
    text,
    *,
    url: str = "placeholder",
    mention: str = "placeholder",
    hashtag: str = "word",
    strip_rt: bool = True,
    strip_truncation: bool = True,
    fix_contractions: bool = True,
    collapse_mentions: bool = True,
    split_camel_hashtags: bool = False,
    lowercase: bool = False,
) -> str:
    """Remove Twitter noise while keeping sentiment-bearing content.

    url / mention: "placeholder" (default, -> "URL" / "USER"), "remove" or "keep".
    hashtag:       "word" (default, '# NewIndia' -> 'NewIndia'), "remove" or "keep".
    strip_rt:      drop a leading retweet marker "RT" (it carries no sentiment).
    strip_truncation: drop the '…' Twitter adds to cut-off tweets (11% of train).
    fix_contractions: rejoin "didn ’ t" -> "didn't" so negation survives tokenisation.
    collapse_mentions: "@a @b @c" -> one placeholder instead of three.
    split_camel_hashtags: 'NewIndia' -> 'New India' (off by default).
    lowercase:     lowercase everything (off by default; emojis are unaffected).

    Emojis, punctuation, digits and every ordinary word (including negations
    such as nahi, nhi, not, never) are left exactly as they are.
    """
    for name, val, allowed in (("url", url, ("placeholder", "remove", "keep")),
                               ("mention", mention, ("placeholder", "remove", "keep")),
                               ("hashtag", hashtag, ("word", "remove", "keep"))):
        if val not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {val!r}")

    # remove invisible and control characters; missing values become empty text
    s = _INVISIBLE_RE.sub("", _as_text(text))
    if not s.strip():
        return ""

    # drop a leading 'RT' (retweet marker)
    if strip_rt:
        s = _LEADING_RT_RE.sub("", s)

    # replace links with a temporary marker so the later steps cannot damage them
    if url != "keep":
        s = _URL_RE.sub(_URL_MARK, s)
    # remove the '…' that Twitter adds to cut-off tweets
    if strip_truncation:
        s = _TRUNCATION_RE.sub(" ", s)

    # replace @mentions with a temporary marker
    if mention != "keep":
        s = _MENTION_RE.sub(_MENTION_MARK, s)
        if collapse_mentions:
            s = re.sub(_MENTION_MARK + r"(?:\s*" + _MENTION_MARK + r")+", _MENTION_MARK, s)

    # '# NewIndia' becomes 'NewIndia': the word is kept, the # sign is dropped
    if hashtag == "word":
        def _tag(m):
            w = m.group(1)
            return _CAMEL_RE.sub(" ", w) if split_camel_hashtags else w
        s = _HASHTAG_RE.sub(_tag, s)
    elif hashtag == "remove":
        s = _HASHTAG_RE.sub(" ", s)

    # turn the markers into the final placeholders URL and USER (or remove them)
    s = s.replace(_URL_MARK, {"placeholder": f" {URL_PLACEHOLDER} ", "remove": " "}.get(url, " "))
    s = s.replace(_MENTION_MARK, {"placeholder": f" {MENTION_PLACEHOLDER} ", "remove": " "}.get(mention, " "))

    # rejoin split contractions so negations such as "didn't" stay intact
    if fix_contractions:
        s = _CONTRACTION_RE.sub("'", s)
    if lowercase:
        s = s.lower()
    # collapse repeated spaces
    return _WS_RE.sub(" ", s).strip()


# ----------------------------------------------- normalize_repeated_chars ---
_REPEAT_RE = {}


def normalize_repeated_chars(text, max_repeat: int = 2) -> str:
    """Shorten runs of the same LETTER longer than `max_repeat` to `max_repeat`.

    Conservative on purpose:
    - only letters are touched (digits, punctuation and emojis such as '😂😂😂' are kept,
      because "!!!" and repeated emojis can signal emphasis);
    - runs of 2 are never touched (they are legitimate: good, yaar, aaj);
    - runs are cut to 2, not 1, so 'goooood' -> 'good' and 'soooo' -> 'soo'
      (reducing to 1 would wrongly turn 'good' into 'god'-like forms);
    - the comparison ignores case, and the first characters' case is kept ('EEEE' -> 'EE').
    """
    if max_repeat < 1:
        raise ValueError("max_repeat must be >= 1")
    # build the pattern once per setting and reuse it
    pat = _REPEAT_RE.get(max_repeat)
    if pat is None:
        pat = _REPEAT_RE[max_repeat] = re.compile(
            r"([^\W\d_])\1{%d,}" % max_repeat, re.IGNORECASE)
    # cut every long run of one letter down to max_repeat letters
    return pat.sub(lambda m: m.group(0)[:max_repeat], _as_text(text))


# ------------------------------------------------------- normalize_hinglish --
def _apply_case(original: str, replacement: str) -> str:
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def build_variant_map(variants: Mapping[str, object]) -> dict:
    """Turn a dictionary into a flat {variant: canonical} map, with checks.

    Accepts either {canonical: [variants...]} (the JSON file format) or an already flat
    {variant: canonical} mapping (string values). Keys starting with '_' are ignored.
    Raises ValueError if a variant maps to two canonical forms, or if a canonical form
    is also used as a variant (this would make normalisation order-dependent).
    """
    flat: dict[str, str] = {}
    canonicals = set()
    for key, val in variants.items():
        if key.startswith("_"):
            continue
        pairs = [(v, key) for v in val] if isinstance(val, (list, tuple, set)) else [(key, val)]
        for variant, canon in pairs:
            variant, canon = variant.strip().lower(), canon.strip().lower()
            if not variant or not canon:
                raise ValueError("empty variant or canonical form in dictionary")
            if variant in flat and flat[variant] != canon:
                raise ValueError(f"{variant!r} maps to both {flat[variant]!r} and {canon!r}")
            if variant != canon:
                flat[variant] = canon
            canonicals.add(canon)
    # a word may not be both a variant and a canonical spelling
    clash = canonicals & set(flat)
    if clash:
        raise ValueError(f"canonical forms also listed as variants: {sorted(clash)}")
    return flat


class HinglishNormalizer:
    """Whole-word, case-insensitive spelling normaliser driven by a dictionary.

    Only exact dictionary words are replaced, so English words and unknown Hindi
    spellings pass through unchanged. The original capitalisation pattern is kept
    ('Acha' -> 'Achha', 'ACHA' -> 'ACHHA').
    """

    def __init__(self, variants: Optional[Mapping[str, object]] = None, path=None):
        if variants is None:
            variants = json.loads(Path(path or DEFAULT_VARIANTS_PATH).read_text(encoding="utf-8"))
        self.variant_map = build_variant_map(variants)
        if self.variant_map:
            # longest variants first, so a long variant is not hidden by a shorter one
            alt = "|".join(sorted(map(re.escape, self.variant_map), key=len, reverse=True))
            # not glued to other word characters or apostrophes ("nhi" must not match inside "nhin'")
            self._re = re.compile(r"(?<![\w'’])(?:%s)(?![\w'’])" % alt, re.IGNORECASE)
        else:
            self._re = None

    def __call__(self, text) -> str:
        s = _as_text(text)
        if self._re is None:
            return s
        # replace each variant with its canonical spelling, keeping the capital letters
        return self._re.sub(lambda m: _apply_case(m.group(0), self.variant_map[m.group(0).lower()]), s)


@lru_cache(maxsize=1)
def get_default_normalizer() -> HinglishNormalizer:
    """The normaliser built from configs/hinglish_variants.json (loaded once)."""
    return HinglishNormalizer()


def normalize_hinglish(text, variants: Optional[Mapping[str, object]] = None) -> str:
    """Map common Romanised-Hindi spelling variants to one canonical spelling.

    `variants`: optional custom dictionary ({canonical: [variants]}); by default
    configs/hinglish_variants.json is used. To normalise many texts with a custom
    dictionary, build one `HinglishNormalizer(variants)` and call it instead.
    """
    norm = get_default_normalizer() if variants is None else HinglishNormalizer(variants)
    return norm(text)


# ---------------------------------------------------------- preprocess_text --
PRESETS = {
    "raw": dict(clean=False, repeats=False, hinglish=False),   # text exactly as loaded
    "clean": dict(clean=True, repeats=False, hinglish=False),  # noise removed, spelling untouched
    "normalized": dict(clean=True, repeats=True, hinglish=True),
}


def preprocess_text(
    text,
    *,
    clean: bool = True,
    repeats: bool = True,
    hinglish: bool = True,
    lowercase: bool = False,
    variants: Optional[Mapping[str, object]] = None,
    clean_options: Optional[Mapping[str, object]] = None,
) -> str:
    """Run the full pipeline on one text. Every stage can be switched off.

    The input is never modified in place; callers keep the original string.
    Use PRESETS ("raw", "clean", "normalized") for the standard experiment settings.
    `lowercase` is applied last so dictionary matching and case handling are consistent.
    """
    s = _as_text(text)
    if clean:
        s = clean_text(s, **(clean_options or {}))
    if repeats:
        s = normalize_repeated_chars(s)
    if hinglish:
        s = normalize_hinglish(s, variants)
    if lowercase:
        s = s.lower()
    return _WS_RE.sub(" ", s).strip()


def tokenize(text) -> list:
    """Whitespace tokens of an (already preprocessed) text, e.g. for a word-level BiLSTM.

    The pipeline always outputs single-space-separated tokens, so this is a plain split.
    Subword models (mBERT, XLM-R) should receive the string itself.
    """
    return _as_text(text).split()


# ------------------------------------------------------------ data helpers ---
def preprocess_frame(df, mode: str = "normalized", column: str = "text", out: str = "text_proc", **kwargs):
    """Return a COPY of `df` with a new column `out` holding the preprocessed `column`.

    All original columns (uid, label, text, tokens, langs, n_tokens) are kept unchanged,
    so the original text stays available next to the processed text and row order, uids
    and labels are untouched. `mode` is a key of PRESETS; extra kwargs go to preprocess_text
    and override the preset.
    """
    if mode not in PRESETS:
        raise ValueError(f"mode must be one of {list(PRESETS)}, got {mode!r}")
    opts = {**PRESETS[mode], **kwargs}
    res = df.copy()
    res[out] = [preprocess_text(t, **opts) for t in df[column]]
    return res


def load_processed_split(split: str, mode: str = "normalized", root=None, **kwargs):
    """`src.data.load_split(split)` plus a `text_proc` column. Same rows, uids and labels.

    The same function and settings must be used for train, dev and test.
    """
    from src.data import load_split  # imported lazily: the text functions above need only the standard library

    return preprocess_frame(load_split(split, root=root), mode=mode, **kwargs)
