# Preprocessing and transliteration normalisation

Code: `src/preprocess.py`. Dictionary: `configs/hinglish_variants.json`. Tests: `tests/test_preprocess.py`.

## Pipeline

```
original text  (src.data.load_split -> df["text"], never modified)
  1. clean_text()                  Twitter noise
  2. normalize_repeated_chars()    "goooood" -> "good"
  3. normalize_hinglish()          "bohot" -> "bahut"
  -> processed text (a plain string, tokens separated by single spaces)
```

`preprocess_text(text, clean=True, repeats=True, hinglish=True, lowercase=False, variants=None, clean_options=None)` runs the steps, and each can be switched off. Three standard settings are in `PRESETS`:

| Preset | Steps | Use |
|---|---|---|
| `raw` | none | text exactly as loaded (what the TF-IDF baseline used) |
| `clean` | 1 | noise removed, spelling untouched |
| `normalized` | 1 + 2 + 3 | the "normalised" side of the raw vs normalised experiment |

Comparing `raw` / `clean` / `normalized` separates the effect of cleaning from the effect of spelling normalisation.

## Step 1: `clean_text`

The input is already tokenised and re-joined with spaces, so URLs, mentions and contractions look like `https // t . co / ID`, `@ name` and `didn ’ t`. The regexes handle this form and normal untokenised tweets.

| Item | Default behaviour | Options |
|---|---|---|
| URL (`https // t . co / X`, `https // t co / X`, `https://a.com/b`, `www.a.com`) | replaced by `URL` | `url="remove" / "keep"` |
| Mention (`@ name`, `@ a _ b` where `_` was split off) | replaced by `USER`; a run of mentions becomes one `USER` | `mention=...`, `collapse_mentions=False` |
| Hashtag (`# NewIndia`, `#NewIndia`) | the `#` is dropped and **the word is kept** | `hashtag="remove" / "keep"`, `split_camel_hashtags=True` |
| Leading `RT` | removed (retweet marker, no sentiment) | `strip_rt=False` |
| Trailing `…` | removed when it is at the end or just before a URL (Twitter's cut-off marker, 11% of train tweets) | `strip_truncation=False` |
| Split contractions (`didn ’ t`, `I ’ m`) | rejoined: `didn't`, `I'm` | `fix_contractions=False` |
| Zero-width and control characters, extra whitespace | removed / collapsed | none |
| Case | kept | `lowercase=True` |

Placeholders are plain words (`URL`, `USER`), not special tokens such as `<url>`, so word-level models and subword tokenisers (mBERT, XLM-R) accept them without extra setup.

## Step 2: `normalize_repeated_chars`

Runs of the same letter longer than 2 are cut to 2 (`max_repeat=2`).

- `goooood -> good`, `soooo -> soo`, `yaaar -> yaar`, `ilaaaj -> ilaaj`, `EEEE -> EE`.
- Only letters are touched. `!!!`, digits and repeated emojis such as `😂😂😂` are kept, because they can signal emphasis.
- Runs of 2 are never touched (`good`, `yaar`, `aaj` are legitimate).
- Cutting to 2 and not 1 is deliberate: it avoids creating other words and keeps a trace of emphasis (`soo` still differs from `so`).

## Step 3: `normalize_hinglish`

Maps spelling variants of Romanised Hindi to one canonical spelling, using `configs/hinglish_variants.json` (format `{canonical: [variants]}`).

Rules that protect English and unknown words:
1. Only **exact whole words** from the dictionary are replaced (`achaar` is not touched by the `acha` entry), case-insensitively. The capitalisation pattern is kept (`Acha -> Achha`, `ACHA -> ACHHA`).
2. A variant is only added if it is **not a common English word**. That is why `he`, `me`, `per`, `par`, `kia`, `hay`, `bi` and `h` are not in the file even though they are often Hindi in this data (`mai he chor du`, `waise bi`). A test checks the shipped file against a list of common English words.
3. `build_variant_map` rejects a variant mapped to two canonical forms and a canonical form that is also a variant, so the result never depends on order.
4. No spelling is guessed or generated. Unknown variants (for example `achhi`, `nahim` before it was added) stay as they are.

Extending the dictionary: add the variant to the list of its canonical form in the JSON file and run `python -m pytest tests/test_preprocess.py`. Only use words from the **training** tweets or general knowledge, never from dev/test data or labels. For a custom dictionary in code: `normalize_hinglish(text, variants={...})` or `HinglishNormalizer({...})`.

## Preserving sentiment information

- No word is deleted in step 1 apart from the retweet marker, the truncation marker, URLs and mentions. Negations (`nahi`, `nhi`, `not`, `never`, `mat`, `na`) are never removed; `nhi` and `nahin` are mapped to `nahi`, so they stay negations.
- Split contractions are rejoined *because* `didn ’ t` would otherwise leave a stray `t` that many tokenisers drop, losing the negation.
- Emojis, exclamation and question marks, digits and capitalisation are kept by default.
- Hashtag words are kept (`# NewIndia` -> `NewIndia`) because they often carry opinion.
- A tweet is never reduced to an empty string unless the input was empty; a tweet made only of a URL becomes `URL`.
- The pipeline is idempotent (`preprocess_text(preprocess_text(x)) == preprocess_text(x)`; tested).

## Leakage and reproducibility

Nothing is fitted or learned: the functions are deterministic, use no statistics from any split and never read labels. Train, dev and test are processed by the same function with the same options. The dictionary is a static file reviewed by hand. Vocabularies for the BiLSTM must still be built from the processed **train** text only.

## Data integration

`src/data.py` is unchanged. Wrappers in `src/preprocess.py`:

```python
from src.preprocess import preprocess_text, preprocess_frame, load_processed_split, tokenize

df = load_processed_split("train", mode="normalized")   # same rows, uid, label, text as load_split
df[["uid", "label", "text", "text_proc"]]              # `text` = original, `text_proc` = processed

# or, on any DataFrame from load_split
dv = preprocess_frame(load_split("dev"), mode="clean", out="text_clean")
tokens = tokenize(df.loc[0, "text_proc"])              # whitespace tokens for a BiLSTM
```

`preprocess_frame` returns a copy; the original columns, row order, uids and labels are untouched. The TF-IDF baseline does not import this module, so its behaviour is unchanged.

## Examples (real training tweets, from `docs/reading_sample.md`)

| Original | Processed (`normalized`) |
|---|---|
| `@ bilalamulla @ Official _ PIA PIA ko jo order mila tha unho ne vohi kia Ab yeh apne official a / c pr yeh to nhi likh s … https // t co / ygczAs3lsZ` | `USER PIA ko jo order mila tha unho ne vohi kia Ab yeh apne official a / c pr yeh to nahi likh s URL` |
| `@ aseeremuhabbat @ VaqasSiddiqui Shahbaz sharif ka bnaya hua hospital unke apne hi ilaaaj k qabil nahim 😂😂😂 what a joke` | `USER Shahbaz sharif ka bnaya hua hospital unke apne hi ilaaj k qabil nahi 😂😂😂 what a joke` |
| `Shandaar Zabardast Zindabad ! # NewIndia begins with Sabka Saath Sabka Vikas aur Sabka Vishwas 🇮🇳🙏 # ModiSwearingIn https // t co / chlpfRu2aQ` | `Shandaar Zabardast Zindabad ! NewIndia begins with Sabka Saath Sabka Vikas aur Sabka Vishwas 🇮🇳🙏 ModiSwearingIn URL` |
| `@ TEAM _ AFRIDI 1992 ko 1992 mai he chor du ab 2019 hai . Plz ye sab drame krne band kro . Aor kch kr k dkhao` | `USER 1992 ko 1992 mai he chor du ab 2019 hai . Plz ye sab drame krne band kro . Aor kuch kr k dkhao` |
| `“ Hasi toh phasi ” is so fake it ’ s actually agar wo hasi toh phasa tu harami .` | `“ Hasi toh phasi ” is so fake it's actually agar wo hasi toh phasa tu harami .` |
| `@ MajorPoonia Chup kar sale porkies thuuuu to you` | `USER Chup kar sale porkies thuu to you` |

Note `he` (Hindi "hai" in the fourth row) and `kr`, `dkhao` are intentionally left alone.

## Known limitations

- The dictionary has about 25 canonical forms. Hindi has far more spelling variants, and many remain unnormalised (`achhi`, `krne`, `dkhao`, `bnaya`). The effect on accuracy has **not been measured**: it needs the raw vs normalised experiment.
- Normalisation is text-level and does not use the gold language tags, so it cannot tell Hindi `he` from English `he`; that is why such words are excluded rather than handled by context. Using the `Hin` tags at training time could allow a larger dictionary, but the tags are not available for new text.
- Words cut off by Twitter before the final `…` stay cut (`hote`, `explos`).
- URL detection targets `t.co` links and joined URLs; a non-`t.co` URL with spaces around its dots (`example . com`) is only partly matched.
- A mention is matched as `@` plus a word, so `@ 5pm` would be treated as a mention.
- No Devanagari transliteration is done (only 59 train tweets contain Devanagari).
- Not tested on the full dataset (it was not available when this was written); real-data checks are in `tests/test_preprocess.py` and skip without the data.
