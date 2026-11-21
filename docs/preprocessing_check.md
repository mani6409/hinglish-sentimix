# Normalization check (Phase 3, step 4)

What the `normalized` preprocessing changes, measured on the **train** split only (14,000 tweets). Nothing here is fitted on dev or test.

## Numbers

| | raw | clean | normalized |
|---|---|---|---|
| unique words (lower-cased) | 50,181 | 32,566 | 32,302 |

- `clean` removes about 35% of the unique words (mostly mentions, URLs and hashtags).
- `normalized` changes **2,048 of 14,000 tweets (14.6%)**, with 2,544 word changes in total and 600 distinct before/after pairs. It merges only 264 more unique words than `clean` (0.8%).
- Every change keeps the number of words in the tweet, so word positions are unchanged.
- The full list of changes with counts is in `results/normalization_changes_train.csv`.

## What the changes look like
The most frequent changes are the intended ones:
`nhi > nahi` (689), `kyu > kyun` (145), `krta > karta` (64), `acha > achha` (60), `nahin > nahi` (58), `kuchh > kuch` (42), `bohot / bohat / bahot / bhot > bahut` (over 130), `pyar > pyaar` (35), `thik > theek` (33). Upper case and title case are kept (`NHI > NAHI`, `Kyu > Kyun`).
Reading the changed words I did not find a real English word that was turned into something wrong. The repeated-letter rule gave the expected results on English words (`Misss > Miss`, `tweeet > tweet`, `Happppppy > Happy`).

## Known failure cases
1. **Repeated-letter spellings miss the dictionary.** Runs of the same letter are cut to 2 (on purpose, so that `good` and `yaar` survive). The dictionary is applied after that, so `nhii` and `nhiii` stay `nhii` (not `nahi`), `yarrr` becomes `yarr` (not `yaar`), and `hiiii` becomes `hii` (not `hi`). Such variants stay separate words.
2. **Hashtags and names lose letters.** `ICCCricketWorldCup2019` becomes `ICCricketWorldCup2019`, and `xxx` becomes `xx`.
3. **Ambiguous short words** such as `III` become `II`.
4. **Other scripts.** Repeated letters in other scripts are shortened too (one Thai tweet changes).
5. **Small effect.** Because only 0.8% of the vocabulary is merged, the normalized and clean models see almost the same text. This is the likely reason normalization did not help (see `results/bilstm_test_analysis/bilstm_final_analysis.md`).

I did not change the code, because the frozen experiments depend on it. A possible fix for case 1 is to collapse runs to 1 letter only for the dictionary lookup. That would need new runs.

## Manual check of 100 changed tweets
`results/normalization_sample_100.csv` has 100 random train tweets that the normalizer changed (random seed 42), with the clean text, the normalized text and the list of changes.

**Result: 99 of 100 correct, 1 wrong.** This was read through by Claude, not by a team member, so it still needs a human to confirm it (the `ok (y/n)` and `comment` columns are filled in and can be changed).
- **Wrong (1):** tweet 25, the acronym `PPP` (a party name) becomes `PP`.
- **Correct but only partly normalized (13):** a long run of letters was cut to two letters, but the word was not merged with its common form, for example `Looool` becomes `Lool` (not `lol`), `Yess` (not `yes`), `Yrr` (not `yaar`). This is failure case 1 above. These are marked `ok, partial` in the comment column.
- All dictionary changes in the sample (`nhi`, `kyu`, `bohot`, `krte`, `thik`, `acha`, `pyar` and so on) were correct.
- The sample only has tweets that the normalizer changed (14.6% of train), so it measures how often a change is right, not how often a variant is missed.
