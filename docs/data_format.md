# SentiMix Hinglish data: source and format

## Source
- Official release: Zenodo record 3974927 (https://zenodo.org/record/3974927), file `Semeval_2020_task9_data.zip` (2,807,728 bytes), licence CC-BY 4.0.
- MD5 checksum: `4d93553c213b3a966b6d55b7a7c03be7` (verified after download).
- Task site: https://ritual-uh.github.io/sentimix2020/
- Data is not committed to Git (see `.gitignore`). Each teammate downloads it and extracts it into `data/raw/`.

```bash
mkdir -p data/raw/zenodo_download && cd data/raw/zenodo_download
curl -L -o Semeval_2020_task9_data.zip "https://zenodo.org/api/records/3974927/files/Semeval_2020_task9_data.zip/content"
md5 -q Semeval_2020_task9_data.zip     # expect 4d93553c213b3a966b6d55b7a7c03be7 (use md5sum on Linux)
unzip Semeval_2020_task9_data.zip -d extracted
```

## Files we use (folder `Hinglish/`)
| File | Content |
|---|---|
| `Hinglish_train_14k_split_conll.txt` | Training tweets with labels |
| `Hinglish_dev_3k_split_conll.txt` | Development tweets with labels |
| `Hinglish_test_unalbelled_conll_updated.txt` | Test tweets, no labels |
| `Hinglish_test_labels.txt` | Test labels (`Uid,Sentiment`), joined by tweet id |

The `Spanglish/` folder is not part of this project.

## Format (CoNLL, one token per line)
Each tweet starts with a header line, then one token per line, with a tab between the token and its language tag. Tweets are separated by a blank line.

```
meta<TAB>4330<TAB>neutral        <- tweet id and sentiment (test file has no sentiment)
nen<TAB>Eng                      <- token, language tag
...
```

Language tags: `Hin` (Hindi), `Eng` (English), `O` (other: mentions, URLs, punctuation, symbols), `EMT` (emoticon).
Labels: `positive`, `negative`, `neutral`.

To get one row per tweet, join the tokens of each tweet with spaces and keep the id, label and the list of language tags (needed for the code-mixing index).

## Counts (computed from the files)
| Split | Tweets | Positive | Negative | Neutral |
|---|---|---|---|---|
| Train | 14,000 | 4,634 | 4,102 | 5,264 |
| Dev | 3,000 | 982 | 890 | 1,128 |
| Test | 3,000 | labels in separate file | | |

- Average tweet length is about 26 tokens; the longest is 56 tokens in train.
- Train token tags: Hin 169,893, Eng 121,412, O 73,735, EMT 520.
- Classes are fairly balanced, with neutral the largest.

## Things to check during EDA
- The language tags in the data may be wrong in places (some tokens look misassigned), so treat them as noisy when computing the code-mixing index.
- `Hinglish_test_labels.txt` has 3,000 lines including the header; confirm it matches the 3,000 test tweets before using it.
- Use the test labels only for the single final test run.
