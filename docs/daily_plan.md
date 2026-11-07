# Hinglish Sentiment Analysis - Team Daily Work Plan

SemEval-2020 Task 9 (SentiMix Hinglish) | Team: Mani, Monis, Ekta, Akansha | 7 Nov to 20 Nov 2026

**Deadline:** 20 Nov 2026, 11:59 pm IST. **Deliverables:** project report and one GitHub repo per group.

## Who does what

| Person | Part | Builds | Writes | Difficulty |
|---|---|---|---|---|
| Mani | Part 1: Data and analysis | Repo owner. Setup, data loading, EDA, code-mixing index (CMI), TF-IDF baseline, CMI-bucket ablation. | Dataset and EDA section; CMI ablation section; report assembly and README. | Easy to Medium, then a Hard analysis task |
| Monis | Part 3: Transformers | Shared training script, mBERT fine-tuning, XLM-R fine-tuning. | Transformer methodology and results sections. | Hard (GPU time and tuning) |
| Ekta | Part 2: Text processing and BiLSTM | Preprocessing, transliteration normalisation (raw and normalised data), BiLSTM with word then word+char embeddings. | Preprocessing section; normalisation ablation section. | Hard (normalisation) plus Medium (BiLSTM) |
| Akansha | Part 4: Indic model, LLM and error analysis | MuRIL/IndicBERT fine-tuning, zero-shot LLM prompting, error analysis. | LLM comparison section; error analysis; limitations and future work. | Medium plus Hard |

Pair review: Mani and Ekta review each other's work. Monis and Akansha review each other's work. On Day 7 (13 Nov) the pairs swap so everyone learns a second area.

## Daily plan

### Day 1 - Sat 7 Nov: Setup and reading

| Person | Today's work |
|---|---|
| Mani | Create the GitHub repo and add all three teammates. Create folders (data, notebooks, src, configs, results, docs, reports), .gitignore, requirements.txt, README skeleton. Download the SentiMix Hinglish data and tell the team its file format. |
| Monis | Read paper [1] (SemEval overview). Confirm GPU access (Colab, Kaggle or university). Install PyTorch and transformers and run a tiny test that loads a model. |
| Ekta | Read paper [1]. Learn what transliteration and normalisation mean for Romanised Hindi. Write down 10 words you know that have several spellings. |
| Akansha | Read paper [1]. Decide which LLM you can use and any cost limit. Make one test call and confirm it works. |

**End of day check:** Repo exists, everyone can push. Everyone can say what code-mixing is and what the three labels are.

### Day 2 - Sun 8 Nov: Foundations

| Person | Today's work |
|---|---|
| Mani | Write src/metrics.py (the official metric from paper [1]), the seed function, and results/experiment_log.csv. Load the data into one row per tweet and push the loader. |
| Monis | Read the XLM-R paper [2] (intro, architecture, results). Write a half-page summary. Print how mBERT and XLM-R split 5 sample tweets into tokens. |
| Ekta | Read paper [2] intro and write a half-page summary. Using the data Mani shared, collect 30 real spelling variants from tweets. |
| Akansha | Read paper [2] intro and write a half-page summary. Draft the first zero-shot prompt (task, three labels, no examples, label only). |

**End of day check:** Four summaries in docs/. Team writes one shared problem statement paragraph in a 20-minute call.

### Day 3 - Mon 9 Nov: Data, trials, scripts

| Person | Today's work |
|---|---|
| Mani | EDA part 1: label distribution, tweet lengths, share of Hindi, English and other tokens. Start the EDA notebook. |
| Monis | Write the shared training script (model name as a parameter, config file for settings). Test it on a 200-tweet sample. |
| Ekta | Trial two normalisation methods on 200 tweets (for example a variant dictionary vs Devanagari transliteration). Note what each gets wrong. |
| Akansha | Test the prompt on 50 dev tweets. Read the outputs. Write the parser that maps answers to the three labels and counts invalid ones. |

**End of day check:** Mani's data loader is merged and used by the others.

### Day 4 - Tue 10 Nov: CMI and methods chosen

| Person | Today's work |
|---|---|
| Mani | Compute CMI per tweet. Plot the CMI distribution and CMI vs label. Read 50 tweets per class and write notes. Finish the EDA notebook and share the data file with a CMI column. |
| Monis | Finish the training script. Set max sequence length from Mani's EDA. Plan the learning-rate search (small values such as 1e-5 to 5e-5). |
| Ekta | Pick the normalisation method with the team. Implement src/preprocess.py with a few tests. Produce the first raw and normalised files. |
| Akansha | Try 2 to 3 prompt variants on the same 50 tweets and log each. Load MuRIL/IndicBERT with Monis's script on the sample. |

**End of day check:** EDA notebook done. Normalisation method chosen. Training script runs end to end.

### Day 5 - Wed 11 Nov: Baselines and first fine-tuning

| Person | Today's work |
|---|---|
| Mani | TF-IDF with logistic regression: word n-grams, then character n-grams. Report dev scores and log them. |
| Monis | mBERT: learning-rate search on dev only, then the first seed. |
| Ekta | BiLSTM with word embeddings on the raw data. Vocabulary from train only, early stopping on dev. |
| Akansha | MuRIL/IndicBERT: first run with the same settings as Monis's mBERT run. |

**End of day check:** Four rows in the experiment log. Team sync: each person explains their result without reading code.

### Day 6 - Thu 12 Nov: Seeds and review

| Person | Today's work |
|---|---|
| Mani | Finish TF-IDF. Start the dataset and EDA section of the report. Review Ekta's preprocessing pull request. |
| Monis | mBERT with 3 seeds. Record mean and spread. |
| Ekta | BiLSTM with word + character embeddings. Run 3 seeds. Review Mani's EDA notebook. |
| Akansha | MuRIL/IndicBERT learning-rate search and seeds. Review Monis's training script. |

**End of day check:** Baselines and mBERT finished with seeds. All pull requests reviewed by the partner.

### Day 7 - Fri 13 Nov: Checkpoint 1 and reviewer swap

| Person | Today's work |
|---|---|
| Mani | Define CMI buckets (low, medium, high) so each has enough tweets. Write the script that scores any predictions file by bucket. Swap reviewer: review Monis's work. |
| Monis | Start XLM-R runs. Swap reviewer: review Ekta's work. |
| Ekta | Finalise the normalised dataset. Rerun the best BiLSTM on normalised data (3 seeds). Swap reviewer: review Mani's work. |
| Akansha | Finish MuRIL/IndicBERT seeds. Start the full zero-shot LLM run on the dev set. Swap reviewer: review Ekta's or Monis's work. |

**End of day check:** Checkpoint: are we on schedule? If GPU time is short, agree now to cut to 2 seeds and note it in the report.

### Day 8 - Sat 14 Nov: Experiments continue

| Person | Today's work |
|---|---|
| Mani | Test the bucket script on baseline predictions. Fix any bucket with too few tweets. |
| Monis | XLM-R seeds. |
| Ekta | Build the raw vs normalised table for BiLSTM. Start the preprocessing section. |
| Akansha | Finish the LLM dev run. Record invalid-answer rate, cost, time and the exact model name and date. |

**End of day check:** LLM numbers logged. Predictions files saved in results/ for every model.

### Day 9 - Sun 15 Nov: Transformer results and ablation runs

| Person | Today's work |
|---|---|
| Mani | Run the CMI-bucket analysis on every model's predictions available so far. |
| Monis | Finish XLM-R. Produce one table: mBERT, XLM-R, MuRIL/IndicBERT (with Akansha's numbers) with mean and spread. |
| Ekta | Run the best transformer on raw vs normalised data (use Monis's script, same seeds and settings). |
| Akansha | Build the confusion matrix for the best model. Find the most confused class pair. |

**End of day check:** All dev results in one place. Best models identified by the team.

### Day 10 - Mon 16 Nov: Checkpoint 2 and ablation write-up

| Person | Today's work |
|---|---|
| Mani | Finish the CMI-bucket tables and plots. Write what happened, a possible reason, and how sure we are. |
| Monis | Write the transformer methodology and results section. |
| Ekta | Finish the normalisation ablation table. Write the interpretation, including whether differences are within seed noise. |
| Akansha | Sample at least 50 misclassified tweets and label each by cause (sarcasm, slang, mixed script, negation, label noise, other). |

**End of day check:** Checkpoint: cut scope if behind. Freeze the list of final models for the test run.

### Day 11 - Tue 17 Nov: Final test runs and analysis

| Person | Today's work |
|---|---|
| Mani | Collect all results tables and figures into results/. Finalise the dataset and EDA section. |
| Monis | Run the chosen transformers on the test set once. Save predictions. |
| Ekta | Run the chosen BiLSTM and the normalised variant on the test set once. Write the preprocessing section. |
| Akansha | Run the chosen LLM prompt on the test set once. Finish the error analysis table with 2 to 3 anonymised examples per category. |

**End of day check:** Test set touched once only. Nobody changes settings after seeing test numbers.

### Day 12 - Wed 18 Nov: Report drafting

| Person | Today's work |
|---|---|
| Mani | Assemble the report in the official format (once Chhavi shares it). Write abstract, introduction and related work. |
| Monis | Polish methodology and transformer results. Make the final figures. |
| Ekta | Finish preprocessing and ablation 1 sections. Check references. |
| Akansha | Finish LLM and error analysis sections. Write limitations and future work. |

**End of day check:** Full first draft of the report exists.

### Day 13 - Thu 19 Nov: Review and reproducibility

| Person | Today's work |
|---|---|
| Mani | Merge everything. Finalise the README and the contribution table. Check every number in the report against the experiment log. |
| Monis | Read the whole report and fix unclear parts. |
| Ekta | Read the whole report. Check that all papers are cited and all text is in our own words. |
| Akansha | Clone the repo on a clean setup and follow the README exactly. Report anything that breaks. |

**End of day check:** A stranger could rerun the main results. Report and repo ready.

### Day 14 - Fri 20 Nov: Submit

| Person | Today's work |
|---|---|
| Mani | Upload the report using the link Chhavi shares. Confirm the repo link is correct and public or accessible. |
| Monis | Final check of figures and tables in the PDF. |
| Ekta | Final proofread. |
| Akansha | Confirm the repo is clean and the README works. Keep this day as buffer. |

**End of day check:** Submit before the 11:59 pm IST deadline. Aim to finish by midday.

## Team rules

- One repo for the group. Use branches and pull requests. Nobody merges their own pull request.
- Same seeds, same splits, same metric for every model.
- Log every experiment in results/experiment_log.csv with one line on what we learned.
- Never tune on the test set. Decisions use the dev set; the test set is run once at the end.
- Keep large data and model weights out of Git.
- Write in your own words and cite every source.
- Each person must be able to explain any part of the project, not only their own.
- Daily: post a short update in the team chat (done, doing, blocked). A 20-minute sync every 2 days: explain your work without reading code.

## Notes

- Times are planning guesses. If a day slips, tell the team the same day.
- Dataset sizes and the official metric are confirmed from paper [1] and the data on Days 1 to 3.
- The report format has not been shared yet. Day 12 follows the official format once it arrives.
- Part assignment (who does which part) is a default. Swap before starting if anyone prefers a different part.
