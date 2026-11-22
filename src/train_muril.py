"""Fine-tune MuRIL (google/muril-base-cased) for Hinglish sentiment. Issue #7.

Usage (GPU strongly recommended, see docs/indian_transformer_results.md):
    python -m src.train_muril                 # seed 42, writes results/muril_dev_predictions.csv
    python -m src.train_muril --seed 43       # other seeds -> results/predictions/muril_s43_seed43_dev.csv
    python -m src.train_muril --inspect-only  # tokenizer report only
"""
from src.indian_transformer import main_for

CHECKPOINT = "google/muril-base-cased"

if __name__ == "__main__":
    main_for("muril", CHECKPOINT)
