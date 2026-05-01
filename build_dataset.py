"""
Full data pipeline: download all historical seasons, compute standings,
add ELO and labels, save to out.csv for model training.

Usage:
    python build_dataset.py

Outputs:
    historical_standings.csv  — raw per-game standings for seasons 1993–2024
    out.csv                   — processed training data (with ELO, form, labels)
"""

import pandas as pd
from data_aquisition import FDCUKDataLoader
from prepare_model_data import prepare

# Seasons used for training: 1993-94 through 2024-25.
# Season 2025 (2025-26) is the prediction target — excluded from training.
TRAIN_START = 1993
TRAIN_END   = 2024

STANDINGS_FILE = "historical_standings.csv"
TRAINING_FILE  = "out.csv"


def main():
    loader = FDCUKDataLoader()

    print(f"Downloading seasons {TRAIN_START}-{TRAIN_END} from football-data.co.uk...")
    df = loader.get_all_seasons(start=TRAIN_START, end=TRAIN_END)

    if df.empty:
        print("No data downloaded — check your internet connection.")
        return

    print(f"\nDownloaded {len(df)} total rows across {df['season'].nunique()} seasons.")
    df.to_csv(STANDINGS_FILE, index=False)
    print(f"Raw standings saved to {STANDINGS_FILE}")

    print("\nAdding ELO, form, and labels...")
    out = prepare(df)
    out.to_csv(TRAINING_FILE, index=False)

    total_seqs = out.groupby(["season", "teamName"]).filter(lambda g: len(g) >= 38)
    n_seqs = total_seqs.groupby(["season", "teamName"]).ngroups

    print(f"Processed data saved to {TRAINING_FILE}")
    print(f"  Rows: {len(out)}")
    print(f"  Complete 38-game sequences available for training: {n_seqs}")
    print(f"  Seasons covered: {sorted(out['season'].unique())}")
    print(f"\nNext step: python model_train.py")


if __name__ == "__main__":
    main()
