"""
Full data pipeline: download every historical season, save raw matches and
per-game standings, then build the processed training table (out.csv).

Usage:
    python -m prem_prediction.build_dataset

Outputs (in data/):
    historical_matches.csv    — one row per fixture, all seasons (Elo source)
    historical_standings.csv  — one row per team per game played
    out.csv                   — processed training data (Elo, form, labels)
"""

from .config import latest_completed_season
from .data_acquisition import FDCUKDataLoader
from .paths import MATCHES_FILE, STANDINGS_FILE, TRAINING_FILE, ensure_dirs
from .prepare_model_data import prepare

TRAIN_START = 1993


def main():
    ensure_dirs()
    train_end = latest_completed_season()  # e.g. 2025 for the 2025-26 season
    loader = FDCUKDataLoader()

    print(f"Downloading seasons {TRAIN_START}-{train_end} from football-data.co.uk...")
    matches = loader.get_all_matches(start=TRAIN_START, end=train_end)
    if matches.empty:
        print("No data downloaded — check your internet connection.")
        return

    matches.to_csv(MATCHES_FILE, index=False)
    print(f"\nRaw matches saved to {MATCHES_FILE} ({len(matches)} fixtures)")

    standings = loader.standings_from_all_matches(matches)
    standings.to_csv(STANDINGS_FILE, index=False)
    print(f"Standings saved to {STANDINGS_FILE} ({len(standings)} rows)")

    print("\nAdding Elo, form, and labels...")
    out = prepare(standings, matches)
    out.to_csv(TRAINING_FILE, index=False)

    n_seqs = (
        out.groupby(["season", "teamName"])
        .filter(lambda g: len(g) >= 38)
        .groupby(["season", "teamName"])
        .ngroups
    )
    print(f"Processed data saved to {TRAINING_FILE}")
    print(f"  Rows: {len(out)}")
    print(f"  Complete 38-game sequences available for training: {n_seqs}")
    print(f"  Seasons covered: {sorted(out['season'].unique())}")
    print("\nNext step: python -m prem_prediction.model_train")


if __name__ == "__main__":
    main()
