"""
Gameweek refresh: pull the latest results for the season in progress, regenerate
title probabilities, and rebuild the README GIF + interactive dashboard.

Run this whenever a gameweek finishes:

    python -m prem_prediction.update

It does NOT retrain the model — that only needs to happen once a season, after it
completes (see add_season + model_train). This just re-scores the live season
with the existing model.
"""

from . import kalshi, season_predict, viz


def main() -> None:
    print("=== Gameweek refresh ===")
    season_predict.main()   # downloads live season, writes outputs/predictions.json
    print("\nRefreshing Kalshi market comparison...")
    try:
        kalshi.build_kalshi_cache()  # data/kalshi_current.csv (free public API)
    except Exception as e:
        print(f"  Kalshi refresh skipped ({e}); comparison GIF will use any cached data.")
    print("\nRebuilding visualisations...")
    viz.build_all()         # title_race.gif + model_vs_kalshi.gif + dashboard.html
    print("\nDone. Updated predictions.json, title_race.gif, model_vs_kalshi.gif, dashboard.html")


if __name__ == "__main__":
    main()
