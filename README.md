# Premier League Title Prediction

An LSTM-based model that estimates each club's probability of winning the Premier League title, updated week-by-week throughout the season. Outputs are formatted in **Kalshi-style percentages** — one probability per team per gameweek.

---

## How It Works

### Data
Historical match results are downloaded from **[football-data.co.uk](https://www.football-data.co.uk/)** — a free source covering every Premier League season from 1993-94 onwards (~32 seasons, ~12,000 team-game snapshots). No API key required.

From the raw match results the pipeline computes cumulative per-game standings for each team: points, goals, form, and position after every game played.

### Features
Each row in the training set represents a team's state after their Nth game of the season:

| Feature | Description |
|---|---|
| `position` | Current league position |
| `form` | Numerical score of last 5 results (W=3, D=1, L=0) |
| `won` / `draw` / `lost` | Cumulative game counts |
| `points` | Cumulative points |
| `goalsFor` / `goalsAgainst` / `goalDifference` | Cumulative goal stats |
| `games_remaining` | 38 − gameweek |
| `elo` | **Custom ELO feature** (see below) |

### ELO Feature
A season-opening strength signal derived from the previous season:

```
elo_raw = prev_position / (1 + trophies_won / possible_trophies)
feature  = 1 / elo_raw         # higher = stronger team
```

- `prev_position`: final league position from the prior season (20 for promoted teams)
- `trophies_won`: PL + FA Cup + League Cup + European trophies won that season
- `possible_trophies`: 3 (domestic) + 1 if the team was in European competition (top-7 finish → Europe)

The inverse is taken so that **higher ELO = stronger team** — consistent with the other features where more is better.

### Model
A two-layer **LSTM** with masking and dropout, trained to predict whether a team wins the league from their cumulative gameweek-by-gameweek sequence:

```
Masking → LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(1, sigmoid)
```

Training label: `won_league = 1` for the title winner, `0` for all other teams. Probabilities are normalised across all 20 teams to sum to 100% at each gameweek.

---

## Setup

```bash
pip install tensorflow scikit-learn pandas numpy requests joblib
```

---

## Running the Pipeline

### Step 1 — Build the training dataset
Downloads all seasons 1993-94 → 2024-25, computes standings, adds ELO and labels.

```bash
python build_dataset.py
```

Outputs:
- `historical_standings.csv` — raw per-game standings
- `out.csv` — processed training data ready for the model

### Step 2 — Train the model
```bash
python model_train.py
```

Outputs:
- `best_model.h5` — best LSTM checkpoint (monitored on validation loss)
- `scaler.joblib` — fitted StandardScaler

### Step 3 — Generate 2025-26 predictions
```bash
python season_predict.py
```

Prints Kalshi-style win probabilities for every completed gameweek of the current season, e.g.:

```
=== Gameweek 15 ===
  Liverpool              41.3%
  Arsenal                22.1%
  Man City               14.8%
  Chelsea                 7.2%
  ...
```

---

## Adding a Newly Finished Season

When a season ends, run `add_season.py` to append its standings to the training set and patch `trophy_data.py` with that year's trophy winners. Then retrain.

**Interactive (will prompt for each trophy winner):**
```bash
python add_season.py --season 2025
```

**Non-interactive:**
```bash
python add_season.py --season 2025 \
    --pl Liverpool --fa "Crystal Palace" --lc Liverpool \
    --cl None --el Tottenham --ecl None
```

Then retrain:
```bash
python model_train.py
python season_predict.py
```

---

## Trophy Data

`trophy_data.py` holds hard-coded historical trophy winners (PL, FA Cup, League Cup, Champions League, Europa League, Conference League) from 1992-93 to 2024-25, keyed by **season start year**.

All team names match the football-data.co.uk convention exactly:
`"Man City"`, `"Man United"`, `"Nott'm Forest"`, `"Sheffield United"`, etc.

---

## File Reference

| File | Purpose |
|---|---|
| `build_dataset.py` | Downloads all history and builds `out.csv` |
| `data_aquisition.py` | `FDCUKDataLoader` — free historical CSV fetcher |
| `trophy_data.py` | Historical trophy data + ELO formula |
| `prepare_model_data.py` | ELO computation, form conversion, label assignment |
| `model_train.py` | LSTM training |
| `season_predict.py` | 2025-26 Kalshi predictions |
| `add_season.py` | Extend the dataset after a season finishes |
| `constants.py` | API key for football-data.org (legacy) |
