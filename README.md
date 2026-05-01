# Premier League Title Prediction

An LSTM-based model that estimates each club's probability of winning the Premier League title, updated week-by-week throughout the season. Outputs are formatted in **Kalshi-style percentages** — one probability per team per gameweek.

---

## How It Works

### Data
Historical match results are downloaded from **[football-data.co.uk](https://www.football-data.co.uk/)** — a free source covering every Premier League season from 1993-94 onwards (~32 seasons). No API key required.

From the raw match results the pipeline computes cumulative per-game standings for each team: points, goals, form, and position after every game played.

### Features
Each row represents a team's state after their Nth game of the season. Features are split into three groups:

**Absolute stats** — the team's raw accumulated numbers:

| Feature | Description |
|---|---|
| `position` | Current league position |
| `form` | Numerical score of last 5 results (W=3, D=1, L=0) |
| `won` / `draw` / `lost` | Cumulative game counts |
| `points` | Cumulative points |
| `goalsFor` / `goalsAgainst` / `goalDifference` | Cumulative goal stats |
| `games_remaining` | 38 − gameweek |

**Relative stats** — how the team stands compared to the rest of the table at that gameweek:

| Feature | Description |
|---|---|
| `points_per_game` | `points / gameweek` — scoring pace, comparable across all stages |
| `points_gap_from_leader` | Team's points minus the leader's points (0 for leader, negative otherwise) |
| `max_obtainable_points` | `points + games_remaining × 3` — mathematical title ceiling |

These give the model context within each snapshot, so a team 8 points clear looks very different from a team 8 points adrift, even if their absolute stats are similar.

**ELO** — a dynamic prior-season strength signal:

| Feature | Description |
|---|---|
| `elo` | Season-opening quality score, updated each gameweek by live position |

### ELO Feature
Encodes how strong a team was entering the season, anchored to their previous year's performance:

```
base_elo = 1 / (prev_position / (1 + trophies_won / possible_trophies))
elo      = base_elo × (100 / current_position)
```

- `prev_position`: final league position from the prior season (20 for promoted teams)
- `trophies_won`: PL + FA Cup + League Cup + European trophies won that season
- `possible_trophies`: 3 domestic + 1 if in European competition (top-7 finish → Europe)
- `current_position`: live table position at that gameweek — makes ELO dynamic throughout the season

Higher ELO = stronger team. A title-winning, trophy-laden side that is currently top of the table gets the highest possible score.

### Model
A two-layer **LSTM** with masking and dropout:

```
Masking → LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(1)
```

The output layer produces a **raw logit** (no sigmoid). At prediction time, logits from all 20 teams are collected and passed through **softmax** together, treating the problem as a 20-way ranking rather than 20 independent binary predictions. This ensures the model inherently knows only one team can win.

Loss during training: `BinaryCrossentropy(from_logits=True)` — mathematically identical to sigmoid + BCE but numerically more stable.

### Partial-Sequence Training
The model is trained on sequences cut off at gameweeks 5, 10, 15, 20, 25, 30, 35, and 38 — with zero-padding for the remaining weeks. This gives the model 8 training examples per team per season instead of 1, teaching it to make confident predictions from early in the season rather than only at the end.

---

## Setup

```bash
pip install tensorflow scikit-learn pandas numpy requests joblib
```

---

## Running the Pipeline

### Step 1 — Build the training dataset
Downloads all seasons 1993-94 → 2024-25, computes standings, adds all features and labels.

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
  Arsenal                41.3%
  Liverpool              22.1%
  Man City               14.8%
  Chelsea                 7.2%
  ...
```

---

## Adding a Newly Finished Season

When a season ends, run `add_season.py` to append its standings to the training set and patch `trophy_data.py` with that year's trophy winners. The new features are computed automatically — no other changes needed.

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

All team names must match the football-data.co.uk convention exactly:
`"Man City"`, `"Man United"`, `"Nott'm Forest"`, `"Sheffield United"`, etc.

---

## File Reference

| File | Purpose |
|---|---|
| `build_dataset.py` | Downloads all history and builds `out.csv` |
| `data_aquisition.py` | `FDCUKDataLoader` — free historical CSV fetcher |
| `trophy_data.py` | Historical trophy data + ELO formula |
| `prepare_model_data.py` | ELO, relative features, form conversion, label assignment |
| `model_train.py` | LSTM training with partial-sequence augmentation |
| `season_predict.py` | 2025-26 Kalshi predictions via softmax across teams |
| `add_season.py` | Extend the dataset after a season finishes |
| `constants.py` | API key for football-data.org (legacy) |
