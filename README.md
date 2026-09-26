# Premier League Title Prediction

An LSTM model that estimates each club's probability of winning the Premier League
title, updated week-by-week through the season. Outputs are **Kalshi-style
percentages** — one probability per team per gameweek, softmaxed across the 20 clubs
so they always sum to 100%.

![Title race animation](outputs/title_race.gif)

> The GIF above regenerates every gameweek. An interactive version (crosshair
> tooltips, per-team table, dark mode) is built at `outputs/dashboard.html`.

---

## How It Works

### Data
Match results are downloaded from **[football-data.co.uk](https://www.football-data.co.uk/)**
— free, no API key, every PL season from 1993-94 onwards. From the raw fixtures the
pipeline derives two things:

- **`data/historical_matches.csv`** — one row per fixture (the Elo engine's input)
- **`data/historical_standings.csv`** — one row per team per game played (points,
  goals, form, live position after every game)

### Features
Each row is a team's state after their Nth game of the season.

**Absolute** — `position`, `form` (last-5 as W=3/D=1/L=0), `won`/`draw`/`lost`,
`points`, `goalsFor`/`goalsAgainst`/`goalDifference`, `games_remaining`.

**Relative** — `points_per_game`, `points_gap_from_leader` (0 for the leader,
negative otherwise), `max_obtainable_points` (`points + games_remaining × 3`, the
title ceiling).

**`elo`** — a results-based strength rating (below).

### Elo — results-based, with prestige seeding
The Elo is a proper chained rating updated after **every match** across all history,
like [clubelo](http://clubelo.com/) / FiveThirtyEight:

```
expected_home = 1 / (1 + 10^((elo_away − (elo_home + home_adv)) / 400))
delta         = K × goal_diff_multiplier × (result − expected_home)
elo_home += delta ;  elo_away −= delta
```

Because it only ever sees match results, it is **completely competition-agnostic** —
a team is never penalised for playing in more competitions.

The context you *do* want — European pedigree and trophies — is folded in
**additively** at the start of each season, on top of the mean-reverted carryover:

```
season_start_elo = 1500 + 0.75 × (prev_end_elo − 1500)      # mean reversion
                 + european_qualification_bonus(prev_finish) # >0 even with no trophies
                 + trophy_bonus(prev_season)                 # stacks per trophy
```

- **European qualification bonus** rewards finishing high enough to reach Europe on
  its own: CL (~top 5) `+40`, EL (~6th) `+25`, ECL (~7th) `+15`.
- **Trophy bonus** stacks per trophy won: PL `+50`, CL `+50`, EL `+30`, FA `+20`,
  League Cup `+15`, ECL `+15`.

Every term is additive, so being in more competitions — or winning more of them —
can only ever *raise* a team's Elo. All constants live at the top of
[`src/prem_prediction/elo.py`](src/prem_prediction/elo.py) for tuning.

> **Why this replaced the old formula.** The previous "elo" divided trophies won by
> the number of competitions entered. Reaching Europe raised that denominator, so a
> side strong enough to qualify got a *lower* prior than an identical non-European
> trophy winner — it punished teams for being in more competitions. It also
> multiplied by `100 / current_position`, which just duplicated the `position`
> feature. Both problems are gone.

### Model
A two-layer LSTM with masking and dropout:

```
Masking → LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(1)
```

The output is a **raw logit** (no sigmoid). At prediction time the 20 teams' logits
are collected and passed through **softmax together**, so the model learns a 20-way
ranking (only one team can win) rather than 20 independent binaries. Trained with
`BinaryCrossentropy(from_logits=True)`.

**Partial-sequence training:** sequences are cut at gameweeks 5, 10, … 38 and
zero-padded, giving 8 training examples per team per season so the model makes
confident calls early, not just at the final whistle.

---

## Project Layout

```
src/prem_prediction/     # the package
  data_acquisition.py    #   download fixtures, build standings
  elo.py                 #   results-based Elo + prestige seeding
  trophy_data.py         #   historical trophy winners + bonus values
  prepare_model_data.py  #   form, labels, relative features, Elo merge
  model_train.py         #   LSTM training
  season_predict.py      #   current-season Kalshi predictions → predictions.json
  viz.py                 #   title_race.gif + dashboard.html
  build_dataset.py       #   full historical rebuild
  add_season.py          #   append a finished season
  update.py              #   gameweek refresh (predict + viz)
  config.py / paths.py   #   season detection, filesystem paths
scripts/update.py        # thin wrapper: python scripts/update.py
data/                    # matches, standings, out.csv
models/                  # best_model.h5, scaler.joblib
outputs/                 # predictions.json, title_race.gif, dashboard.html
```

The current season is detected automatically from the date (`config.current_season_start_year`),
so nothing is hard-coded to a particular year.

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .          # makes `python -m prem_prediction.*` work
```

## Running the Pipeline

```bash
# 1. Build the training dataset (downloads 1993-94 → latest completed season)
python -m prem_prediction.build_dataset

# 2. Train the model
python -m prem_prediction.model_train

# 3. Predict the current season + build the graph/dashboard
python -m prem_prediction.update
```

`update` is the one you re-run each gameweek: it pulls the latest results, re-scores
every completed gameweek, and rebuilds `outputs/title_race.gif` and
`outputs/dashboard.html`. It does **not** retrain — that only happens once a season,
after it completes.

## Adding a Finished Season

When a season ends, append it and retrain:

```bash
python -m prem_prediction.add_season --season 2025 \
    --pl Arsenal --fa "Man City" --lc "Man City" \
    --cl None --el "Aston Villa" --ecl "Crystal Palace"
python -m prem_prediction.model_train
```

Team names must match the football-data.co.uk convention exactly
(`"Man City"`, `"Nott'm Forest"`, …).

## Automated Weekly Refresh

A scheduled cloud agent can run the refresh for you every week and push the updated
graph to the repo. It needs GitHub connected to your Claude account
(`/web-setup` or https://claude.ai/connect-github) and the code pushed to `main`.

---

## Configuration Notes

- `constants.py` is gone. The legacy football-data.org API key (only used by the
  optional `FootballDataAPI` wrapper) is now read from the `FOOTBALL_DATA_API_KEY`
  environment variable — see `.env.example`. The main pipeline needs no key.
