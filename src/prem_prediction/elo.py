"""
Results-based Elo rating with prestige seeding.

WHY THIS EXISTS
---------------
The previous "elo" was a heuristic:

    base = (1 + trophies_won / possible_trophies) / prev_position
    elo  = base * (100 / current_position)

That had two problems:

1.  The trophy term is a win *rate* (trophies / opportunities). Qualifying for
    Europe raised the denominator from 3 to 4, so a side strong enough to reach
    Europe got a *lower* prior than an identical non-European trophy winner —
    it punished teams for being in more competitions.
2.  The `* (100 / current_position)` term is almost perfectly collinear with the
    `position` feature the model already has, so it added noise, not signal.

THIS VERSION
------------
A proper chained Elo, updated after every match across all history (à la
clubelo / FiveThirtyEight). It only ever sees match results, so it is completely
competition-agnostic — the "more competitions" penalty cannot exist.

The prestige context the user asked for is folded in *additively* at the start
of each season, on top of the rating carried over from the previous year:

    season_start_elo = MEAN + REGRESSION * (prev_end_elo - MEAN)   # mean reversion
                     + european_qualification_bonus(prev_finish)   # >0 even w/ no trophies
                     + trophy_bonus(prev_season)                   # stacks per trophy

Because every term is additive, being in more competitions (or winning more
trophies) can only ever raise a team's Elo. Finishing high enough to reach
Europe is rewarded on its own, exactly as requested.
"""

import math

import numpy as np
import pandas as pd

from .paths import SQUAD_VALUE_FILE
from .trophy_data import trophy_bonus

# --- Core match-Elo constants ---
MEAN_ELO = 1500.0          # league-average anchor
NEW_TEAM_ELO = 1425.0      # promoted / never-before-seen teams enter below average
HOME_ADVANTAGE = 65.0      # Elo points added to the home side's expectation
K_FACTOR = 20.0            # base update size per match
SEASON_REGRESSION = 0.75   # fraction of last season's deviation from mean carried over

# --- Prestige seeding: European qualification bonus by previous-season finish ---
# Rewards finishing high enough to reach Europe, even with zero trophies.
CL_QUALIFY_BONUS = 40.0    # ~top 5  -> Champions League
EL_QUALIFY_BONUS = 25.0    # ~6th    -> Europa League
ECL_QUALIFY_BONUS = 15.0   # ~7th    -> Conference League

# --- Prestige seeding: squad market value ---
# Elo points added per within-season z-score of log(squad market value). Using the
# z-score of the log value makes this RELATIVE to the other squads that season, so
# it is immune to transfer-fee inflation (a 2008 league and a 2026 league are scored
# on the same relative scale). Only available from ~2005 (see squad_value.py); older
# seasons get 0, so Elo stays defined for all history. Top squads land ~+150, the
# cheapest ~-140, widening the gap between the elite and the rest before kickoff.
VALUE_ELO_SCALE = 80.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score for A vs B (rating_b should already include any HFA)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def goal_diff_multiplier(goal_diff: int) -> float:
    """
    World-Football-Elo margin-of-victory index. A draw (gd=0) keeps K unchanged,
    bigger wins scale it up, so a 4-0 moves ratings more than a 1-0.
    """
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0  # gd=3 -> 1.75, gd=4 -> 1.875, ...


def european_qualification_bonus(prev_position: int | None) -> float:
    """Additive Elo bonus for having qualified for Europe by league finish."""
    if prev_position is None:
        return 0.0
    if prev_position <= 5:
        return CL_QUALIFY_BONUS
    if prev_position == 6:
        return EL_QUALIFY_BONUS
    if prev_position == 7:
        return ECL_QUALIFY_BONUS
    return 0.0


def _final_positions_by_season(standings_df: pd.DataFrame) -> dict[int, dict[str, int]]:
    """{season: {team: final_league_position}} derived from the standings frame."""
    result: dict[int, dict[str, int]] = {}
    for season, sdf in standings_df.groupby("season"):
        finals = (
            sdf.sort_values("gameweek").groupby("teamName").last().reset_index()
        )
        finals = finals.sort_values(
            ["points", "goalDifference", "goalsFor"], ascending=[False, False, False]
        ).reset_index(drop=True)
        result[int(season)] = {
            row["teamName"]: pos for pos, row in zip(range(1, len(finals) + 1), finals.to_dict("records"))
        }
    return result


def load_value_bonuses(path=SQUAD_VALUE_FILE) -> dict[int, dict[str, float]]:
    """
    {season: {team: elo_bonus}} from squad_values.csv, where the bonus is
    VALUE_ELO_SCALE × the within-season z-score of log(market value). Returns {} if
    the file is absent, so the Elo pipeline still runs without squad data.
    """
    if not path.exists():
        return {}
    sv = pd.read_csv(path)
    out: dict[int, dict[str, float]] = {}
    for season, g in sv.groupby("season"):
        log_val = np.log(g["value_eur"].to_numpy())
        std = log_val.std()
        if len(g) < 2 or std == 0:
            out[int(season)] = {t: 0.0 for t in g["teamName"]}
            continue
        z = (log_val - log_val.mean()) / std
        out[int(season)] = {
            t: VALUE_ELO_SCALE * float(zi) for t, zi in zip(g["teamName"], z)
        }
    return out


def season_start_rating(
    team: str,
    prev_season: int,
    carried: float | None,
    prev_positions: dict[str, int],
    value_bonus: float = 0.0,
) -> float:
    """Rating a team begins a season on: mean-reverted carryover + prestige bonuses."""
    if carried is None:
        base = NEW_TEAM_ELO
    else:
        base = MEAN_ELO + SEASON_REGRESSION * (carried - MEAN_ELO)
    base += european_qualification_bonus(prev_positions.get(team))
    base += trophy_bonus(team, prev_season)
    base += value_bonus  # relative squad-value seed (0 before ~2005)
    return base


def compute_match_elo(
    matches_df: pd.DataFrame,
    standings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the chained Elo over every match in `matches_df` (all seasons) and return
    a long frame of (season, teamName, gameweek, elo), where `elo` is the team's
    rating *after* its Nth match of the season — aligning with the standings
    convention that a row is a team's state after N games.
    """
    matches = matches_df.copy()
    matches["Date"] = pd.to_datetime(matches["Date"], format="mixed", dayfirst=True, errors="coerce")
    matches = matches.sort_values(["season", "Date"]).reset_index(drop=True)

    final_positions = _final_positions_by_season(standings_df)
    value_bonuses = load_value_bonuses()

    ratings: dict[str, float] = {}   # persistent across seasons
    records = []

    for season, sdf in matches.groupby("season"):
        season = int(season)
        prev_positions = final_positions.get(season - 1, {})
        season_values = value_bonuses.get(season, {})

        # Season-start reseed for every team playing this season.
        teams = pd.unique(pd.concat([sdf["HomeTeam"], sdf["AwayTeam"]]).astype(str).str.strip())
        for team in teams:
            ratings[team] = season_start_rating(
                team, season - 1, ratings.get(team), prev_positions,
                season_values.get(team, 0.0),
            )

        games_played: dict[str, int] = {}
        for _, m in sdf.iterrows():
            home = str(m["HomeTeam"]).strip()
            away = str(m["AwayTeam"]).strip()
            hg, ag = int(m["FTHG"]), int(m["FTAG"])

            r_home, r_away = ratings[home], ratings[away]
            exp_home = expected_score(r_home + HOME_ADVANTAGE, r_away)

            if hg > ag:
                s_home = 1.0
            elif hg < ag:
                s_home = 0.0
            else:
                s_home = 0.5

            mult = K_FACTOR * goal_diff_multiplier(hg - ag)
            delta = mult * (s_home - exp_home)
            ratings[home] = r_home + delta
            ratings[away] = r_away - delta

            for team in (home, away):
                games_played[team] = games_played.get(team, 0) + 1
                records.append({
                    "season": season,
                    "teamName": team,
                    "gameweek": games_played[team],
                    "elo": ratings[team],
                })

    return pd.DataFrame(records)


def add_match_elo(standings_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a chained-Elo `elo` column onto a standings frame, keyed by
    (season, teamName, gameweek). Any unmatched rows fall back to NEW_TEAM_ELO.
    """
    elo_df = compute_match_elo(matches_df, standings_df)
    merged = standings_df.merge(elo_df, on=["season", "teamName", "gameweek"], how="left")
    merged["elo"] = merged["elo"].fillna(NEW_TEAM_ELO)
    return merged
