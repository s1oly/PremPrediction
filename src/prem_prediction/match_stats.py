"""
Shot-based match-performance rating.

WHY THIS EXISTS
---------------
Every feature the model already has — points, form, goal difference, Elo — is
derived from *results*. None of them can tell a side that won 1-0 riding its
goalkeeper apart from one that won 1-0 having battered the opponent. Yet how a
team is *performing* — not just what it is collecting — is a leading indicator:
strong underlying numbers surface in the table a few weeks before the points do,
and a squad thinned by injury creates worse chances before its results dip.

This module distils each match into a single performance score from the shot
counts football-data.co.uk already provides (shots and shots on target, from the
2000-01 season on), then rolls it over each team's recent games. The output is
`shot_perf`: a rating of recent performance quality, deliberately decoupled from
the scoreline.

WHY A SEPARATE FEATURE, NOT FOLDED INTO ELO
-------------------------------------------
Elo is intentionally a pure *results* rating (see elo.py). Shot counts are not
calibrated to an expected match outcome the way xG is, so there is no principled
"expected score" to feed the Elo update — the LSTM is left to weight this signal
itself. (xG, if added later, WOULD have a natural home inside the Elo update, and
could sit alongside the results-Elo as a second rating; shots do not.)

Per-match performance is a weighted differential of shots on target and total
shots, from the home team's view (negated for the away side); shots on target is
the stronger signal so it carries most of the weight. Seasons before 2000-01 have
no shot data, so `shot_perf` is 0 there — neutral, exactly like the pre-2005
squad-value fallback, so the feature stays defined across all of history.
"""

import pandas as pd

# Per-match performance weighting and the rolling window, grouped here for tuning.
SOT_WEIGHT = 0.7    # weight on the shots-on-target differential (the sharper signal)
SHOT_WEIGHT = 0.3   # weight on the total-shots differential
ROLL_WINDOW = 5     # matches in the rolling "recent performance" window


def _num(v) -> float:
    """Coerce a cell to float, mapping missing/blank values (pre-2000 seasons) to 0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(f) else f


def _match_perf(sot_diff: float, shot_diff: float) -> float:
    """A single match's performance score for one team (already signed)."""
    return SOT_WEIGHT * sot_diff + SHOT_WEIGHT * shot_diff


def compute_shot_perf(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll a shot-based performance rating over every match and return a long frame
    of (season, teamName, gameweek, shot_perf), where the value at gameweek N is the
    mean per-match performance across the team's last ROLL_WINDOW games *including*
    its Nth — matching the standings/Elo convention that a row is state after N games.
    """
    matches = matches_df.copy()
    matches["Date"] = pd.to_datetime(
        matches["Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    matches = matches.sort_values(["season", "Date"]).reset_index(drop=True)

    records = []
    for season, sdf in matches.groupby("season"):
        season = int(season)
        history: dict[str, list[float]] = {}
        games_played: dict[str, int] = {}

        for _, m in sdf.iterrows():
            home = str(m["HomeTeam"]).strip()
            away = str(m["AwayTeam"]).strip()

            hst, ast = _num(m.get("HST")), _num(m.get("AST"))
            hs, as_ = _num(m.get("HS")), _num(m.get("AS"))
            home_perf = _match_perf(hst - ast, hs - as_)  # away is the negative

            for team, perf in ((home, home_perf), (away, -home_perf)):
                history.setdefault(team, []).append(perf)
                games_played[team] = games_played.get(team, 0) + 1
                window = history[team][-ROLL_WINDOW:]
                records.append({
                    "season": season,
                    "teamName": team,
                    "gameweek": games_played[team],
                    "shot_perf": sum(window) / len(window),
                })

    return pd.DataFrame(records)


def add_shot_perf(standings_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a `shot_perf` column onto a standings frame, keyed by
    (season, teamName, gameweek). Rows with no shot data fall back to 0 (neutral),
    so pre-2000 seasons and the odd missing fixture stay defined.
    """
    perf_df = compute_shot_perf(matches_df)
    merged = standings_df.merge(perf_df, on=["season", "teamName", "gameweek"], how="left")
    merged["shot_perf"] = merged["shot_perf"].fillna(0.0)
    return merged
