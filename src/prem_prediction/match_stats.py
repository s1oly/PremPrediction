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


def _shot_quality(sot: float, shots: float) -> float:
    """A single match's chance-quality score for one side (shots on target weighted up)."""
    return SOT_WEIGHT * sot + SHOT_WEIGHT * shots


def compute_shot_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll two shot-based ratings over every match and return a long frame of
    (season, teamName, gameweek, shot_att, shot_def):

      shot_att — chance quality a team CREATES  (its own SOT/shots), rolled over
                 the last ROLL_WINDOW games: a measure of attacking output.
      shot_def — chance quality a team CONCEDES (opponents' SOT/shots), same window:
                 a measure of defensive solidity (lower = a meaner defence).

    Previously these were fused into one net differential (`shot_perf`), which the
    model could not weight separately — so a mean defence and a potent attack looked
    identical. Splitting them lets the LSTM learn each side's predictive value on its
    own (chances conceded often predicts titles more than chances created). The value
    at gameweek N spans the team's last ROLL_WINDOW games including its Nth, matching
    the standings/Elo "state after N games" convention.
    """
    matches = matches_df.copy()
    matches["Date"] = pd.to_datetime(
        matches["Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    matches = matches.sort_values(["season", "Date"]).reset_index(drop=True)

    records = []
    for season, sdf in matches.groupby("season"):
        season = int(season)
        att_hist: dict[str, list[float]] = {}
        def_hist: dict[str, list[float]] = {}
        games_played: dict[str, int] = {}

        for _, m in sdf.iterrows():
            home = str(m["HomeTeam"]).strip()
            away = str(m["AwayTeam"]).strip()

            hst, ast = _num(m.get("HST")), _num(m.get("AST"))
            hs, as_ = _num(m.get("HS")), _num(m.get("AS"))
            home_created = _shot_quality(hst, hs)   # = away conceded
            away_created = _shot_quality(ast, as_)  # = home conceded

            for team, created, conceded in (
                (home, home_created, away_created),
                (away, away_created, home_created),
            ):
                att_hist.setdefault(team, []).append(created)
                def_hist.setdefault(team, []).append(conceded)
                games_played[team] = games_played.get(team, 0) + 1
                aw = att_hist[team][-ROLL_WINDOW:]
                dw = def_hist[team][-ROLL_WINDOW:]
                records.append({
                    "season": season,
                    "teamName": team,
                    "gameweek": games_played[team],
                    "shot_att": sum(aw) / len(aw),
                    "shot_def": sum(dw) / len(dw),
                })

    return pd.DataFrame(records)


def add_shot_features(standings_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge `shot_att` and `shot_def` columns onto a standings frame, keyed by
    (season, teamName, gameweek). Rows with no shot data fall back to 0 (neutral),
    so pre-2000 seasons and the odd missing fixture stay defined.
    """
    feat_df = compute_shot_features(matches_df)
    merged = standings_df.merge(feat_df, on=["season", "teamName", "gameweek"], how="left")
    merged["shot_att"] = merged["shot_att"].fillna(0.0)
    merged["shot_def"] = merged["shot_def"].fillna(0.0)
    return merged
