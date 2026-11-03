"""
Strength-of-schedule features: how hard a team's fixtures have been, and how
hard the ones still to come are.

WHY THIS EXISTS
---------------
Points, form and goal difference all say *how much* a team has collected, but
nothing about *who it was collected against*. Two sides on 21 points are not
equivalent if one ground them out against the top six while the other feasted on
the bottom three. The market prices this in — a lead built against a soft run is
trusted less than the same lead built against a brutal one — but the model had no
feature carrying it. These two do:

    sos_played    — average strength of the opponents faced through gameweek N
    sos_remaining — average strength of the opponents still to be played

Together they let the LSTM read "tough run so far" versus "tough run ahead",
which is exactly the schedule-vs-points context a title race turns on late on.

HOW STRENGTH IS MEASURED (and why it does not leak)
---------------------------------------------------
Opponent strength is the opponent's results-based **Elo** (elo.py), snapshotted at
the SAME gameweek N the feature is computed for. So at gameweek N every opponent —
whether already played or still to come — is rated by what it has shown through
gameweek N, and nothing after. No final-table hindsight enters, and the feature is
fully defined for the in-progress season, unlike anything keyed on final position.

Both values are centred on the league mean (MEAN_ELO), so 0 is an exactly average
schedule, positive is harder than average, negative is easier. A team at the end of
the season has no opponents left, so `sos_remaining` falls back to 0 (neutral).

REMAINING OPPONENTS WITHOUT A FIXTURE LIST
------------------------------------------
football-data.co.uk only publishes matches once played, so there is no forward
fixture list to read. We do not need one: the Premier League is a double
round-robin — every side plays every other home and away — so a team's full
opponent multiset is simply every other club that season, twice. Subtract the
opponents already faced and what remains is exactly the fixtures still to come.
"""

from collections import Counter

import pandas as pd

from .elo import MEAN_ELO, compute_match_elo


def _opponents_in_order(season_matches: pd.DataFrame) -> dict[str, list[str]]:
    """{team: [opponent per match, in chronological order]} for one season."""
    opps: dict[str, list[str]] = {}
    for _, m in season_matches.iterrows():
        home = str(m["HomeTeam"]).strip()
        away = str(m["AwayTeam"]).strip()
        opps.setdefault(home, []).append(away)
        opps.setdefault(away, []).append(home)
    return opps


def compute_schedule_strength(
    matches_df: pd.DataFrame,
    standings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Long frame (season, teamName, gameweek, sos_played, sos_remaining) where each
    value at gameweek N uses every team's Elo as of gameweek N, centred on MEAN_ELO.
    """
    matches = matches_df.copy()
    matches["Date"] = pd.to_datetime(
        matches["Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    matches = matches.sort_values(["season", "Date"]).reset_index(drop=True)

    elo_df = compute_match_elo(matches_df, standings_df)

    records = []
    for season, sdf in matches.groupby("season"):
        season = int(season)
        opps = _opponents_in_order(sdf)
        teams = list(opps.keys())

        # Elo of every team at each gameweek, forward-filled so an opponent that
        # has played fewer games still carries its latest known rating.
        e = elo_df[elo_df["season"] == season]
        pivot = (
            e.pivot_table(index="teamName", columns="gameweek", values="elo", aggfunc="last")
            .sort_index(axis=1)
            .ffill(axis=1)
        )
        gameweeks = list(pivot.columns)
        max_gw = int(gameweeks[-1]) if gameweeks else 0

        def snapshot(gw: int) -> dict[str, float]:
            """{team: Elo as of gameweek gw} (latest known rating at or before gw)."""
            col = min(gw, max_gw)
            snap = {}
            for t in teams:
                v = pivot.loc[t, col] if (t in pivot.index and col in pivot.columns) else None
                if v is None or pd.isna(v):
                    row = pivot.loc[t].dropna() if t in pivot.index else pd.Series(dtype=float)
                    v = float(row.iloc[-1]) if len(row) else MEAN_ELO
                snap[t] = float(v)
            return snap

        # Each team plays every other team twice (double round-robin).
        full_schedule = Counter({t: 2 for t in teams})

        snaps = {n: snapshot(n) for n in range(1, max_gw + 1)}
        for team, played_order in opps.items():
            full = full_schedule.copy()
            del full[team]
            total = len(played_order)
            for n in range(1, total + 1):
                snap = snaps.get(n, snapshot(n))
                played = played_order[:n]
                sos_played = sum(snap[o] for o in played) / len(played)

                remaining = full - Counter(played)
                rem = list(remaining.elements())
                sos_remaining = (sum(snap[o] for o in rem) / len(rem)) if rem else MEAN_ELO

                records.append({
                    "season": season,
                    "teamName": team,
                    "gameweek": n,
                    "sos_played": sos_played - MEAN_ELO,
                    "sos_remaining": sos_remaining - MEAN_ELO,
                })

    return pd.DataFrame(records)


def add_schedule_strength(standings_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge `sos_played` and `sos_remaining` onto a standings frame, keyed by
    (season, teamName, gameweek). Unmatched rows fall back to 0.0 (neutral).
    """
    sos_df = compute_schedule_strength(matches_df, standings_df)
    merged = standings_df.merge(sos_df, on=["season", "teamName", "gameweek"], how="left")
    merged["sos_played"] = merged["sos_played"].fillna(0.0)
    merged["sos_remaining"] = merged["sos_remaining"].fillna(0.0)
    return merged
