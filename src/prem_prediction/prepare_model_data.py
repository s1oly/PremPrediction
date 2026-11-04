import numpy as np
import pandas as pd

from .elo import add_match_elo, _final_positions_by_season
from .match_stats import add_shot_features
from .schedule_strength import add_schedule_strength
from .trophy_data import trophy_bonus
from .paths import MATCHES_FILE, SQUAD_VALUE_FILE, STANDINGS_FILE, TRAINING_FILE

# Value used when a team was not in the PL the previous season (promoted / new).
# One rung below last place, so "promoted" reads as weaker than "finished 20th".
PROMOTED_PREV_POSITION = 21


def numerical_form(form_string) -> float:
    """Convert a form string (e.g. 'WWDL') to a numerical score (W=3, D=1, L=0)."""
    if pd.isna(form_string) or str(form_string).strip() == "":
        return 0.0
    total = 0.0
    for ch in str(form_string).upper():
        if ch == "W":
            total += 3
        elif ch == "D":
            total += 1
    return total


def convert_form(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the string form column to a numerical score."""
    df = df.copy()
    df["form"] = df["form"].apply(numerical_form)
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add games_remaining, won_league (1/0), and final_points columns.

    final_points — the regression target: the team's total points at the end of
      its season. Every row of a (season, team) carries that team's final total, so
      the model learns to predict the final table from a partial-season sequence.
      Title probability is then derived as P(a team's final points are the league's
      highest) by simulation at prediction time (see season_predict).
    won_league — kept for evaluation/reference: 1 for the season's top team, else 0.
    """
    df = df.copy()
    df["games_remaining"] = 38 - df["gameweek"]
    df["won_league"] = 0
    df["final_points"] = np.nan

    for season in df["season"].unique():
        season_df = df[df["season"] == season]
        final_rows = (
            season_df.sort_values("gameweek").groupby("teamName").last().reset_index()
        )
        if final_rows.empty:
            continue
        # Final points per team this season -> broadcast onto all that team's rows.
        final_pts = dict(zip(final_rows["teamName"], final_rows["points"]))
        mask = df["season"] == season
        df.loc[mask, "final_points"] = df.loc[mask, "teamName"].map(final_pts)

        winner = final_rows.sort_values(
            ["points", "goalDifference", "goalsFor"], ascending=[False, False, False]
        ).iloc[0]["teamName"]
        df.loc[mask & (df["teamName"] == winner), "won_league"] = 1

    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three gameweek-relative features encoding each team's standing versus the
    rest of the table at the same snapshot.

    points_per_game        — scoring pace; comparable across gameweeks
    points_gap_from_leader — 0 for the leader, negative for everyone else
    max_obtainable_points  — points + games_remaining * 3; the title ceiling
    """
    df = df.copy()
    df["points_per_game"] = df["points"] / df["gameweek"].clip(lower=1)
    leader_pts = df.groupby(["season", "gameweek"])["points"].transform("max")
    df["points_gap_from_leader"] = df["points"] - leader_pts  # <= 0
    df["max_obtainable_points"] = df["points"] + df["games_remaining"] * 3
    return df


def add_prev_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `prev_position`: the team's final league position the previous season — a
    clean pre-season prior for EVERY team, not just champions or Europe qualifiers.
    Promoted / previously-absent teams get PROMOTED_PREV_POSITION.

    It is constant within a season, so at gameweek 1 it is the model's main
    differentiator (a side that finished 1st looks very different from one that
    finished 17th before a ball is even kicked).
    """
    df = df.copy()
    finals = _final_positions_by_season(df)  # {season: {team: final_pos}}
    prev = {
        (season + 1, team): pos
        for season, table in finals.items()
        for team, pos in table.items()
    }
    df["prev_position"] = [
        prev.get((s, t), PROMOTED_PREV_POSITION)
        for s, t in zip(df["season"], df["teamName"])
    ]
    return df


def add_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the pre-season prestige priors as DIRECT model features, so the LSTM learns
    their weight instead of them being hardcoded into the Elo seed (Elo is now pure
    results — see elo.py). Constant within a season, like prev_position:

      squad_value_z — within-season z-score of log(squad market value); relative, so
        immune to transfer-fee inflation. 0.0 when squad data is absent (pre-~2005).
      trophy_score  — stacked prestige of the trophies the team won the PREVIOUS
        season (PL/CL/EL/FA/LC/ECL), in the same point units trophy_data defines; 0
        for a team that won nothing.
    """
    df = df.copy()

    # squad_value_z from squad_values.csv (season, teamName, value_eur).
    value_z: dict[tuple[int, str], float] = {}
    if SQUAD_VALUE_FILE.exists():
        sv = pd.read_csv(SQUAD_VALUE_FILE)
        for season, g in sv.groupby("season"):
            log_val = np.log(g["value_eur"].to_numpy())
            std = log_val.std()
            if len(g) < 2 or std == 0:
                for t in g["teamName"]:
                    value_z[(int(season), t)] = 0.0
            else:
                z = (log_val - log_val.mean()) / std
                for t, zi in zip(g["teamName"], z):
                    value_z[(int(season), t)] = float(zi)
    df["squad_value_z"] = [value_z.get((s, t), 0.0) for s, t in zip(df["season"], df["teamName"])]

    # trophy_score: prestige from the PREVIOUS season's trophies.
    df["trophy_score"] = [trophy_bonus(t, int(s) - 1) for s, t in zip(df["season"], df["teamName"])]
    return df


def prepare(standings_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing: form -> labels -> relative -> priors -> Elo -> shots -> schedule strength."""
    df = convert_form(standings_df)
    df = add_labels(df)
    df = add_relative_features(df)
    df = add_prev_position(df)
    df = add_prior_features(df)
    df = add_match_elo(df, matches_df)
    df = add_shot_features(df, matches_df)
    df = add_schedule_strength(df, matches_df)
    return df


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

def main():
    standings = pd.read_csv(STANDINGS_FILE)
    matches = pd.read_csv(MATCHES_FILE)
    out = prepare(standings, matches)
    out.to_csv(TRAINING_FILE, index=False)
    print(f"Saved {len(out)} rows to {TRAINING_FILE}")
    print(
        out[["season", "gameweek", "teamName", "position", "elo", "won_league"]]
        .head(20)
        .to_string()
    )


if __name__ == "__main__":
    main()
