import pandas as pd
import numpy as np
from trophy_data import get_trophies, compute_elo


# Default ELO for the inaugural 1993-94 season (no prior PL year to reference)
_DEFAULT_ELO = compute_elo(10, 0, 3)  # equivalent to mid-table, no trophies


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


def _get_prev_final_position(df: pd.DataFrame, team: str, prev_season: int) -> int:
    """
    Look up a team's final league position in prev_season from the standings DataFrame.
    Returns 20 (relegated/promoted default) when not found.
    """
    prev_df = df[df["season"] == prev_season]
    if prev_df.empty:
        return 20

    last_gw_per_team = prev_df.groupby("teamName")["gameweek"].max()
    if team not in last_gw_per_team:
        return 20  # not in PL that season (promoted team)

    max_gw = last_gw_per_team[team]
    rows = prev_df[(prev_df["teamName"] == team) & (prev_df["gameweek"] == max_gw)]
    if rows.empty:
        return 20
    return int(rows.iloc[0]["position"])


def _get_prev_final_standings(df: pd.DataFrame, prev_season: int) -> dict[str, int]:
    """Return {teamName: final_position} for all teams in prev_season."""
    prev_df = df[df["season"] == prev_season]
    if prev_df.empty:
        return {}

    final_rows = (
        prev_df.sort_values("gameweek")
        .groupby("teamName")
        .last()
        .reset_index()
    )
    final_rows = final_rows.sort_values(
        ["points", "goalDifference", "goalsFor"], ascending=[False, False, False]
    ).reset_index(drop=True)
    final_rows["final_position"] = range(1, len(final_rows) + 1)
    return dict(zip(final_rows["teamName"], final_rows["final_position"]))


def add_elo_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a dynamic ELO feature for every row.

    base_elo  = 1 / (prev_position / (1 + trophies / possible))
                Encodes prior-season quality (constant within a season per team).

    dynamic   = base_elo * (100 / current_position)
                Scales the base each gameweek by the team's live table position,
                so ELO rises as a team climbs and falls as they drop.

    Higher ELO always means a stronger team.
    """
    df = df.copy()
    df["elo"] = _DEFAULT_ELO  # safe fallback

    for season in df["season"].unique():
        prev_season = season - 1
        prev_standings = _get_prev_final_standings(df, prev_season)

        for team in df[df["season"] == season]["teamName"].unique():
            prev_pos = prev_standings.get(team, 20)

            possible = 3
            if prev_pos <= 7:
                possible = 4
            t_won, t_possible = get_trophies(team, prev_season)
            possible = max(possible, t_possible)

            base_elo = compute_elo(prev_pos, t_won, possible)

            # Apply dynamic scaling using live position each gameweek
            mask = (df["season"] == season) & (df["teamName"] == team)
            df.loc[mask, "elo"] = base_elo * (100 / df.loc[mask, "position"])

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add won_league (1/0) and games_remaining columns.

    won_league = 1 for every row of the team that had the most points at
    the end of each season. All other rows in that season get 0.
    """
    df = df.copy()
    df["games_remaining"] = 38 - df["gameweek"]
    df["won_league"] = 0

    for season in df["season"].unique():
        season_df = df[df["season"] == season]
        # Find winner: team with best record at their final gameweek
        final_rows = (
            season_df.sort_values("gameweek")
            .groupby("teamName")
            .last()
            .reset_index()
        )
        if final_rows.empty:
            continue
        winner_row = final_rows.sort_values(
            ["points", "goalDifference", "goalsFor"], ascending=[False, False, False]
        ).iloc[0]
        winner = winner_row["teamName"]
        df.loc[(df["season"] == season) & (df["teamName"] == winner), "won_league"] = 1

    return df


def convert_form(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the string form column to a numerical score in-place."""
    df = df.copy()
    df["form"] = df["form"].apply(numerical_form)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: form → labels → ELO."""
    df = convert_form(df)
    df = add_labels(df)
    df = add_elo_feature(df)
    return df


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv("historical_standings.csv")
    out = prepare(df)
    out.to_csv("out.csv", index=False)
    print(f"Saved {len(out)} rows to out.csv")
    print(out[["season", "gameweek", "teamName", "position", "elo", "won_league"]].head(20).to_string())


if __name__ == "__main__":
    main()
