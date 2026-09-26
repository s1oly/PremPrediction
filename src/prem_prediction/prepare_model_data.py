import pandas as pd

from .elo import add_match_elo
from .paths import MATCHES_FILE, STANDINGS_FILE, TRAINING_FILE


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
    Add won_league (1/0) and games_remaining columns.

    won_league = 1 for every row of the team that finished top of the table at
    the end of each season; all other rows in that season get 0.
    """
    df = df.copy()
    df["games_remaining"] = 38 - df["gameweek"]
    df["won_league"] = 0

    for season in df["season"].unique():
        season_df = df[df["season"] == season]
        final_rows = (
            season_df.sort_values("gameweek").groupby("teamName").last().reset_index()
        )
        if final_rows.empty:
            continue
        winner = final_rows.sort_values(
            ["points", "goalDifference", "goalsFor"], ascending=[False, False, False]
        ).iloc[0]["teamName"]
        df.loc[(df["season"] == season) & (df["teamName"] == winner), "won_league"] = 1

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


def prepare(standings_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing: form -> labels -> relative features -> chained match Elo."""
    df = convert_form(standings_df)
    df = add_labels(df)
    df = add_relative_features(df)
    df = add_match_elo(df, matches_df)
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
