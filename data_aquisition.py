import io
import time
import requests
import pandas as pd
from constants import API_KEY


def season_to_fdcuk_code(start_year: int) -> str:
    """Convert season start year to football-data.co.uk URL code (e.g. 2023 → '2324')."""
    y1 = start_year % 100
    y2 = (start_year + 1) % 100
    return f"{y1:02d}{y2:02d}"


class FDCUKDataLoader:
    """
    Downloads and processes Premier League data from football-data.co.uk.
    Free, no API key required. Covers seasons 1993-94 onwards (start year 1993).

    Each row in the output represents a team's cumulative state after their
    Nth game of the season (gameweek = games played so far).
    """

    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    def download_season_matches(self, start_year: int) -> pd.DataFrame | None:
        """Download raw match results for a season. Returns DataFrame or None on failure."""
        code = season_to_fdcuk_code(start_year)
        url = f"{self.BASE_URL}/{code}/E0.csv"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                print(f"  Season {start_year}: missing columns {missing}, skipping.")
                return None
            df = df[needed].dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
            df = df[df["HomeTeam"].str.strip() != ""].reset_index(drop=True)
            return df
        except Exception as e:
            print(f"  Failed to download season {start_year}: {e}")
            return None

    def compute_standings_from_matches(self, matches_df: pd.DataFrame, start_year: int) -> pd.DataFrame:
        """
        Compute per-game cumulative standings from a season's match results.

        Returns one row per (team, game played), sorted chronologically.
        'gameweek' = number of games the team has completed (1–38).
        'position'  = live league rank by pts → GD → GF at that snapshot.
        """
        matches_df = matches_df.copy()
        matches_df["Date"] = pd.to_datetime(matches_df["Date"], dayfirst=True, errors="coerce")
        matches_df = matches_df.sort_values("Date").reset_index(drop=True)

        team_state: dict[str, dict] = {}
        rows = []

        for _, match in matches_df.iterrows():
            home = str(match["HomeTeam"]).strip()
            away = str(match["AwayTeam"]).strip()
            hg = int(match["FTHG"])
            ag = int(match["FTAG"])
            ftr = str(match["FTR"]).strip()

            for team in (home, away):
                if team not in team_state:
                    team_state[team] = {
                        "won": 0, "draw": 0, "lost": 0,
                        "gf": 0, "ga": 0, "points": 0,
                        "form": [],
                    }

            if ftr == "H":
                team_state[home]["won"] += 1
                team_state[away]["lost"] += 1
                team_state[home]["form"].append("W")
                team_state[away]["form"].append("L")
            elif ftr == "D":
                team_state[home]["draw"] += 1
                team_state[away]["draw"] += 1
                team_state[home]["form"].append("D")
                team_state[away]["form"].append("D")
            else:  # A
                team_state[home]["lost"] += 1
                team_state[away]["won"] += 1
                team_state[home]["form"].append("L")
                team_state[away]["form"].append("W")

            team_state[home]["gf"] += hg
            team_state[home]["ga"] += ag
            team_state[away]["gf"] += ag
            team_state[away]["ga"] += hg

            for t in (home, away):
                s = team_state[t]
                s["points"] = s["won"] * 3 + s["draw"]
                played = s["won"] + s["draw"] + s["lost"]
                rows.append({
                    "season": start_year,
                    "gameweek": played,
                    "teamName": t,
                    "won": s["won"],
                    "draw": s["draw"],
                    "lost": s["lost"],
                    "points": s["points"],
                    "goalsFor": s["gf"],
                    "goalsAgainst": s["ga"],
                    "goalDifference": s["gf"] - s["ga"],
                    "form": "".join(s["form"][-5:]),
                })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Rank teams within each (season, gameweek) snapshot
        def rank_snapshot(g):
            g = g.copy()
            g = g.sort_values(
                ["points", "goalDifference", "goalsFor"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
            g["position"] = range(1, len(g) + 1)
            return g

        df = df.groupby(["season", "gameweek"], group_keys=False).apply(rank_snapshot)
        return df.reset_index(drop=True)

    def get_all_seasons(self, start: int = 1993, end: int = 2024) -> pd.DataFrame:
        """Download and process all seasons from start_year to end_year (inclusive)."""
        all_dfs = []
        for year in range(start, end + 1):
            print(f"  Fetching {year}-{(year+1) % 100:02d} season...", end=" ", flush=True)
            matches = self.download_season_matches(year)
            if matches is not None and not matches.empty:
                standings = self.compute_standings_from_matches(matches, year)
                all_dfs.append(standings)
                print(f"{len(standings)} rows")
            else:
                print("skipped")
            time.sleep(0.4)  # polite rate-limit

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def get_season(self, start_year: int) -> pd.DataFrame | None:
        """Download and process a single season."""
        matches = self.download_season_matches(start_year)
        if matches is None or matches.empty:
            return None
        return self.compute_standings_from_matches(matches, start_year)

    def get_final_standings(self, start_year: int) -> pd.DataFrame:
        """
        Return a DataFrame with each team's final league position and stats
        for the given season (state after game 38 or the last game played).
        """
        season_df = self.get_season(start_year)
        if season_df is None or season_df.empty:
            return pd.DataFrame()
        last_gw_per_team = season_df.groupby("teamName")["gameweek"].max()
        finals = []
        for team, max_gw in last_gw_per_team.items():
            row = season_df[(season_df["teamName"] == team) & (season_df["gameweek"] == max_gw)]
            if not row.empty:
                finals.append(row.iloc[0])
        if not finals:
            return pd.DataFrame()
        final_df = pd.DataFrame(finals)
        final_df = final_df.sort_values(
            ["points", "goalDifference", "goalsFor"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        final_df["position"] = range(1, len(final_df) + 1)
        return final_df


# ---------------------------------------------------------------------------
# Legacy class kept for backwards compatibility and current-season API access
# ---------------------------------------------------------------------------

class FootballDataAPI:
    """football-data.org API wrapper (requires API key; free tier limited to 3 seasons)."""

    def __init__(self, api_key=None):
        self.api_key = api_key or API_KEY
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {"X-Auth-Token": self.api_key}

    def get_current_standings(self, season=None, gameweek=None):
        url = f"{self.base_url}/competitions/PL/standings"
        params = {}
        if season:
            params["season"] = season
        if gameweek:
            params["matchday"] = gameweek
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json().get("standings", [])
        return None

    def determine_season_winner(self, season=None):
        standings = self.get_current_standings(season, 38)
        if standings:
            for table in standings:
                for entry in table["table"]:
                    if entry["position"] == 1:
                        return entry["team"]["id"]
        return None
