"""
Runtime configuration: current-season detection and the (optional) legacy
football-data.org API key, read from the environment rather than a tracked file.
"""

import datetime
import os

# ---------------------------------------------------------------------------
# Legacy football-data.org API key (only needed by FootballDataAPI).
# The main pipeline uses football-data.co.uk, which needs no key.
# Set it via the environment or a .env file (see .env.example); never commit it.
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")


def current_season_start_year(today: datetime.date | None = None) -> int:
    """
    Return the start year of the Premier League season in progress.

    A PL season spans August -> May, so anything from August onwards belongs to
    the season that starts that calendar year; earlier months belong to the one
    that started the previous year.

    e.g. 2026-09-26 -> 2026 (the 2026-27 season).
    """
    today = today or datetime.date.today()
    return today.year if today.month >= 8 else today.year - 1


def latest_completed_season(today: datetime.date | None = None) -> int:
    """Start year of the most recently completed season (one before the current)."""
    return current_season_start_year(today) - 1


def season_label(start_year: int) -> str:
    """Format a season start year as e.g. '2026-27'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"
