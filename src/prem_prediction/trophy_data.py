# Historical trophy winners, keyed by SEASON START YEAR (e.g. 2023 = 2023-24).
# All team names match the football-data.co.uk convention exactly
# ("Man City", "Man United", "Nott'm Forest", ...).
#
# These feed the Elo prestige seeding in elo.py: winning a trophy adds an
# additive Elo bonus at the start of the following season. Unlike the old
# ratio-based formula, extra competitions can only ever *help* a team.
#
# To add a newly completed season, run: python -m prem_prediction.add_season --season <start_year>

PL_CHAMPIONS = {
    1992: None,               # Inaugural season — no prior PL year
    1993: "Man United",       # 1993-94
    1994: "Blackburn",        # 1994-95
    1995: "Man United",       # 1995-96
    1996: "Man United",       # 1996-97
    1997: "Arsenal",          # 1997-98
    1998: "Man United",       # 1998-99
    1999: "Man United",       # 1999-2000
    2000: "Man United",       # 2000-01
    2001: "Arsenal",          # 2001-02
    2002: "Man United",       # 2002-03
    2003: "Arsenal",          # 2003-04 (Invincibles)
    2004: "Chelsea",          # 2004-05
    2005: "Chelsea",          # 2005-06
    2006: "Man United",       # 2006-07
    2007: "Man United",       # 2007-08
    2008: "Man United",       # 2008-09
    2009: "Chelsea",          # 2009-10
    2010: "Man United",       # 2010-11
    2011: "Man City",         # 2011-12
    2012: "Man United",       # 2012-13
    2013: "Man City",         # 2013-14
    2014: "Chelsea",          # 2014-15
    2015: "Leicester",        # 2015-16
    2016: "Chelsea",          # 2016-17
    2017: "Man City",         # 2017-18
    2018: "Man City",         # 2018-19
    2019: "Liverpool",        # 2019-20
    2020: "Man City",         # 2020-21
    2021: "Man City",         # 2021-22
    2022: "Man City",         # 2022-23
    2023: "Man City",         # 2023-24
    2024: "Liverpool",        # 2024-25
    2025: "Arsenal",          # 2025-26
}

FA_CUP_WINNERS = {
    1992: "Arsenal",          # 1992-93
    1993: "Man United",       # 1993-94
    1994: "Everton",          # 1994-95
    1995: "Man United",       # 1995-96
    1996: "Chelsea",          # 1996-97
    1997: "Arsenal",          # 1997-98
    1998: "Man United",       # 1998-99
    1999: "Chelsea",          # 1999-2000
    2000: "Liverpool",        # 2000-01
    2001: "Arsenal",          # 2001-02
    2002: "Arsenal",          # 2002-03
    2003: "Man United",       # 2003-04
    2004: "Arsenal",          # 2004-05
    2005: "Liverpool",        # 2005-06
    2006: "Chelsea",          # 2006-07
    2007: "Portsmouth",       # 2007-08
    2008: "Chelsea",          # 2008-09
    2009: "Chelsea",          # 2009-10
    2010: "Man City",         # 2010-11
    2011: "Chelsea",          # 2011-12
    2012: "Wigan",            # 2012-13
    2013: "Arsenal",          # 2013-14
    2014: "Arsenal",          # 2014-15
    2015: "Man United",       # 2015-16
    2016: "Arsenal",          # 2016-17
    2017: "Chelsea",          # 2017-18
    2018: "Man City",         # 2018-19
    2019: "Arsenal",          # 2019-20
    2020: "Leicester",        # 2020-21
    2021: "Liverpool",        # 2021-22
    2022: "Man City",         # 2022-23
    2023: "Man United",       # 2023-24
    2024: "Crystal Palace",   # 2024-25
    2025: "Man City",         # 2025-26 (beat Chelsea 1-0)
}

LEAGUE_CUP_WINNERS = {
    1992: "Arsenal",          # 1992-93
    1993: "Aston Villa",      # 1993-94
    1994: "Liverpool",        # 1994-95
    1995: "Aston Villa",      # 1995-96
    1996: "Leicester",        # 1996-97
    1997: "Chelsea",          # 1997-98
    1998: "Tottenham",        # 1998-99
    1999: "Leicester",        # 1999-2000
    2000: "Liverpool",        # 2000-01
    2001: "Blackburn",        # 2001-02
    2002: "Liverpool",        # 2002-03
    2003: "Middlesbrough",    # 2003-04
    2004: "Chelsea",          # 2004-05
    2005: "Man United",       # 2005-06
    2006: "Chelsea",          # 2006-07
    2007: "Tottenham",        # 2007-08
    2008: "Man United",       # 2008-09
    2009: "Man United",       # 2009-10
    2010: "Birmingham",       # 2010-11
    2011: "Liverpool",        # 2011-12
    2012: "Swansea",          # 2012-13
    2013: "Man City",         # 2013-14
    2014: "Chelsea",          # 2014-15
    2015: "Man City",         # 2015-16
    2016: "Man United",       # 2016-17
    2017: "Man City",         # 2017-18
    2018: "Man City",         # 2018-19
    2019: "Man City",         # 2019-20
    2020: "Man City",         # 2020-21
    2021: "Liverpool",        # 2021-22
    2022: "Man United",       # 2022-23
    2023: "Liverpool",        # 2023-24
    2024: "Newcastle",        # 2024-25
    2025: "Man City",         # 2025-26 (beat Arsenal 2-0)
}

# English clubs only. Key = season start year the competition was played in.
CL_WINNERS_ENGLISH = {
    1998: "Man United",   # 1998-99
    2004: "Liverpool",    # 2004-05
    2007: "Man United",   # 2007-08
    2011: "Chelsea",      # 2011-12
    2018: "Liverpool",    # 2018-19
    2020: "Chelsea",      # 2020-21
    2022: "Man City",     # 2022-23
    # 2025-26: PSG beat Arsenal in the final — no English winner.
}

# Europa League / UEFA Cup — English clubs only.
EL_WINNERS_ENGLISH = {
    2000: "Liverpool",    # 2000-01 UEFA Cup
    2018: "Chelsea",      # 2018-19
    2024: "Tottenham",    # 2024-25
    2025: "Aston Villa",  # 2025-26 (beat Freiburg 3-0)
}

# Conference League — English clubs only (competition started 2021-22).
ECL_WINNERS_ENGLISH = {
    2022: "West Ham",       # 2022-23
    2025: "Crystal Palace", # 2025-26 (beat Rayo Vallecano 1-0)
}


# Elo prestige bonus (points) awarded for winning each trophy last season.
# Additive and stacking — see elo.py.
TROPHY_BONUS = {
    "PL": 50.0,
    "CL": 50.0,
    "EL": 30.0,
    "FA": 20.0,
    "LC": 15.0,
    "ECL": 15.0,
}

_TROPHY_DICTS = {
    "PL": PL_CHAMPIONS,
    "FA": FA_CUP_WINNERS,
    "LC": LEAGUE_CUP_WINNERS,
    "CL": CL_WINNERS_ENGLISH,
    "EL": EL_WINNERS_ENGLISH,
    "ECL": ECL_WINNERS_ENGLISH,
}


def get_trophy_codes(team: str, season: int) -> list[str]:
    """Return the codes of every trophy `team` won in `season` (e.g. ['PL', 'FA'])."""
    return [code for code, table in _TROPHY_DICTS.items() if table.get(season) == team]


def trophy_bonus(team: str, season: int) -> float:
    """Total additive Elo prestige bonus for the trophies `team` won in `season`."""
    return sum(TROPHY_BONUS[c] for c in get_trophy_codes(team, season))
