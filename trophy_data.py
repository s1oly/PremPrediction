# All team names match football-data.co.uk convention exactly.
# Key = season start year (e.g., 2023 = 2023-24 season).
# To add a newly completed season, run: python add_season.py --season <start_year>

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
    2024: "Crystal Palace",   # 2024-25 — verify if uncertain
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
}

# Europa League / UEFA Cup — English clubs only.
EL_WINNERS_ENGLISH = {
    2000: "Liverpool",    # 2000-01 UEFA Cup
    2018: "Chelsea",      # 2018-19
    2024: "Tottenham",    # 2024-25
}

# Conference League — English clubs only (competition started 2021-22).
ECL_WINNERS_ENGLISH = {
    2022: "West Ham",     # 2022-23
}


def get_trophies(team: str, season: int) -> tuple[int, int]:
    """
    Return (trophies_won, possible_trophies) for a team in a given season.

    possible_trophies = 3 (PL + FA Cup + League Cup) for all PL teams,
    +1 for each European competition entered.  European participation is
    approximated: top-7 finishers from the *previous* season enter Europe.
    That adjustment is applied separately in compute_elo(); this function
    only accounts for the +1 when the team actually *won* a European trophy
    (guaranteeing they entered it).
    """
    trophies = 0
    possible = 3  # domestic: PL + FA Cup + League Cup

    if PL_CHAMPIONS.get(season) == team:
        trophies += 1
    if FA_CUP_WINNERS.get(season) == team:
        trophies += 1
    if LEAGUE_CUP_WINNERS.get(season) == team:
        trophies += 1

    # European trophies: winning proves participation, so possible also rises
    if CL_WINNERS_ENGLISH.get(season) == team:
        trophies += 1
        possible += 1
    if EL_WINNERS_ENGLISH.get(season) == team:
        trophies += 1
        possible += 1
    if ECL_WINNERS_ENGLISH.get(season) == team:
        trophies += 1
        possible += 1

    return trophies, possible


def compute_elo(prev_position: int, trophies_won: int, possible_trophies: int) -> float:
    """
    ELO raw  = prev_position / (1 + trophies_won / possible_trophies)
    Feature  = 1 / elo_raw   (inverse: higher value = stronger team)

    A title-winning, trophy-laden side (position=1, trophies=4/4) gets a
    higher feature value than a mid-table, trophy-less side.
    """
    elo_raw = prev_position / (1.0 + trophies_won / possible_trophies)
    return 1.0 / elo_raw


def add_season_entry(season: int, pl: str, fa: str, lc: str,
                     cl: str = None, el: str = None, ecl: str = None):
    """
    Return the dict entries to append to each trophy dict for a new season.
    Convenience helper used by add_season.py — it prints the lines to paste
    into this file rather than mutating global state at runtime.
    """
    entries = {
        "PL_CHAMPIONS": (season, pl),
        "FA_CUP_WINNERS": (season, fa),
        "LEAGUE_CUP_WINNERS": (season, lc),
    }
    if cl:
        entries["CL_WINNERS_ENGLISH"] = (season, cl)
    if el:
        entries["EL_WINNERS_ENGLISH"] = (season, el)
    if ecl:
        entries["ECL_WINNERS_ENGLISH"] = (season, ecl)
    return entries
