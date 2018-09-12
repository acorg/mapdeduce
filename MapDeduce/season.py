"""Code for classifying dates into influenza seasons and related processes."""

import pandas as pd

octToDec = 10, 11, 12
janToMay = 1, 2, 3, 4, 5
aprToNov = 4, 5, 6, 7, 8, 9, 10, 11


def in_season(season):
    """Make function to test if a date is in season.

    Args:
        season (str or int): E.g. '2006-2007' or '2005'. Northern hemisphere
            seasons straddle two years, and are specified like '2005-2006'. 
            Southern hemisphere seasons are fully contained within one year and
            are specified like '2005' or as ints: 2005.

    Returns:
        function
    """
    if "-" in season:
        # Northern hemisphere season
        yr1, yr2 = map(int, season.split('-'))

        def fun(date):
            """Check if a date is in a northern hemisphere season.

            Args:
                date (pd.Timestamp)

            Returns:
                Bool
            """
            if date.year == yr1 and date.month in octToDec:
                return True
            elif date.year == yr2 and date.month in janToMay:
                return True
            else:
                return False

    elif len(str(season)) == 4:
        yr = int(season)

        def fun(date):
            """Check if a date is in the southern hemisphere season.

            Args:
                date (pd.Timestamp)

            Returns:
                Bool
            """
            if date.year == yr and date.month in aprToNov:
                return True
            else:
                return False

    else:
        raise ValueError("'{}' doesn't look like a flu season.".format(season))

    return fun


def season_from_timestamp(ts):
    """Convert timestamp to a season

    Args:
        ts (pd.Timestamp)

    Returns:
        str. Like 2006-2007
    """
    if ts is None:
        return None

    elif ts.month in octToDec:
        return "{}-{}".format(ts.year, ts.year + 1)

    elif ts.month in janToMay:
        return "{}-{}".format(ts.year - 1, ts.year)

    else:
        return "Not in main season"


def date_str_to_timestamp(date):
    """Make date field in fasta header into a pd.timestamp.

    Args:
        date (str): Date from fasta header

    Returns:
        pd.timestamp
    """
    if '(Month and day unknown)' in date:
        # Can't know season
        return None

    elif '(Day unknown)' in date:
        return pd.to_datetime(date[:7], format='%Y-%m')

    else:
        return pd.to_datetime(date)
