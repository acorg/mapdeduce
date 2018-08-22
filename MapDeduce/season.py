"""Code for classifying dates into influenza seasons and related processes."""

import pandas as pd

octToDec = 10, 11, 12
janToMay = 1, 2, 3, 4, 5


def in_season(season):
    """Make function to test if a date is in season.
    
    Args:
        season (str): E.g. '2006-2007'
    
    Returns:
        function
    """
    yr1, yr2 = map(int, season.split('-'))
    
    def fun(date):
        """Check if a date is in a season.

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
