"""Helper functions."""

import pandas as pd
from data import amino_acids


def is_not_amino_acid(a):
    """Test if a is not an amino acid.

    Args:
        a (str): String to test.

    Returns:
        bool. True if a is not an amino acid.
    """
    try:
        a = a.upper()
    except AttributeError:
        return True

    if a not in amino_acids:
        return True
    else:
        return False

def string_to_series(arg):
    """Each element in string becomes element in series

    Notes:
        If arg is not cannot be coerced to a series, return an empty series.

    Args:
        arg (str)

    Returns:
        pd.Series
    """
    try:
        return pd.Series(tuple(arg))
    except TypeError:
        return pd.Series()

def expand_sequences(series):
    """Expand Series containing sequences into DataFrame.

    Notes:
        Any elements in series that cannot be expanded will be dropped.

    Args:
        series (pd.Series)

    Returns:
        pd.DataFrame: Columns are sequence positions, indexes match
            series.index.
    """
    df = series.apply(string_to_series)
    df.columns = range(df.shape[1])
    df.columns += 1
    return df[df.notnull().all(axis=1)]
