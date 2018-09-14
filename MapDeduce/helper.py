"""Helper functions."""

from __future__ import absolute_import
from builtins import range

import pandas as pd
from .data import amino_acids


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
    df.columns = list(range(df.shape[1]))
    df.columns += 1
    return df[df.notnull().all(axis=1)]

def check_all_not_null(df):
    """Check that each row in df does not contain only null entries.

    Args:
        df (pd.DataFrame)

    Returns:
        If any rows are not null, returns pd.Index
    """
    mask = df.isnull().all(axis=1)
    if mask.sum():
        return mask[mask].index

def site_consensus(series):
    """Return the consensus amino acid of a series.

    Notes:
        - Should return the amino acid with the highest value count in series.
        - When two amino acids tie with the highest value count, return "X".
        - "X", NaN and "-" should not contribute to value counts.

    Args:
        series (pd.Series)

    Returns:
        str
    """
    series = series.mask(series == "X").mask(series == "-").mask(pd.isnull)
    vc = series.value_counts()
    if vc.empty:  # There are only "X", "-" or null
        return "X"
    else:
        all_highest = vc[vc == vc.max()]  # All sites with count == max_count
        if all_highest.shape[0] == 1:
            return all_highest.index[0]
        else:
            return "X"
