#!/usr/bin/env python

"""Tests for helper functions."""

import unittest

import numpy as np
import pandas as pd

from mapdeduce.helper import expand_sequences


class ExpandSequences(unittest.TestCase):
    """Tests for mapdeduce.helper.expand_sequences"""

    def test_returns_df(self):
        series = pd.Series(["abc", "def"])
        df = expand_sequences(series)
        self.assertIsInstance(df, pd.DataFrame)

    def test_handles_nan(self):
        series = pd.Series(["abc", np.nan, "def"])
        expand_sequences(series)

    def test_columns_are_integers(self):
        series = pd.Series(["abc", np.nan, "def"])
        df = expand_sequences(series)
        self.assertEqual(1, df.columns[0])

    def test_preserves_row_count(self):
        """expand_sequences should return the same number of rows as the
        input series length when all elements are valid equal-length
        strings."""
        series = pd.Series(["abc", "def", "ghi"])
        df = expand_sequences(series)
        self.assertEqual(len(df), len(series))

    def test_preserves_row_count_with_duplicate_indexes(self):
        """Row count should be preserved even with repeated indexes."""
        series = pd.Series(
            ["abc", "def", "ghi", "jkl"],
            index=["s1", "s1", "s2", "s2"],
        )
        df = expand_sequences(series)
        self.assertEqual(len(df), len(series))


if __name__ == "__main__":
    unittest.main()
