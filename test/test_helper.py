#!/usr/bin/env python

"""Tests for helper functions."""

import numpy as np
import pandas as pd
from mapdeduce.helper import expand_sequences

try:
    import unittest2 as unittest
except ImportError:
    import unittest


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


if __name__ == "__main__":
    unittest.main()
