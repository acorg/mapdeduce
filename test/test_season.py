#!/usr/bin/env python

"""Tests for code in MapDeduce/season.py"""

import unittest2 as unittest

from MapDeduce.season import in_season
from pandas import to_datetime

class MapSeqInSeason(unittest.TestCase):
    """Tests for MapDeduce.season.in_season"""

    def test_returns_fun_nh(self):
        fun = in_season("2005-2006")
        self.assertTrue(hasattr(fun, "__call__"))

    def test_returns_fun_sh(self):
        fun = in_season("2005")
        self.assertTrue(hasattr(fun, "__call__"))

    def test_nov_in_sh(self):
        """November is in SH season."""
        fun = in_season("2005")
        date = to_datetime('2005-11-04')
        self.assertTrue(fun(date))

    def test_dec_notin_sh(self):
        """December is not in SH season."""
        fun = in_season("2005")
        date = to_datetime('2005-12-04')
        self.assertFalse(fun(date))

    def test_nov_in_sh_but_yr_checked(self):
        """November is in SH season, but year has to be correct."""
        fun = in_season("2006")
        date = to_datetime('2005-11-04')
        self.assertFalse(fun(date))

    def test_apr_in_sh(self):
        """April is in SH season."""
        fun = in_season("2005")
        date = to_datetime('2005-04-04')
        self.assertTrue(fun(date))

    def test_mar_not_in_sh(self):
        """March is not in SH season."""
        fun = in_season("2005")
        date = to_datetime('2005-03-04')
        self.assertFalse(fun(date))

if __name__ == "__main__":
    unittest.main()
