#!/usr/bin/env python

"""Tests for code in MapDeduce/season.py"""

import unittest2 as unittest

from MapDeduce.season import in_season, season_from_timestamp, \
    hemisphere_from_season
from pandas import to_datetime

class InSeasonTests(unittest.TestCase):
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

class SeasonFromTimestampTests(unittest.TestCase):
    """Tests for MapDeduce.season.season_from_timestamp"""

    def test_throws_without_hemisphere(self):
        """Must give a hemisphere"""
        with self.assertRaises(TypeError):
            season_from_timestamp(ts=to_datetime("2005-05-19"))

    def test_throws_if_hemisphere_not_N_or_S(self):
        ts = to_datetime("2005-05-19")
        with self.assertRaises(ValueError):
            season_from_timestamp(ts=ts, hemisphere="a")

    def test_SH_example_a(self):
        ts = to_datetime("2005-05-19")
        season = season_from_timestamp(ts=ts, hemisphere="S")
        self.assertEqual("2005", season)

    def test_SH_example_b(self):
        ts = to_datetime("2005-01-19")
        season = season_from_timestamp(ts=ts, hemisphere="S")
        self.assertEqual("Not in main season / unknown", season)

    def test_ts_None(self):
        ts = to_datetime("2005")
        season = season_from_timestamp(ts=None, hemisphere="S")
        self.assertIsNone(season)

class HemisphereFromSeasonTests(unittest.TestCase):
    """Tests for MapDeduce.season.hemisphere_from_season"""

    def test_nh_example(self):
        self.assertEqual("N", hemisphere_from_season("2005-2006"))

    def test_sh_example(self):
        self.assertEqual("S", hemisphere_from_season("2014"))

    def test_raises_example_1(self):
        with self.assertRaises(ValueError):
            hemisphere_from_season("2005-2004")

    def test_raises_example_2(self):
        with self.assertRaises(ValueError):
            hemisphere_from_season("205")


if __name__ == "__main__":
    unittest.main()
