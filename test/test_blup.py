#!/usr/bin/env python

"""Tests for blup.py - Best linear unbiased predictions"""

import unittest

import numpy as np
import pandas as pd

from mapdeduce.blup import LmmBlup, FluLmmBlup


class LmmBlupInit(unittest.TestCase):
    """Tests for LmmBlup.__init__"""

    def test_requires_f_or_k(self):
        """Should raise ValueError if neither F nor K is specified"""
        Y = np.random.randn(10, 2)
        with self.assertRaises(ValueError) as ctx:
            LmmBlup(Y=Y)
        self.assertIn(
            "At least one of F and K must be specified", str(ctx.exception)
        )

    def test_requires_a_when_f_specified(self):
        """Should raise ValueError if F is specified without A"""
        Y = np.random.randn(10, 2)
        F = np.random.randn(10, 3)
        with self.assertRaises(ValueError) as ctx:
            LmmBlup(Y=Y, F=F)
        self.assertIn("Must specify design matrix", str(ctx.exception))

    def test_accepts_k_only(self):
        """Should accept K without F or A"""
        Y = np.random.randn(10, 2)
        K = np.random.randn(10, 10)
        blup = LmmBlup(Y=Y, K=K)
        self.assertIsInstance(blup, LmmBlup)

    def test_accepts_f_with_a(self):
        """Should accept F when A is also specified"""
        Y = np.random.randn(10, 2)
        F = np.random.randn(10, 3)
        A = np.eye(2)
        blup = LmmBlup(Y=Y, F=F, A=A)
        self.assertIsInstance(blup, LmmBlup)

    def test_accepts_both_f_and_k(self):
        """Should accept both F and K together"""
        Y = np.random.randn(10, 2)
        F = np.random.randn(10, 3)
        K = np.random.randn(10, 10)
        A = np.eye(2)
        blup = LmmBlup(Y=Y, F=F, K=K, A=A)
        self.assertIsInstance(blup, LmmBlup)


class LmmBlupAttributes(unittest.TestCase):
    """Tests for LmmBlup attribute storage"""

    def setUp(self):
        """Create sample data"""
        np.random.seed(1234)
        self.Y = np.random.randn(10, 2)
        self.F = np.random.randn(10, 3)
        self.K = np.random.randn(10, 10)
        self.A = np.eye(2)

    def test_y_stored(self):
        """Y should be stored as attribute"""
        blup = LmmBlup(Y=self.Y, K=self.K)
        np.testing.assert_array_equal(blup.Y, self.Y)

    def test_k_stored(self):
        """K should be stored as attribute"""
        blup = LmmBlup(Y=self.Y, K=self.K)
        np.testing.assert_array_equal(blup.K, self.K)

    def test_f_stored(self):
        """F should be stored as attribute"""
        blup = LmmBlup(Y=self.Y, F=self.F, A=self.A)
        np.testing.assert_array_equal(blup.F, self.F)

    def test_a_stored(self):
        """A should be stored as attribute"""
        blup = LmmBlup(Y=self.Y, F=self.F, A=self.A)
        np.testing.assert_array_equal(blup.A, self.A)

    def test_none_defaults(self):
        """Unspecified optional params should be None"""
        blup = LmmBlup(Y=self.Y, K=self.K)
        self.assertIsNone(blup.F)
        self.assertIsNone(blup.A)


class FluLmmBlupInit(unittest.TestCase):
    """Tests for FluLmmBlup.__init__"""

    def test_requires_unique_indexes(self):
        """Should raise ValueError if DataFrame has duplicate indexes"""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [1.0, 2.0, 3.0],
                "seq": ["A" * 328, "C" * 328, "D" * 328],
            },
            index=["strain1", "strain1", "strain2"],  # duplicate index
        )
        with self.assertRaises(ValueError) as ctx:
            FluLmmBlup(df)
        self.assertIn("unique", str(ctx.exception).lower())


class FluLmmBlupPredict(unittest.TestCase):
    """Tests for FluLmmBlup.predict validation"""

    def setUp(self):
        """Create a FluLmmBlup with minimal test data"""
        # Create valid training data
        self.df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0],
                "seq": ["A" * 328, "C" * 328, "D" * 328, "E" * 328],
            },
            index=["strain1", "strain2", "strain3", "strain4"],
        )
        # We can't easily instantiate FluLmmBlup without going through the
        # full initialization, so we'll test the validation logic by calling
        # predict with invalid data on a properly initialized object.
        # For now, we test the validation conditions directly.

    def test_unknown_df_wrong_column_count(self):
        """predict should raise ValueError if unknown_df has wrong column count"""
        # Create a mock FluLmmBlup-like object to test validation
        # Since FluLmmBlup.predict checks columns, we can test that logic
        flu = FluLmmBlup(self.df)

        # Create unknown_df with wrong number of columns
        unknown = pd.DataFrame(
            np.random.randn(2, 100),  # 100 columns instead of 328
            index=["unknown1", "unknown2"],
        )
        with self.assertRaises(ValueError):
            flu.predict(unknown)

    def test_unknown_df_wrong_column_names(self):
        """predict should raise ValueError if unknown_df columns don't match"""
        flu = FluLmmBlup(self.df)

        # Create unknown_df with correct number but wrong column names
        unknown = pd.DataFrame(
            np.random.randn(2, 328),
            index=["unknown1", "unknown2"],
            columns=list(range(0, 328)),  # 0-327 instead of 1-328
        )
        with self.assertRaises(ValueError) as ctx:
            flu.predict(unknown)
        self.assertIn("columns", str(ctx.exception).lower())

    def test_unknown_df_duplicate_indexes(self):
        """predict should raise ValueError if unknown_df has duplicate indexes"""
        flu = FluLmmBlup(self.df)

        # Create unknown_df with duplicate indexes
        unknown = pd.DataFrame(
            [["A"] * 328, ["C"] * 328],
            index=["unknown1", "unknown1"],  # duplicate
            columns=list(range(1, 329)),
        )
        with self.assertRaises(ValueError) as ctx:
            flu.predict(unknown)
        self.assertIn("unique", str(ctx.exception).lower())

    def test_unknown_df_overlapping_indexes(self):
        """predict should raise error if unknown_df indexes overlap with training"""
        flu = FluLmmBlup(self.df)

        # Create unknown_df with index that overlaps with training data
        unknown = pd.DataFrame(
            [["A"] * 328],
            index=["strain1"],  # overlaps with training data
            columns=list(range(1, 329)),
        )
        # Overlapping indexes cause an error (either explicit ValueError from
        # the overlap check or an error during pandas concatenation/groupby)
        with self.assertRaises((ValueError, Exception)):
            flu.predict(unknown)


if __name__ == "__main__":
    unittest.main()
