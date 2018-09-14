#!/usr/bin/env python

"""Tests for data"""

import unittest2 as unittest

import numpy as np
import pandas as pd

from MapDeduce.dataframes import CoordDf, SeqDf


class CoordDfPairedDistTests(unittest.TestCase):
    """Tests for CoordDf.paired_distances"""

    def test_array_returned(self):
        """Should return an np.ndarray"""
        size, ndim = 4, 2
        df = pd.DataFrame(np.random.normal(size, ndim))
        other = pd.DataFrame(np.random.normal(size, ndim))
        cdf = CoordDf(df)
        distances = cdf.paired_distances(other)
        self.assertIsInstance(distances, np.ndarray)

    def test_len_array_returned(self):
        """Should return an np.ndarray of particular length"""
        size, ndim = 4, 2
        df = pd.DataFrame(np.random.normal(size, ndim))
        other = pd.DataFrame(np.random.normal(size, ndim))
        cdf = CoordDf(df)
        distances = cdf.paired_distances(other)
        self.assertEqual(size, distances.shape[0])

    def test_mismatch_index_dim_raises(self):
        """If other has different dimensions, raise ValueError

        Here, index len mismatch.
        """
        size, ndim = 4, 2
        df = pd.DataFrame(np.random.normal(size, ndim))

        size += 1
        other = pd.DataFrame(np.random.normal(size, ndim))
        cdf = CoordDf(df)

        with self.assertRaises(ValueError):
            cdf.paired_distances(other)

    def test_mismatch_column_dim_raises(self):
        """If other has different dimensions, raise ValueError

        Here, column len mismatch.
        """
        size, ndim = 4, 2
        df = pd.DataFrame(np.random.normal(size, ndim))

        ndim += 1
        other = pd.DataFrame(np.random.normal(size, ndim))
        cdf = CoordDf(df)

        with self.assertRaises(ValueError):
            cdf.paired_distances(other)

    def test_computation_1dim(self):
        """Test correct distances are computed, 1 dim"""
        df = pd.DataFrame({0: [0, 1, 2, 3]})
        other = pd.DataFrame({0: [1, 1, 2.5, -1]})
        cdf = CoordDf(df)
        expect = 1, 0, 0.5, 4
        result = tuple(cdf.paired_distances(other))
        self.assertEqual(expect, result)

    def test_computation_2dim(self):
        """Test correct distances are computed, 2 dim"""
        df = pd.DataFrame({0: [0, 1, 2], 1: [0, 0, 5]})
        other = pd.DataFrame({0: [0, 0, -1.5], 1: [0, 4, -3]})
        cdf = CoordDf(df)
        expect = 0, 17 ** 0.5, (3.5 ** 2 + 8 ** 2) ** 0.5
        result = tuple(cdf.paired_distances(other))
        self.assertEqual(expect, result)

class SeqDfConsensusTests(unittest.TestCase):
    """Tests for MapDeduce.dataframes.SeqDf.consensus."""

    def setUp(self):
        """
        Position 1 tests what happens with a tie - should produce an X.
        Position 2 tests that X doesn't contribute to consensus.
        Position 3 tests that - doesn't contribute to consensus.
        Position 5 tests a uniform site.
        Position 5 tests the most abundant amino acid.
        Position 6 tests that NaN does not contribute to consensus.
        """
        df = pd.DataFrame.from_dict({
            #            1    2    3    4    5
            "strainA": ["A", "N", "_", "K", "S", "E"],
            "strainB": ["A", "X", "_", "K", "T", "E"],
            "strainC": ["D", "X", "_", "K", "S", "E"],
            "strainC": ["D", "X", "R", "K", "S", None]},
            orient="index", columns=list(range(1, 7)))
        self.sdf = SeqDf(df)

    def test_returns_series(self):
        cons = self.sdf.consensus()
        self.assertIsInstance(cons, pd.Series)

    def test_length(self):
        cons = self.sdf.consensus()
        self.assertEqual(self.sdf.df.shape[1], cons.shape[0])

    def test_sequence(self):
        cons = self.sdf.consensus()
        self.assertEqual("XNRKSE", "".join(cons))

if __name__ == "__main__":
    unittest.main()
