#!/usr/bin/env python

"""Tests for data"""

import unittest2 as unittest

import numpy as np
import pandas as pd

from MapSeq.dataframes import CoordDf


class CoordDfPairedDistTests(unittest.TestCase):
    """Tests for CoordDf.paired_distances"""

    def test_array_returned(self):
        """Should return an np.ndarray"""
        size = 4
        ndim = 2
        df = pd.DataFrame(
            np.random.normal(size=size * ndim).reshape(size, ndim)
        )
        other = pd.DataFrame(
            np.random.normal(size=size * ndim).reshape(size, ndim)
        )
        cdf = CoordDf(df)
        distances = cdf.paired_distances(other)
        self.assertIsInstance(distances, np.ndarray)

    def test_len_array_returned(self):
        """Should return an np.ndarray of particular length"""
        size = 4
        ndim = 2
        df = pd.DataFrame(
            np.random.normal(size=size * ndim).reshape(size, ndim)
        )
        other = pd.DataFrame(
            np.random.normal(size=size * ndim).reshape(size, ndim)
        )
        cdf = CoordDf(df)
        distances = cdf.paired_distances(other)
        self.assertEqual(size, distances.shape[0])

    def test_mismatch_index_dim_raises(self):
        """If other has different dimensions, raise ValueError

        Here, index len mismatch.
        """
        size = 4
        ndim = 2
        df = pd.DataFrame(
            np.random.normal(size=size * ndim).reshape(size, ndim)
        )

        size += 1
        other = pd.DataFrame(
            np.random.normal(size=size * ndim).reshape(size, ndim)
        )
        cdf = CoordDf(df)

        with self.assertRaises(ValueError):
            cdf.paired_distances(other)

    def test_mismatch_column_dim_raises(self):
        """If other has different dimensions, raise ValueError

        Here, column len mismatch.
        """
        size = 4
        ndim = 2
        df = pd.DataFrame(
            np.random.normal(size=size * ndim).reshape(size, ndim)
        )

        ndim += 1
        other = pd.DataFrame(
            np.random.normal(size=size * ndim).reshape(size, ndim)
        )
        cdf = CoordDf(df)

        with self.assertRaises(ValueError):
            cdf.paired_distances(other)

    def test_computation_1dim(self):
        """Test correct distances are computed, 1 dim"""
        df = pd.DataFrame({
            0: [0, 1, 2, 3]
        })

        other = pd.DataFrame({
            0: [1, 1, 2.5, -1]
        })
        cdf = CoordDf(df)

        expect = 1, 0, 0.5, 4
        result = tuple(cdf.paired_distances(other))

        self.assertEqual(expect, result)

    def test_computation_2dim(self):
        """Test correct distances are computed, 2 dim"""
        df = pd.DataFrame({
            0: [0, 1, 2],
            1: [0, 0, 5]
        })

        other = pd.DataFrame({
            0: [0, 0, -1.5],
            1: [0, 4, -3]

        })
        cdf = CoordDf(df)

        expect = 0, 17 ** 0.5, (3.5 ** 2 + 8 ** 2) ** 0.5
        result = tuple(cdf.paired_distances(other))

        self.assertEqual(expect, result)


if __name__ == "__main__":

    unittest.main()
