#!/usr/bin/env python

"""Tests for data munging functions"""

import os
import unittest

import pandas as pd

import mapdeduce
from mapdeduce.munging import dict_from_fasta, df_from_fasta, handle_duplicate_coords


class DictFromFastaTests(unittest.TestCase):
    """Tests for reading sequences from fasta files to a dictionary"""

    def setUp(self):
        """Run df_from_fasta on a sample fasta file."""
        module_directory = os.path.dirname(mapdeduce.__path__[0])
        fasta_path = os.path.join(module_directory, "data", "test", "fasta-sample.fa")
        self.dict = dict_from_fasta(path=fasta_path)

    def test_lower_absent(self):
        """Keys should all be upper case"""
        strain = "a/zhejiang/48/2004"
        self.assertNotIn(strain, list(self.dict.keys()))

    def test_upper_present(self):
        """Keys should all be upper case"""
        strain = "a/zhejiang/48/2004".upper()
        self.assertIn(strain, list(self.dict.keys()))


class DfFromFastaTests(unittest.TestCase):
    """Tests for reading in sequence DataFrames"""

    def setUp(self):
        """Run df_from_fasta on a sample fasta file."""
        module_directory = os.path.dirname(mapdeduce.__path__[0])
        fasta_path = os.path.join(module_directory, "data", "test", "fasta-sample.fa")
        self.df = df_from_fasta(path=fasta_path, positions=(1, 2, 3, 4, 5))

    def test_df_is_dataframe(self):
        """The function should return a DataFrame"""
        self.assertIsInstance(self.df, pd.core.frame.DataFrame)

    def test_df_positions_length(self):
        """The dataframe should have five columns"""
        self.assertEqual(5, self.df.shape[1])

    def test_df_rows_length(self):
        """The dataframe should have 13 rows"""
        self.assertEqual(13, self.df.shape[0])

    def test_lookup(self):
        """Test 5th position of a/zhejiang/48/2004 is a G"""
        strain = "a/zhejiang/48/2004".upper()
        self.assertEqual("G", self.df.loc[strain, 5])


class HandleDuplicateCoordsTests(unittest.TestCase):
    """Tests for handling duplicate strains in a dataframe"""

    def setUp(self):
        """Create a dataframe with duplicate strains"""
        self.df = pd.DataFrame(
            {
                "x": [1, 1.1, 5, 5.1, 10, 20],
                "y": [1, 1.1, 5, 5.1, 10, 20],
            },
            index=[
                "strain1",
                "strain1",
                "strain2",
                "strain2",
                "strain3",
                "strain3",
            ],
        )

    def test_result_shape(self):
        """Test that the result has the correct shape"""
        result = handle_duplicate_coords(self.df, threshold=2)
        self.assertEqual(2, result.shape[0])

    def test_strain1_present(self):
        """Test that strain1 is present in the result"""
        result = handle_duplicate_coords(self.df, threshold=2)
        self.assertIn("strain1", result.index)

    def test_strain2_present(self):
        """Test that strain2 is present in the result"""
        result = handle_duplicate_coords(self.df, threshold=2)
        self.assertIn("strain2", result.index)

    def test_strain3_absent(self):
        """Test that strain3 is absent from the result"""
        result = handle_duplicate_coords(self.df, threshold=2)
        self.assertNotIn("strain3", result.index)

    def test_strain1_averaged_x(self):
        """Test that strain1's x position is averaged"""
        result = handle_duplicate_coords(self.df, threshold=2)
        self.assertAlmostEqual(1.05, result.loc["strain1", "x"])

    def test_strain2_averaged_y(self):
        """Test that strain2's y position is averaged"""
        result = handle_duplicate_coords(self.df, threshold=2)
        self.assertAlmostEqual(5.05, result.loc["strain2", "y"])

    def test_negative_threshold_raises(self):
        """Test that passing a negative threshold raises a ValueError"""
        df = pd.DataFrame({"x": [1, 1.1], "y": [1, 1.1]}, index=["strain1", "strain1"])
        with self.assertRaises(ValueError):
            handle_duplicate_coords(df, threshold=-1)


if __name__ == "__main__":
    unittest.main()
