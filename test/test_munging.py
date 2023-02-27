#!/usr/bin/env python

"""Tests for data munginig functions"""

import unittest

import pandas as pd
import os

import mapdeduce
from mapdeduce.munging import dict_from_fasta, df_from_fasta


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


if __name__ == "__main__":
    unittest.main()
