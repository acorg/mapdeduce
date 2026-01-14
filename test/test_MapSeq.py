#!/usr/bin/env python

"""Tests for MapSeq class"""

import unittest

import numpy as np
import pandas as pd

import mapdeduce
from mapdeduce.mapseq import MapSeq, OrderedMapSeq


class MapSeqAttributes(unittest.TestCase):
    """Tests for MapSeq class attributes"""

    def setUp(self):
        """Sequences and coordinates to use in tests"""
        seq_df = pd.DataFrame(
            {
                1: ("Q", "Q", "Q", "Q"),
                2: ("K", "K", "N", "K"),
                3: ("L", "P", "A", "-"),
            },
            index=("strain1", "strain2", "strain3", "strain5"),
        )
        coord_df = pd.DataFrame(
            {
                "x": (0, 0, 1, 1),
                "y": (0, 1, 0, 1),
            },
            index=("strain1", "strain2", "strain3", "strain4"),
        )
        self.ms = MapSeq(seq_df=seq_df, coord_df=coord_df)

    def test_common_strains(self):
        """
        MapSeq common_strain attribute should be a set comprising the
        intersection of the strains in the sequence and coordinate dfs
        """
        expect = {"strain1", "strain2", "strain3"}
        self.assertEqual(expect, self.ms.strains_in_both)

    def test_seq_in_both_indexes(self):
        """Indexes of self.seq_in_both should match strains_in_both"""
        self.assertEqual(
            self.ms.strains_in_both, set(self.ms.seq_in_both.index)
        )

    def test_coords_in_both_indexes(self):
        """Indexes of self.coords_in_both should match strains_in_both"""
        self.assertEqual(
            self.ms.strains_in_both, set(self.ms.coords_in_both.index)
        )

    def test_unkown_sequence(self):
        """
        Anything in fasta that isn't one of the 20 standard amino acids
        should be NaN.
        """
        self.assertTrue(np.isnan(self.ms.all_seqs.loc["strain5", 3]))


class MapSeqDuplicates(unittest.TestCase):
    """Tests for handling duplicate sequences in input DataFrames"""

    def setUp(self):
        """Sequences and coordinates to use in tests"""
        index = (
            "strain1",
            "strain1",
            "strain2",
            "strain3",
            "strain3",
            "strain4",
            "strain4",
        )
        seq_df = pd.DataFrame(
            {
                1: ("Q", "Q", "Q", "Q", "Q", "D", "D"),
                2: ("K", "K", "K", "K", "K", "A", "A"),
                3: ("L", "L", "L", "L", "A", "V", "V"),
            },
            index=index,
        )
        coord_df = pd.DataFrame(
            {
                "x": (0, 0, 0, 1, 0, 1, 0),
                "y": (0, 1, 0, 1, 1, 0, 1),
            },
            index=index,
        )
        self.ms = MapSeq(seq_df=seq_df, coord_df=coord_df)

    def test_same_sequences_different_index(self):
        """
        Strains with different indexes, but the same sequence should
        be kept.
        """
        for test in "strain1", "strain2":
            self.assertIn(test, set(self.ms.all_seqs.index))

    def test_different_sequences_same_index_len(self):
        """
        Duplicate indexes should be removed. Ambiguous positions (due to
        different sequences) should be replaced with X.

        strain3 is an example. Sequence should be (Q, K, nan)
        """
        self.assertIsInstance(
            self.ms.all_seqs.loc["strain3", :], pd.core.frame.Series
        )

    def test_different_sequences_same_index_value(self):
        """
        Duplicate indexes should be removed. Ambiguous positions (due to
        different sequences) should be replaced with X.

        strain3 is an example. Sequence should be (Q, K, nan)
        """
        self.assertIs(np.nan, self.ms.all_seqs.loc["strain3", 3])

    def test_same_sequences_same_index(self):
        """Strains with duplicate index and sequence should be removed."""
        for test in "strain1", "strain4":
            self.assertIn(test, set(self.ms.all_seqs.index))


class MapSeqStrainsWithCombinations(unittest.TestCase):
    """Tests for MapSeq.strains_with_combinations"""

    def setUp(self):
        """Sequences and coordinates to use in tests"""
        seq_df = pd.DataFrame(
            {
                1: ("Q", "Q", "Q", "Q"),
                2: ("K", "K", "N", "K"),
                3: ("L", "P", "A", "L"),
            },
            index=("strain1", "strain2", "strain3", "strain5"),
        )
        coord_df = pd.DataFrame(
            {
                "x": (0, 0, 1, 1),
                "y": (0, 1, 0, 1),
            },
            index=("strain1", "strain2", "strain3", "strain4"),
        )
        self.ms = MapSeq(seq_df=seq_df, coord_df=coord_df)

    def test_returns_df(self):
        """Should return a df"""
        self.assertIsInstance(
            self.ms.strains_with_combinations({1: "Q"}, verbose=False),
            pd.core.frame.DataFrame,
        )

    def test_returns_df_combinations_absent(self):
        """
        Should return a df, even if no strains have the requested
        combination
        """
        self.assertIsInstance(
            self.ms.strains_with_combinations({1: "L"}, verbose=False),
            pd.core.frame.DataFrame,
        )

    def test_correct_strains_1(self):
        """
        Test correct strains returned.
        Should only return strains in seq_in_both and coords_in_both.
        (I.e. out of "strain1", "strain2", "strain3")
        """
        output = self.ms.strains_with_combinations({1: "Q"}, verbose=False)
        self.assertEqual(
            set(("strain1", "strain2", "strain3")), set(output.index)
        )

    def test_correct_strains_3(self):
        """
        Test correct strains returned.
        Expect no strains.
        """
        output = self.ms.strains_with_combinations({1: "K"}, verbose=False)
        expect = set()
        self.assertEqual(expect, set(output.index))

    def test_raises_value_error_positions_absent(self):
        """Should raise a value error when a position requested is absent"""
        with self.assertRaises(ValueError):
            self.ms.strains_with_combinations({4: "K"}, verbose=False)


class MapSeqDuplicateSeqeunces(unittest.TestCase):
    """Tests for MapSeq.duplicate_sequences"""

    def setUp(self):
        """Sequences and coordinates to use in tests"""
        seq_df = pd.DataFrame(
            {
                1: ("Q", "Q", "Q", "Q", "Q"),
                2: ("K", "K", "N", "K", "N"),
                3: ("L", "L", "A", "L", "-"),
            },
            index=("strain1", "strain2", "strain3", "strain5", "strain6"),
        )
        coord_df = pd.DataFrame(
            {
                "x": (0, 0, 1, 1, 0),
                "y": (0, 1, 0, 1, 0),
            },
            index=("strain1", "strain2", "strain3", "strain4", "strain6"),
        )
        self.ms = MapSeq(seq_df=seq_df, coord_df=coord_df)

    def test_returns_pd_groupby(self):
        """Should return pd.core.groupby.DataFrameGroupBy"""
        self.assertIsInstance(
            self.ms.duplicate_sequences(), pd.core.groupby.DataFrameGroupBy
        )

    def test_correct_groups_1(self):
        """
        Test correct strains found.

        (Only strain1-3 should be found).
        """
        grouped = self.ms.duplicate_sequences()
        strains = grouped.groups[("Q", "K", "L")]
        test = set(strains)
        self.assertEqual({"strain1", "strain2"}, test)

    def test_correct_groups_2(self):
        """
        Test correct strains found.

        (Only strain1-3 should be found).
        """
        grouped = self.ms.duplicate_sequences()
        strains = grouped.groups[("Q", "N", "A")]
        test = set(strains)
        self.assertEqual(
            {
                "strain3",
            },
            test,
        )

    def test_unknown_sequence(self):
        """
        Any non-amino acids in sequences (e.g. "X" / "-" in fasta) should not
        be included.

        strain6 should match strain3
        """
        grouped = self.ms.duplicate_sequences()
        strains = grouped.groups[("Q", "N", "A")]
        test = set(strains)
        self.assertEqual({"strain3"}, test)


class OrderedMapSeqTests(unittest.TestCase):
    def setUp(self):
        """Sequences and coordinates to use in tests"""
        seq_df = pd.DataFrame(
            {
                1: list("QQQQQA"),
                2: list("KKNKNA"),
                3: list("LLAL-A"),
            },
            index="flu1 flu2 flu3 flu5 flu6 flu7".split(),
        )

        coord_df = pd.DataFrame(
            {
                "x": (0, 0, 1, 1, 0, np.nan),
                "y": (0, 1, 0, 1, 0, np.nan),
            },
            index="flu2 flu1 flu3 flu4 flu6 flu7".split(),
        )

        self.oms = OrderedMapSeq(seq_df=seq_df, coord_df=coord_df)

    def test_attribute_coord(self):
        self.assertIsInstance(self.oms.coord, mapdeduce.dataframes.CoordDf)

    def test_indexes_contain_intersection(self):
        """Indexes of the sequence and coordinate dataframe should contain
        the intersection of the original dataframes.

        flu7 should be dropped because it's coordinates are nan.
        """
        expect = set("flu1 flu2 flu3 flu6".split())
        self.assertEqual(expect, set(self.oms.coord.df.index))
        self.assertEqual(expect, set(self.oms.seqs.df.index))

    def test_reordering(self):
        """Sequence and coordinate dataframes should be reordered such that
        their indexes match.
        """
        self.assertEqual(
            list(self.oms.coord.df.index), list(self.oms.seqs.df.index)
        )


if __name__ == "__main__":
    unittest.main()
