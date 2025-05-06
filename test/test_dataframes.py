#!/usr/bin/env python

"""Tests for data"""

import unittest

import numpy as np
import pandas as pd

from mapdeduce.dataframes import CoordDf, SeqDf, remove_inverse_profiles


class CoordDfPairedDistTests(unittest.TestCase):
    """Tests for CoordDf.paired_distances"""

    def test_array_returned(self):
        """Should return an np.ndarray"""
        size, ndim = 4, 2
        df = pd.DataFrame(np.random.randn(size, ndim))
        other = pd.DataFrame(np.random.randn(size, ndim))
        cdf = CoordDf(df)
        distances = cdf.paired_distances(other)
        self.assertIsInstance(distances, np.ndarray)

    def test_len_array_returned(self):
        """Should return an np.ndarray of particular length"""
        size, ndim = 4, 2
        df = pd.DataFrame(np.random.randn(size, ndim))
        other = pd.DataFrame(np.random.randn(size, ndim))
        cdf = CoordDf(df)
        distances = cdf.paired_distances(other)
        self.assertEqual(size, distances.shape[0])

    def test_mismatch_index_dim_raises(self):
        """If other has different dimensions, raise ValueError

        Here, index len mismatch.
        """
        size, ndim = 4, 2
        df = pd.DataFrame(np.random.randn(size, ndim))

        size += 1
        other = pd.DataFrame(np.random.randn(size, ndim))
        cdf = CoordDf(df)

        with self.assertRaises(ValueError):
            cdf.paired_distances(other)

    def test_mismatch_column_dim_raises(self):
        """If other has different dimensions, raise ValueError

        Here, column len mismatch.
        """
        size, ndim = 4, 2
        df = pd.DataFrame(np.random.randn(size, ndim))

        ndim += 1
        other = pd.DataFrame(np.random.randn(size, ndim))
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
        expect = 0, 17**0.5, (3.5**2 + 8**2) ** 0.5
        result = tuple(cdf.paired_distances(other))
        self.assertEqual(expect, result)


class SeqDfConsensusTests(unittest.TestCase):
    """Tests for mapdeduce.dataframes.SeqDf.consensus."""

    def setUp(self):
        """
        Position 1 tests what happens with a tie - should produce an X.
        Position 2 tests that X doesn't contribute to consensus.
        Position 3 tests that - doesn't contribute to consensus.
        Position 5 tests a uniform site.
        Position 5 tests the most abundant amino acid.
        Position 6 tests that NaN does not contribute to consensus.
        """
        df = pd.DataFrame.from_dict(
            {
                #            1    2    3    4    5
                "strainA": ["A", "N", "-", "K", "S", "E"],
                "strainB": ["A", "X", "-", "K", "T", "E"],
                "strainC": ["D", "X", "-", "K", "S", "E"],
                "strainD": ["D", "X", "R", "K", "S", None],
            },
            orient="index",
            columns=list(range(1, 7)),
        )
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


class SeqDfGeneralTests(unittest.TestCase):

    def test_groupby_at_site(self):
        df = pd.DataFrame.from_dict(
            dict(a=list("AKN"), b=list("RRD"), c=list("KRE")),
            orient="index",
            columns=list(range(1, 4)),
        )
        sdf = SeqDf(df)
        result = sdf.groupby_amino_acid_at_site(2)
        self.assertEqual(("a",), tuple(result["K"]))
        self.assertEqual(("b", "c"), tuple(sorted(result["R"])))


class SeqDfDummiesTests(unittest.TestCase):

    def test_get_dummies(self):
        """
        get_dummies should attach a pandas DataFrame containing dummies
        """
        df = pd.DataFrame({1: list("AAAAA"), 2: list("KKKKK"), 3: list("TTTTT")})
        sdf = SeqDf(df)
        sdf.get_dummies()  # Attaches dummies
        self.assertIsInstance(sdf.dummies, pd.DataFrame)

    def test_dummies_columns(self):
        """
        Test dummies columns are as expected.
        """
        df = pd.DataFrame({1: list("AAAAA"), 2: list("KKKKK"), 3: list("TTTTT")})
        sdf = SeqDf(df)
        sdf.get_dummies()
        expect = ["1A", "2K", "3T"]
        self.assertEqual(expect, list(sdf.dummies))

    def test_remove_invariant(self):
        """
        remove_invariant should remove sites that are constant.
        """
        df = pd.DataFrame({1: list("AAAAA"), 2: list("KKKKK"), 3: list("TTTTN")})
        sdf = SeqDf(df).remove_invariant()
        sdf.get_dummies()
        self.assertEqual({"3N", "3T"}, set(sdf.dummies))

    def test_merge_duplicate(self):
        """
        remove_invariant should remove sites that are constant.
        """
        sdf = SeqDf(
            pd.DataFrame(
                {
                    # Here 1 and 2 both have the same pattern, so they should be merged
                    1: list("AAAAK"),
                    2: list("TTTTN"),
                    3: list("QKRPP"),
                }
            )
        )
        sdf.get_dummies()

        # Before calling merge_duplicates these columns should be present
        self.assertIn("1A", sdf.dummies.columns)
        self.assertIn("1K", sdf.dummies.columns)
        self.assertIn("2T", sdf.dummies.columns)
        self.assertIn("2N", sdf.dummies.columns)

        sdf.merge_duplicate_dummies()

        # After calling merge_duplicates these columns should not be present
        self.assertNotIn("1A", sdf.dummies.columns)
        self.assertNotIn("1K", sdf.dummies.columns)
        self.assertNotIn("2T", sdf.dummies.columns)
        self.assertNotIn("2N", sdf.dummies.columns)

        # This new column should be present
        self.assertIn("1K|2N", sdf.dummies.columns)

    def test_merge_duplicates_opposites(self):
        """
        Test merging duplicate dummies when amino acid polymorphisms are encoded with opposite use
        of 0/1.

        For instance if
        """
        sdf = SeqDf(
            pd.DataFrame(
                {
                    # Here 1 and 2 both have the same pattern, so they should be merged. But both
                    # 1A|2T and 1K|2N should not be produced.
                    1: list("AAAKKK"),
                    2: list("TTTNNN"),
                }
            )
        )
        sdf.get_dummies()
        sdf.merge_duplicate_dummies()
        self.assertEqual({"1K|2N"}, set(sdf.dummies.columns))


class TestRemoveInverseProfiles(unittest.TestCase):
    def test_simple(self):
        """Basic test case"""
        dummies = pd.DataFrame({"A": [0, 0, 0, 1, 1, 1], "B": [1, 1, 1, 0, 0, 0]})
        new, _ = remove_inverse_profiles(dummies)
        self.assertEqual({"A"}, set(new.columns))

    def test_no_inverses(self):
        """With no inverses, the output should match the input."""
        dummies = pd.DataFrame(
            {
                "A": [0, 0, 0, 1, 1, 1],
                "B": [1, 1, 1, 0, 0, 1],
                "C": [1, 0, 1, 0, 1, 0],
                "D": [1, 1, 1, 0, 1, 1],
            }
        )
        new, _ = remove_inverse_profiles(dummies)
        self.assertTrue(all(dummies == new))

    def test_names_of_inverse_aaps_returned(self):
        """Dict should contain names of inverse AAPs"""
        dummies = pd.DataFrame({"A": [0, 0, 0, 1, 1, 1], "B": [1, 1, 1, 0, 0, 0]})
        _, inverse = remove_inverse_profiles(dummies)
        self.assertEqual({"A": ["B"]}, inverse)

    def test_multiple(self):
        """Dict should contain names of inverse AAPs"""
        dummies = pd.DataFrame(
            {"A": [0, 0, 0, 1, 1, 1], "B": [1, 1, 1, 0, 0, 0], "C": [1, 1, 1, 0, 0, 0]}
        )
        _, inverse = remove_inverse_profiles(dummies)
        self.assertEqual({"A": ["B", "C"]}, inverse)

    def test_order(self):
        """
        First dummy in the DataFrame should be kept.
        """
        dummies = pd.DataFrame({"B": [1, 1, 1, 0, 0, 0], "A": [0, 0, 0, 1, 1, 1]})
        new, _ = remove_inverse_profiles(dummies)
        self.assertEqual({"B"}, set(new.columns))


class SeqDfMergeTests(unittest.TestCase):
    """Tests for mapdeduce.dataframes.SeqDf.consensus."""

    def setUp(self):
        """StrainC should be replaced by the consensus of the strainC seqs."""
        data = [
            ["A", "A", "D", "D"],
            ["N", "X", "X", "X"],
            ["-", "-", "-", "R"],
            ["K", "K", "K", "K"],
            ["S", "T", "S", "S"],
            ["E", "E", "E", None],
        ]
        df = pd.DataFrame(
            data,
            index=list(range(1, 7)),
            columns=["strainA", "strainB", "strainC", "strainC"],
        ).T
        self.sdf = SeqDf(df)

    def test_df_smaller(self):
        """df should be one row shorter."""
        sdf = self.sdf.merge_duplicate_strains()
        self.assertEqual(3, sdf.df.shape[0])

    def test_update_inplace(self):
        self.sdf.merge_duplicate_strains(inplace=True)
        self.assertEqual(3, self.sdf.df.shape[0])

    def test_other_sequences_unchanged(self):
        self.sdf.merge_duplicate_strains(inplace=True)
        self.assertEqual("AN-KSE", "".join(self.sdf.df.loc["strainA"]))

    def test_strainC_is_its_consensus(self):
        self.sdf.merge_duplicate_strains(inplace=True)
        self.assertEqual("DXRKSE", "".join(self.sdf.df.loc["strainC"]))

    def test_only_single_strainC(self):
        self.sdf.merge_duplicate_strains(inplace=True)
        n = self.sdf.df.index.value_counts()["strainC"]
        self.assertEqual(1, n)

    def test_returns_seqdf(self):
        sdf = self.sdf.merge_duplicate_strains(inplace=False)
        self.assertIsInstance(sdf, SeqDf)


if __name__ == "__main__":
    unittest.main()
