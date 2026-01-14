#!/usr/bin/env python

"""Tests for data"""

import unittest

import numpy as np
import pandas as pd

from mapdeduce.dataframes import CoordDf, SeqDf


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


class SeqDfMergeDuplicateDummiesTests(unittest.TestCase):
    """Tests for SeqDf.merge_duplicate_dummies"""

    def test_merges_identical_snps(self):
        """Identical SNPs should be merged with pipe-separated names"""
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = pd.DataFrame(
            {
                "1A": [0.0, 1.0, 0.0, 1.0],
                "2B": [0.0, 1.0, 0.0, 1.0],  # identical to 1A
                "3C": [1.0, 1.0, 0.0, 0.0],  # different
            },
            index=["s1", "s2", "s3", "s4"],
        )

        result = sdf.merge_duplicate_dummies()

        self.assertEqual(2, result.shape[1])
        self.assertIn("1A|2B", result.columns)
        self.assertIn("3C", result.columns)

    def test_merges_complement_snps(self):
        """Complement SNPs should be merged with minus prefix"""
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = pd.DataFrame(
            {
                "1A": [0.0, 1.0, 0.0, 1.0],
                "2B": [1.0, 0.0, 1.0, 0.0],  # complement of 1A
                "3C": [1.0, 1.0, 0.0, 0.0],  # different
            },
            index=["s1", "s2", "s3", "s4"],
        )

        result = sdf.merge_duplicate_dummies()

        self.assertEqual(2, result.shape[1])
        # 1A comes first alphabetically, 2B is complement
        self.assertIn("1A|-2B", result.columns)
        self.assertIn("3C", result.columns)

    def test_first_aap_defines_pattern(self):
        """First AAP in merged name should match the stored values"""
        dummies = pd.DataFrame(
            {
                "1A": [0.0, 1.0, 0.0, 1.0],
                "2B": [1.0, 0.0, 1.0, 0.0],  # complement of 1A
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies()

        # Find the merged column
        merged_col = next(c for c in result.columns if "1A" in c)

        # Stored values should match the first AAP's original values
        expected_values = [0, 1, 0, 1]
        actual_values = list(result[merged_col].values)

        self.assertEqual(expected_values, actual_values)

    def test_merge_complements_false(self):
        """With merge_complements=False, complements should not be merged"""
        dummies = pd.DataFrame(
            {
                "1A": [0.0, 1.0, 0.0, 1.0],
                "2B": [1.0, 0.0, 1.0, 0.0],  # complement of 1A
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies(merge_complements=False)

        # Both should remain separate
        self.assertEqual(2, result.shape[1])
        self.assertIn("1A", result.columns)
        self.assertIn("2B", result.columns)

    def test_merges_duplicates_and_complements_together(self):
        """Duplicates and complements of those duplicates should all merge"""
        dummies = pd.DataFrame(
            {
                "1A": [0.0, 1.0, 0.0, 1.0],
                "2B": [0.0, 1.0, 0.0, 1.0],  # duplicate of 1A
                "3C": [1.0, 0.0, 1.0, 0.0],  # complement of 1A and 2B
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies()

        # All three should merge into one column
        self.assertEqual(1, result.shape[1])

        self.assertEqual("1A|2B|-3C", result.columns[0])

    def test_all_complements_first_becomes_normal(self):
        """Duplicates should merge correctly regardless of their values.

        Internally, merge_duplicate_dummies uses lexicographic comparison to
        pick a "canonical" form for grouping SNPs. For values like [1,0,1,1],
        the complement [0,1,0,0] is lexicographically smaller, so [0,1,0,0]
        becomes the canonical key and the SNP is internally marked as a
        "complement" of that canonical form.

        This test checks that when all SNPs in a group are internally marked
        as "complements" (because their values > their complement values):
        1. The merged column name doesn't start with "-"
        2. The stored values match the original SNP values (not the canonical)
        """
        # Values where complement is smaller lexicographically
        dummies = pd.DataFrame(
            {
                "1A": [1.0, 0.0, 1.0, 1.0],
                "2B": [1.0, 0.0, 1.0, 1.0],  # duplicate of 1A
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies()

        # Should merge, first AAP should not have minus
        merged_col = result.columns[0]
        self.assertFalse(merged_col.startswith("-"))

        # Values should match original 1A
        np.testing.assert_array_equal(
            result[merged_col].values, dummies["1A"].values
        )

    def test_inplace_true(self):
        """inplace=True should modify self.dummies"""
        dummies = pd.DataFrame(
            {
                "1A": [0.0, 1.0, 0.0, 1.0],
                "2B": [0.0, 1.0, 0.0, 1.0],
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies(inplace=True)

        self.assertIsNone(result)
        self.assertEqual(1, sdf.dummies.shape[1])

    def test_inplace_false(self):
        """inplace=False should return new DataFrame, not modify original"""
        dummies = pd.DataFrame(
            {
                "1A": [0.0, 1.0, 0.0, 1.0],
                "2B": [0.0, 1.0, 0.0, 1.0],
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies.copy()

        result = sdf.merge_duplicate_dummies(inplace=False)

        self.assertIsNotNone(result)
        self.assertEqual(2, sdf.dummies.shape[1])  # original unchanged
        self.assertEqual(1, result.shape[1])

    def test_empty_dataframe(self):
        """Empty dummies DataFrame should return empty DataFrame"""
        dummies = pd.DataFrame(index=["s1", "s2", "s3"])
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies()

        self.assertEqual((3, 0), result.shape)

    def test_single_column(self):
        """Single column should be returned unchanged"""
        dummies = pd.DataFrame(
            {"1A": [0.0, 1.0, 0.0, 1.0]},
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies()

        self.assertEqual(1, result.shape[1])
        self.assertIn("1A", result.columns)
        np.testing.assert_array_equal(result["1A"].values, [0, 1, 0, 1])

    def test_all_independent_snps(self):
        """All independent SNPs should remain separate"""
        dummies = pd.DataFrame(
            {
                "1A": [0.0, 0.0, 1.0, 1.0],
                "2B": [0.0, 1.0, 0.0, 1.0],
                "3C": [0.0, 1.0, 1.0, 0.0],
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies()

        self.assertEqual(list(dummies.columns), list(result.columns))

    def test_multiple_independent_groups(self):
        """Multiple independent groups should each merge separately"""
        dummies = pd.DataFrame(
            {
                # Group 1: 1A and 2B are duplicates
                "1A": [0.0, 0.0, 1.0, 1.0],
                "2B": [0.0, 0.0, 1.0, 1.0],
                # Group 2: 3C and 4D are complements
                "3C": [0.0, 1.0, 0.0, 1.0],
                "4D": [1.0, 0.0, 1.0, 0.0],
                # Group 3: 5E is independent (not dup/complement of others)
                "5E": [0.0, 0.0, 0.0, 1.0],
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies()

        self.assertEqual(["1A|2B", "3C|-4D", "5E"], list(result.columns))

    def test_row_index_preserved(self):
        """Row index (strain names) should be preserved"""
        index = ["strain_A", "strain_B", "strain_C", "strain_D"]
        dummies = pd.DataFrame(
            {
                "1A": [0.0, 1.0, 0.0, 1.0],
                "2B": [0.0, 1.0, 0.0, 1.0],
            },
            index=index,
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        result = sdf.merge_duplicate_dummies()

        self.assertEqual(list(result.index), index)

    def test_raises_on_pipe_in_name(self):
        """Should raise ValueError if SNP name contains '|'"""
        dummies = pd.DataFrame(
            {
                "1A|2B": [0.0, 1.0, 0.0, 1.0],
                "3C": [1.0, 0.0, 1.0, 0.0],
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        with self.assertRaises(ValueError) as err:
            sdf.merge_duplicate_dummies()

        self.assertIn("1A|2B", str(err.exception))

    def test_raises_on_dash_in_name(self):
        """Should raise ValueError if SNP name contains '-'"""
        dummies = pd.DataFrame(
            {
                "1A-B": [0.0, 1.0, 0.0, 1.0],
                "3C": [1.0, 0.0, 1.0, 0.0],
            },
            index=["s1", "s2", "s3", "s4"],
        )
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = dummies

        with self.assertRaises(ValueError) as err:
            sdf.merge_duplicate_dummies()

        self.assertIn("1A-B", str(err.exception))


class SeqDfMergeTests(unittest.TestCase):
    """Tests for mapdeduce.dataframes.SeqDf.merge_duplicate_strains."""

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
