#!/usr/bin/env python

"""Tests for data"""

import unittest

import numpy as np
import pandas as pd

from mapdeduce.dataframes import CoordDf, SeqDf, columns_at_positions


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


class SeqDfGetDummiesTests(unittest.TestCase):
    """Tests for mapdeduce.dataframes.SeqDf.get_dummies."""

    def setUp(self):
        df = pd.DataFrame(
            [
                ["A", "K", "D"],
                ["A", "K", "D"],
                ["A", "N", "D"],
                ["A", "N", "E"],
            ],
            index=["strain1", "strain2", "strain3", "strain4"],
            columns=[145, 156, 189],
        )
        self.sdf = SeqDf(df)

    def test_dummies_dataframe(self):
        """get_dummies should create a dummies DataFrame"""
        self.sdf.get_dummies(inplace=True)
        self.assertIsInstance(self.sdf.dummies, pd.DataFrame)

    def test_dummies_dataframe_not_inplace_returns_df(self):
        """get_dummies with inplace=False should return a DataFrame"""
        dummies = self.sdf.get_dummies(inplace=False)
        self.assertIsInstance(dummies, pd.DataFrame)
        self.assertFalse(hasattr(self.sdf, "dummies"))

    def test_dummies_shape(self):
        """Dummies DataFrame should have correct shape"""
        self.sdf.get_dummies(inplace=True)
        # There are 4 strains and 5 unique SNPs: 145A, 156K, 156N, 189D, 189E
        self.assertEqual((4, 5), self.sdf.dummies.shape)

    def test_dummies_values(self):
        """Dummies DataFrame should only contain ones and zeros"""
        self.sdf.get_dummies(inplace=True)
        self.assertTrue(self.sdf.dummies.isin([0, 1]).all().all())

    def test_dummies_column_names(self):
        """Dummies DataFrame should have correct column names"""
        self.sdf.get_dummies(inplace=True)
        expected_columns = {"145A", "156K", "156N", "189D", "189E"}
        self.assertEqual(expected_columns, set(self.sdf.dummies.columns))

    def test_dummies_row_index(self):
        """Dummies DataFrame should have same row index as original df"""
        self.sdf.get_dummies(inplace=True)
        self.assertTrue((self.sdf.df.index == self.sdf.dummies.index).all())

    def test_145A_values(self):
        """Dummies DataFrame should have correct values for 145A"""
        self.sdf.get_dummies(inplace=True)
        self.assertEqual([1, 1, 1, 1], list(self.sdf.dummies["145A"].values))

    def test_156N_values(self):
        """Dummies DataFrame should have correct values for 156N"""
        self.sdf.get_dummies(inplace=True)
        self.assertEqual([0, 0, 1, 1], list(self.sdf.dummies["156N"].values))

    def test_189E_values(self):
        """Dummies DataFrame should have correct values for 189E"""
        self.sdf.get_dummies(inplace=True)
        self.assertEqual([0, 0, 0, 1], list(self.sdf.dummies["189E"].values))


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


class SeqDfGetDummiesAtPositionsTests(unittest.TestCase):
    """Tests for SeqDf.get_dummies_at_positions with complement SNPs"""

    def setUp(self):
        """Create a SeqDf with dummy variables including complements"""
        df = pd.DataFrame(
            [
                ["A", "K", "D"],
                ["A", "K", "D"],
                ["A", "N", "D"],
                ["A", "N", "E"],
            ],
            index=["strain1", "strain2", "strain3", "strain4"],
            columns=[145, 156, 189],
        )
        self.sdf = SeqDf(df)
        self.sdf.get_dummies(inplace=True)

    def test_finds_snp_not_compound_name(self):
        """Should find SNP in simple dummy name"""
        dummies_145 = self.sdf.get_dummies_at_positions([145])
        self.assertEqual({"145A"}, dummies_145)

    def test_finds_snp_in_compound_name(self):
        """Should find SNP in compound dummy name"""
        # Before merging duplicate dummies 156K and 156N should both exist
        dummies_156 = self.sdf.get_dummies_at_positions([156])
        self.assertEqual({"156K", "156N"}, dummies_156)

        # merge duplicates to create compound names
        self.sdf.merge_duplicate_dummies(inplace=True, merge_complements=True)

        # After merge, only one dummy should exist for position 156
        dummies_156 = self.sdf.get_dummies_at_positions([156])
        self.assertEqual({"156N|-156K"}, dummies_156)

    def test_finds_dummy_using_site_in_complement(self):
        """Should find dummy when site is in a complement SNP name"""
        # mock a dummy DataFrame with a complement dummy at site 145
        self.sdf.dummies = pd.DataFrame(
            {
                "135N|-145A": [1, 1, 0, 1],
                "156K": [1, 1, 0, 0],
                "189D": [1, 1, 1, 0],
            },
            index=["strain1", "strain2", "strain3", "strain4"],
        )
        dummies_145 = self.sdf.get_dummies_at_positions([145])
        self.assertEqual({"135N|-145A"}, dummies_145)

    def test_finds_position_in_tilde_joined_collinear_group(self):
        """Should find position in a ~-joined collinear group name"""
        self.sdf.dummies = pd.DataFrame(
            {
                "145A~189D": [1, 1, 0, 0],
                "156K": [1, 1, 0, 0],
            },
            index=["strain1", "strain2", "strain3", "strain4"],
        )
        result = self.sdf.get_dummies_at_positions([189])
        self.assertEqual({"145A~189D"}, result)

    def test_finds_position_in_compound_pipe_tilde_name(self):
        """Should find position in a compound |+~ name like 145K|155S~189K"""
        self.sdf.dummies = pd.DataFrame(
            {
                "145K|155S~189K": [1, 1, 0, 0],
                "156N": [0, 0, 1, 1],
            },
            index=["strain1", "strain2", "strain3", "strain4"],
        )
        # Query position 189 — should return the full compound name
        result = self.sdf.get_dummies_at_positions([189])
        self.assertEqual({"145K|155S~189K"}, result)

        # Query position 155 — should also return the full compound name
        result = self.sdf.get_dummies_at_positions([155])
        self.assertEqual({"145K|155S~189K"}, result)


class SeqDfPruneCollinearDummiesTests(unittest.TestCase):
    """Tests for SeqDf.prune_collinear_dummies"""

    def test_inplace_true_updates_dummies_and_sets_mappings(self):
        """inplace=True updates self.dummies and sets both mapping attrs"""
        sdf = SeqDf(pd.DataFrame())
        snp = [0, 1, 0, 1, 1, 0]
        sdf.dummies = pd.DataFrame(
            {"1A": snp, "2B": snp, "3C": [1, 1, 0, 0, 1, 0]}
        )

        result = sdf.prune_collinear_dummies(threshold=0.95, inplace=True)

        self.assertIsNone(result)
        # Dummies should be updated (2 columns -> pruned)
        self.assertEqual(2, sdf.dummies.shape[1])
        self.assertTrue(hasattr(sdf, "collinear_mapping"))
        self.assertTrue(hasattr(sdf, "collinear_removed_to_kept"))

    def test_inplace_false_returns_pruned_df_no_modification(self):
        """inplace=False returns pruned DataFrame, does not modify original"""
        sdf = SeqDf(pd.DataFrame())
        snp = [0, 1, 0, 1, 1, 0]
        sdf.dummies = pd.DataFrame(
            {"1A": snp, "2B": snp, "3C": [1, 1, 0, 0, 1, 0]}
        )
        original_cols = list(sdf.dummies.columns)

        result = sdf.prune_collinear_dummies(threshold=0.95, inplace=False)

        self.assertIsNotNone(result)
        self.assertEqual(2, result.shape[1])
        # Original should be unchanged
        self.assertEqual(original_cols, list(sdf.dummies.columns))
        self.assertFalse(hasattr(sdf, "collinear_mapping"))

    def test_identical_snps_pruned_at_threshold_095(self):
        """Identical SNPs (r2=1) are pruned at threshold=0.95"""
        sdf = SeqDf(pd.DataFrame())
        snp = [0, 1, 0, 1, 1, 0]
        sdf.dummies = pd.DataFrame({"1A": snp, "2B": snp, "3C": snp})

        result = sdf.prune_collinear_dummies(threshold=0.95, inplace=False)

        self.assertEqual(1, result.shape[1])

    def test_independent_snps_all_retained(self):
        """Independent SNPs are all retained"""
        sdf = SeqDf(pd.DataFrame())
        sdf.dummies = pd.DataFrame(
            {
                "1A": [0, 1, 0, 1, 0, 1],
                "2B": [1, 0, 0, 1, 1, 0],
                "3C": [0, 0, 1, 0, 1, 1],
            }
        )

        result = sdf.prune_collinear_dummies(threshold=0.95, inplace=False)

        self.assertEqual(3, result.shape[1])

    def test_collinear_mapping_correct(self):
        """collinear_mapping maps retained -> [pruned list] correctly"""
        sdf = SeqDf(pd.DataFrame())
        snp = [0, 1, 0, 1, 1, 0]
        sdf.dummies = pd.DataFrame(
            {"1A": snp, "2B": snp, "3C": [1, 1, 0, 0, 1, 0]}
        )

        sdf.prune_collinear_dummies(threshold=0.95, inplace=True)

        self.assertIn("1A", sdf.collinear_mapping)
        self.assertEqual(["2B"], sdf.collinear_mapping["1A"])
        self.assertEqual({"2B": "1A"}, sdf.collinear_removed_to_kept)

    def test_retained_columns_with_pruned_members_get_tilde_name(self):
        """Retained columns with pruned members get ~-joined names"""
        sdf = SeqDf(pd.DataFrame())
        snp = [0, 1, 0, 1, 1, 0]
        sdf.dummies = pd.DataFrame(
            {"1A": snp, "2B": snp, "3C": [1, 1, 0, 0, 1, 0]}
        )

        sdf.prune_collinear_dummies(threshold=0.95, inplace=True)

        # 1A and 2B are identical, one should be pruned and the retained
        # column should be renamed to "1A~2B"
        self.assertIn("1A~2B", sdf.dummies.columns)
        self.assertIn("3C", sdf.dummies.columns)

    def test_columns_without_pruned_members_keep_original_name(self):
        """Columns with no pruned members keep original name"""
        sdf = SeqDf(pd.DataFrame())
        snp = [0, 1, 0, 1, 1, 0]
        sdf.dummies = pd.DataFrame(
            {"1A": snp, "2B": snp, "3C": [1, 1, 0, 0, 1, 0]}
        )

        result = sdf.prune_collinear_dummies(threshold=0.95, inplace=False)

        self.assertIn("3C", result.columns)

    def test_works_after_merge_duplicate_dummies(self):
        """Works when input names contain | from merge_duplicate_dummies"""
        sdf = SeqDf(pd.DataFrame())
        snp = [0, 1, 0, 1, 1, 0]
        sdf.dummies = pd.DataFrame(
            {
                "145K|155S": snp,
                "189K": snp,
                "169T|170A": [1, 1, 0, 0, 1, 0],
            }
        )

        sdf.prune_collinear_dummies(threshold=0.95, inplace=True)

        self.assertEqual(2, sdf.dummies.shape[1])
        # 145K|155S and 189K are identical, should be merged with ~
        self.assertIn("145K|155S~189K", sdf.dummies.columns)
        self.assertIn("169T|170A", sdf.dummies.columns)


class ColumnsAtPositionsTests(unittest.TestCase):
    """Tests for columns_at_positions standalone function"""

    def test_simple_columns(self):
        """Should find simple column names at given positions"""
        columns = ["145A", "156K", "189D"]
        result = columns_at_positions(columns, [145])
        self.assertEqual({"145A"}, result)

    def test_multiple_positions(self):
        """Should find columns at multiple positions"""
        columns = ["145A", "156K", "189D"]
        result = columns_at_positions(columns, [145, 189])
        self.assertEqual({"145A", "189D"}, result)

    def test_pipe_joined_columns(self):
        """Should find position in pipe-joined compound name"""
        columns = ["145K|155S", "189D"]
        result = columns_at_positions(columns, [155])
        self.assertEqual({"145K|155S"}, result)

    def test_tilde_joined_columns(self):
        """Should find position in tilde-joined collinear group"""
        columns = ["145A~189D", "156K"]
        result = columns_at_positions(columns, [189])
        self.assertEqual({"145A~189D"}, result)

    def test_complement_columns(self):
        """Should find position in complement column name (with -)"""
        columns = ["135N|-145A", "156K"]
        result = columns_at_positions(columns, [145])
        self.assertEqual({"135N|-145A"}, result)

    def test_no_match(self):
        """Should return empty set when no columns match"""
        columns = ["145A", "156K", "189D"]
        result = columns_at_positions(columns, [999])
        self.assertEqual(set(), result)

    def test_compound_pipe_tilde_name(self):
        """Should find position in compound |+~ name"""
        columns = ["145K|155S~189K", "156N"]
        result = columns_at_positions(columns, [189])
        self.assertEqual({"145K|155S~189K"}, result)


if __name__ == "__main__":
    unittest.main()
