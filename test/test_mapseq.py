#!/usr/bin/env python

"""Tests for MapSeq class"""

import gzip
import json
import os
import shutil
import tempfile
import unittest

import matplotlib.pyplot as plt
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

    def test_unknown_sequence(self):
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

    def test_error_message_includes_position(self):
        """Error message should include the unknown position number"""
        with self.assertRaises(ValueError) as ctx:
            self.ms.strains_with_combinations({999: "K"}, verbose=False)
        # Check that the position number appears in the error message
        self.assertIn("999", str(ctx.exception))


class MapSeqDuplicateSequences(unittest.TestCase):
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


class MapSeqDiskIO(unittest.TestCase):
    """Tests for MapSeq.to_disk and MapSeq.from_disk"""

    def setUp(self):
        """Sequences and coordinates to use in tests"""
        self.seq_df = pd.DataFrame(
            {
                1: ("Q", "Q", "Q"),
                2: ("K", "K", "N"),
                3: ("L", "P", "A"),
            },
            index=("strain1", "strain2", "strain3"),
        )
        self.coord_df = pd.DataFrame(
            {
                "x": (0.0, 0.5, 1.0),
                "y": (0.0, 1.0, 0.5),
            },
            index=("strain1", "strain2", "strain3"),
        )
        self.tmpdir = None

    def tearDown(self):
        """Clean up temp files"""
        if self.tmpdir and self.tmpdir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_tmpfile(self):
        """Create a temp file path"""
        self.tmpdir = tempfile.mkdtemp()
        return f"{self.tmpdir}/mapseq.json"

    def test_to_disk_creates_file(self):
        """to_disk should create a file at the specified path"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()

        ms.to_disk(path)

        self.assertTrue(os.path.exists(path))

    def test_to_disk_creates_valid_json(self):
        """to_disk should create valid JSON"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()

        ms.to_disk(path)

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        self.assertIsInstance(data, dict)

    def test_to_disk_contains_version(self):
        """to_disk output should contain a version field"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()

        ms.to_disk(path)

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        self.assertIn("version", data)
        self.assertEqual(data["version"], 1)

    def test_to_disk_contains_seq_df(self):
        """to_disk output should contain seq_df with split orient structure"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()

        ms.to_disk(path)

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        self.assertIn("seq_df", data)
        self.assertIn("columns", data["seq_df"])
        self.assertIn("index", data["seq_df"])
        self.assertIn("data", data["seq_df"])

    def test_to_disk_contains_coord_df(self):
        """
        to_disk output should contain coord_df with split orient structure
        """
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()

        ms.to_disk(path)

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        self.assertIn("coord_df", data)
        self.assertIn("columns", data["coord_df"])
        self.assertIn("index", data["coord_df"])
        self.assertIn("data", data["coord_df"])

    def test_to_disk_contains_map_when_set(self):
        """to_disk output should contain map field when set"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df, map=2017)
        path = self._get_tmpfile()

        ms.to_disk(path)

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        self.assertIn("map", data)
        self.assertEqual(data["map"], 2017)

    def test_to_disk_map_null_when_not_set(self):
        """to_disk output should have map as null when not set"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()

        ms.to_disk(path)

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        self.assertIn("map", data)
        self.assertIsNone(data["map"])

    def test_from_disk_returns_mapseq(self):
        """from_disk should return a MapSeq instance"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()
        ms.to_disk(path)

        loaded = MapSeq.from_disk(path)

        self.assertIsInstance(loaded, MapSeq)

    def test_roundtrip_preserves_seq_data(self):
        """Round-trip should preserve sequence data"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()
        ms.to_disk(path)

        loaded = MapSeq.from_disk(path)

        pd.testing.assert_frame_equal(
            ms.all_seqs.sort_index(),
            loaded.all_seqs.sort_index(),
        )

    def test_roundtrip_preserves_coord_data(self):
        """Round-trip should preserve coordinate data"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()
        ms.to_disk(path)

        loaded = MapSeq.from_disk(path)

        pd.testing.assert_frame_equal(
            ms.all_coords.sort_index(),
            loaded.all_coords.sort_index(),
        )

    def test_roundtrip_preserves_map_metadata(self):
        """Round-trip should preserve map metadata"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df, map=2009)
        path = self._get_tmpfile()
        ms.to_disk(path)

        loaded = MapSeq.from_disk(path)

        self.assertEqual(ms.map, loaded.map)

    def test_roundtrip_preserves_none_map(self):
        """Round-trip should preserve None map metadata"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df, map=None)
        path = self._get_tmpfile()
        ms.to_disk(path)

        loaded = MapSeq.from_disk(path)

        self.assertIsNone(loaded.map)

    def test_to_disk_creates_compressed_file(self):
        """to_disk should create a gzip-compressed file"""
        ms = MapSeq(seq_df=self.seq_df, coord_df=self.coord_df)
        path = self._get_tmpfile()

        ms.to_disk(path)

        with gzip.open(path, "rt") as f:
            data = json.load(f)

        self.assertIsInstance(data, dict)


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


def _make_oms(seq_data, positions, strain_names):
    """Helper to build an OrderedMapSeq from compact sequence data."""
    seq_df = pd.DataFrame(seq_data, columns=positions, index=strain_names)
    coord_df = pd.DataFrame(
        {
            "x": np.arange(len(strain_names), dtype=float),
            "y": np.arange(len(strain_names), dtype=float),
        },
        index=strain_names,
    )
    return OrderedMapSeq(seq_df=seq_df, coord_df=coord_df)


class OrderedMapSeqFilterMergeDuplicateDummies(unittest.TestCase):
    """merge_duplicate_dummies should concatenate identical-profile dummy
    names with '|'."""

    def setUp(self):
        """Six strains, three variant positions.

        Positions 200 and 300 have identical amino-acid distributions
        (K<->D, N<->E, A<->F) so after get_dummies each pair of
        cross-position dummies shares the same binary profile.
        Position 400 is independent (alternating S/T).

        Using three amino acids per position avoids accidental
        complement merging across positions.
        """
        strains = [f"s{i}" for i in range(1, 7)]
        #               pos200 pos300 pos400
        seqs = [
            ["K", "D", "S"],
            ["K", "D", "T"],
            ["K", "D", "S"],
            ["N", "E", "T"],
            ["N", "E", "S"],
            ["A", "F", "T"],
        ]
        self.oms = _make_oms(seqs, [200, 300, 400], strains)

    def test_identical_dummies_merged_with_pipe(self):
        """Dummies with identical profiles are merged using '|'."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
        )
        cols = set(self.oms.seqs.dummies.columns)

        # 200K and 300D have the same profile -> merged
        self.assertTrue(
            any("200K" in c and "300D" in c and "|" in c for c in cols),
            f"Expected 200K|300D merge, got columns: {cols}",
        )

    def test_all_identical_pairs_merged(self):
        """Each pair of cross-position identical dummies is merged."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
        )
        cols = set(self.oms.seqs.dummies.columns)

        # 200N and 300E share a profile
        self.assertTrue(
            any("200N" in c and "300E" in c for c in cols),
            f"Expected 200N/300E merge, got columns: {cols}",
        )

        # 200A and 300F share a profile
        self.assertTrue(
            any("200A" in c and "300F" in c for c in cols),
            f"Expected 200A/300F merge, got columns: {cols}",
        )

    def test_column_count_reduced(self):
        """Merging should reduce the number of dummy columns.

        Without merging there are 9 dummies (3 amino acids x 3
        positions).  Merging identical profiles collapses that.
        """
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
        )
        n_cols = self.oms.seqs.dummies.shape[1]
        self.assertLess(n_cols, 9)

    def test_independent_dummy_not_merged_cross_position(self):
        """Dummies at position 400 (independent) should not merge with
        any position-200 or position-300 dummy."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
        )
        cols = set(self.oms.seqs.dummies.columns)

        col_with_400 = [c for c in cols if "400" in c]
        self.assertEqual(len(col_with_400), 1)
        for c in col_with_400:
            self.assertNotIn("200", c)
            self.assertNotIn("300", c)

    def test_values_preserved(self):
        """After merging, the dummy values for a merged column should
        be 1 for strains that had K at position 200 (equivalently D
        at 300) and 0 otherwise (or the complement of that)."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
        )
        merged_col = next(
            c
            for c in self.oms.seqs.dummies.columns
            if "200K" in c and "300D" in c
        )
        series = self.oms.seqs.dummies[merged_col]

        # s1, s2, s3 have K at 200 (and D at 300) -> same value
        # s4, s5 have N at 200 -> different value
        # s6 has A at 200 -> different value
        k_strains = {"s1", "s2", "s3"}
        non_k_strains = {"s4", "s5", "s6"}
        k_vals = set(series.loc[series.index.isin(k_strains)].values)
        non_k_vals = set(series.loc[series.index.isin(non_k_strains)].values)

        # All K-strains should share one value and all non-K-strains
        # should share the other.
        self.assertEqual(len(k_vals), 1)
        self.assertEqual(len(non_k_vals), 1)
        self.assertNotEqual(k_vals, non_k_vals)


class OrderedMapSeqFilterPruneCollinearDummies(unittest.TestCase):
    """prune_collinear_dummies should concatenate near-collinear dummy
    names with '~'."""

    def setUp(self):
        """Eight strains, two variant positions.

        Position 200 has K (strains 1-6) or N (strains 7-8).
        Position 300 has D (strains 1-5) or E (strains 6-8).

        After get_dummies the within-position complements (e.g. 200K
        and 200N) have r^2 = 1, while the cross-position pair
        (200K vs 300D) has r^2 ~ 0.56.
        """
        strains = [f"s{i}" for i in range(1, 9)]
        #               pos200 pos300
        seqs = [
            ["K", "D"],
            ["K", "D"],
            ["K", "D"],
            ["K", "D"],
            ["K", "D"],
            ["K", "E"],
            ["N", "E"],
            ["N", "E"],
        ]
        self.oms = _make_oms(seqs, [200, 300], strains)

    def test_collinear_dummies_merged_with_tilde(self):
        """Dummies whose r^2 exceeds the threshold are joined with '~'."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=False,
            prune_collinear_dummies=0.5,
        )
        cols = list(self.oms.seqs.dummies.columns)

        self.assertTrue(
            any("~" in c for c in cols),
            f"Expected '~' in column names, got: {cols}",
        )

    def test_high_threshold_no_cross_position_merge(self):
        """With threshold=0.99 the cross-position pair (r^2 ~ 0.56) is
        not pruned, but within-position complements (r^2 = 1) are."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=False,
            prune_collinear_dummies=0.99,
        )
        cols = list(self.oms.seqs.dummies.columns)

        # 2 retained columns (one per position), each with a ~-joined
        # complement.
        self.assertEqual(len(cols), 2)
        for c in cols:
            self.assertIn("~", c)

        # No column should span both positions.
        for c in cols:
            self.assertFalse(
                "200" in c and "300" in c,
                f"Column '{c}' unexpectedly spans both positions",
            )

    def test_low_threshold_all_collapse(self):
        """With a very low threshold all dummies collapse into one
        column."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=False,
            prune_collinear_dummies=0.1,
        )
        cols = list(self.oms.seqs.dummies.columns)
        self.assertEqual(len(cols), 1)
        self.assertIn("~", cols[0])

    def test_column_count_reduced(self):
        """Pruning should reduce the number of columns compared to
        no pruning."""
        oms_no_prune = _make_oms(
            [
                ["K", "D"],
                ["K", "D"],
                ["K", "D"],
                ["K", "D"],
                ["K", "D"],
                ["K", "E"],
                ["N", "E"],
                ["N", "E"],
            ],
            [200, 300],
            [f"s{i}" for i in range(1, 9)],
        )
        oms_no_prune.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=False,
            prune_collinear_dummies=None,
        )
        n_no_prune = oms_no_prune.seqs.dummies.shape[1]

        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=False,
            prune_collinear_dummies=0.5,
        )
        n_pruned = self.oms.seqs.dummies.shape[1]

        self.assertLess(n_pruned, n_no_prune)


class OrderedMapSeqFilterMergeThenPrune(unittest.TestCase):
    """When both merge_duplicate_dummies and prune_collinear_dummies
    are enabled, merge runs first (creating '|' groups), then prune
    runs on the merged result (creating '~' groups).  The final column
    names can therefore contain both '|' and '~'."""

    def setUp(self):
        """Eight strains, four variant positions.

        Positions 200/300 share identical amino-acid distributions
        (K<->D, N<->E) so their dummies merge with '|'.

        Positions 400/500 likewise share identical distributions
        (S<->L, T<->M) and also merge with '|'.

        The resulting merged groups (200/300 block vs 400/500 block)
        are correlated (r^2 ~ 0.56), so with threshold=0.5 they are
        further joined with '~'.
        """
        strains = [f"s{i}" for i in range(1, 9)]
        #               200  300  400  500
        seqs = [
            ["K", "D", "S", "L"],
            ["K", "D", "S", "L"],
            ["K", "D", "S", "L"],
            ["K", "D", "S", "L"],
            ["K", "D", "S", "L"],
            ["K", "D", "T", "M"],
            ["N", "E", "T", "M"],
            ["N", "E", "T", "M"],
        ]
        self.oms = _make_oms(seqs, [200, 300, 400, 500], strains)

    def test_merge_before_prune(self):
        """A '~'-joined column name should contain '|'-joined groups,
        proving that '|' merge ran first and '~' prune ran second."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
            prune_collinear_dummies=0.5,
        )
        cols = list(self.oms.seqs.dummies.columns)

        has_both = any("|" in c and "~" in c for c in cols)
        self.assertTrue(
            has_both,
            f"Expected a column with both '|' and '~', got: {cols}",
        )

    def test_pipe_groups_inside_tilde_groups(self):
        """Within a '~'-joined name, each segment between '~' should
        be a '|'-joined group (or a single dummy name)."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
            prune_collinear_dummies=0.5,
        )
        cols = list(self.oms.seqs.dummies.columns)

        for col in cols:
            if "~" in col:
                tilde_groups = col.split("~")
                for group in tilde_groups:
                    for name in group.split("|"):
                        stripped = name.lstrip("-")
                        self.assertRegex(
                            stripped,
                            r"^\d+[A-Z]$",
                            f"Invalid SNP name '{name}' in column " f"'{col}'",
                        )

    def test_identical_dummies_in_same_pipe_group(self):
        """200K and 300D (identical profiles) should appear in the same
        '|'-delimited segment, not separated by '~'."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
            prune_collinear_dummies=0.5,
        )
        cols = list(self.oms.seqs.dummies.columns)
        col_200K = next(c for c in cols if "200K" in c)

        for tilde_group in col_200K.split("~"):
            parts = tilde_group.split("|")
            if "200K" in parts:
                self.assertIn(
                    "300D",
                    parts,
                    f"200K and 300D should be in the same '|' group, "
                    f"got column '{col_200K}'",
                )

    def test_collinear_groups_joined_by_tilde(self):
        """The 200/300 merged group and the 400/500 merged group
        should be joined by '~' (since r^2 > 0.5)."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
            prune_collinear_dummies=0.5,
        )
        cols = list(self.oms.seqs.dummies.columns)
        col_200K = next(c for c in cols if "200K" in c)

        self.assertTrue(
            "400" in col_200K or "500" in col_200K,
            f"Expected 200/300 group to be '~'-joined with 400/500 "
            f"group, got column '{col_200K}'",
        )

    def test_high_threshold_no_tilde_across_blocks(self):
        """With threshold=0.99, the two '|' groups are not collinear
        enough to merge, so no column spans both blocks."""
        self.oms.filter(
            plot=False,
            get_dummies=True,
            merge_duplicate_dummies=True,
            prune_collinear_dummies=0.99,
        )
        cols = list(self.oms.seqs.dummies.columns)

        for c in cols:
            has_200_or_300 = "200" in c or "300" in c
            has_400_or_500 = "400" in c or "500" in c
            self.assertFalse(
                has_200_or_300 and has_400_or_500,
                f"Column '{c}' unexpectedly merges the two blocks",
            )


class ScatterColoredByAminoAcidRandomZ(unittest.TestCase):
    """Tests for scatter_colored_by_amino_acid with randomz=True."""

    def setUp(self):
        seq_df = pd.DataFrame(
            {
                1: ("Q", "Q", "Q", "Q", "Q", "Q"),
                2: ("K", "K", "N", "K", "N", "K"),
            },
            index=(
                "strain1",
                "strain2",
                "strain3",
                "strain4",
                "strain5",
                "strain6",
            ),
        )
        coord_df = pd.DataFrame(
            {
                "x": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                "y": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            },
            index=(
                "strain1",
                "strain2",
                "strain3",
                "strain4",
                "strain5",
                "strain6",
            ),
        )
        self.ms = MapSeq(seq_df=seq_df, coord_df=coord_df)

    def tearDown(self):
        plt.close("all")

    def test_all_points_plotted(self):
        """All strains should be plotted when randomz=True."""
        self.ms.scatter_colored_by_amino_acid(
            p=2, randomz=True, ellipses=False
        )
        ax = plt.gca()
        # Count total number of points across all PathCollections
        n_points = sum(len(c.get_offsets()) for c in ax.collections)
        # 6 strains in both seq and coord
        self.assertEqual(n_points, 6)

    def test_legend_contains_all_amino_acids(self):
        """Legend should have entries for each amino acid at position 2."""
        self.ms.scatter_colored_by_amino_acid(
            p=2, randomz=True, ellipses=False
        )
        ax = plt.gca()
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        amino_acids_in_legend = {t.split()[0] for t in legend_texts}
        self.assertIn("K", amino_acids_in_legend)
        self.assertIn("N", amino_acids_in_legend)

    def test_fewer_scatter_calls_than_points(self):
        """With the optimized approach, there should be far fewer
        PathCollections than individual points."""
        self.ms.scatter_colored_by_amino_acid(
            p=2, randomz=True, ellipses=False
        )
        ax = plt.gca()
        # Old approach: one PathCollection per point (6 + extra for labels)
        # New approach: 1 PathCollection for all points + legend handles
        n_collections = len(ax.collections)
        # Should be much less than the number of points (6)
        self.assertLess(n_collections, 6)


if __name__ == "__main__":
    unittest.main()
