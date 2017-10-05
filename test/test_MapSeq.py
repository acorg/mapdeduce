#!/usr/bin/env python

"""Tests for MapSeq class"""

import unittest2 as unittest
import pandas as pd

from MapSeq.MapSeq import MapSeq


class MapSeqTests(unittest.TestCase):
    """Tests for MapSeq class"""

    def setUp(self):
        """Basic sequences and coordinates to use in tests"""
        sequence_df = pd.DataFrame({
            1: ('Q', 'Q', 'Q', 'Q'),
            2: ('K', 'K', 'N', 'K'),
            3: ('L', 'P', 'A', 'L'),
        }, index=('strain1', 'strain2', 'strain3', 'strain5'))
        coordinate_df = pd.DataFrame({
            'x': (0, 0, 1, 1),
            'y': (0, 1, 0, 1),
        }, index=('strain1', 'strain2', 'strain3', 'strain4'))
        self.ms = MapSeq(sequence_df=sequence_df, coordinate_df=coordinate_df)

    def test_common_strains(self):
        """
        MapSeq common_strain attribute should be a set comprising the
        intersection of the strains in the sequence and coordinate dfs
        """
        expect = {'strain1', 'strain2', 'strain3'}
        self.assertEqual(expect, self.ms.common_strains)

    def test_main_df_indexes(self):
        """Indexes of the main df should match common_strains"""
        self.assertEqual(self.ms.common_strains, set(self.ms.df.index))
