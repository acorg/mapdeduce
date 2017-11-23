#!/usr/bin/env python

"""Tests for data"""

import unittest2 as unittest

from MapSeq.data import not_109_to_301, not_110_to_199


class MapSeqData(unittest.TestCase):
    """Tests for the data"""

    def test_correct_positions_not_in_109_to_301(self):
        """Test the correct positions are in this list"""
        self.assertIn(108, not_109_to_301)
        self.assertNotIn(109, not_109_to_301)
        self.assertIn(302, not_109_to_301)
        self.assertNotIn(301, not_109_to_301)
        self.assertIn(328, not_109_to_301)
        self.assertNotIn(329, not_109_to_301)
        self.assertIn(1, not_109_to_301)
        self.assertNotIn(0, not_109_to_301)

    def test_correct_positions_not_in_110_to_199(self):
        """Test correct positions are in this list."""
        self.assertIn(109, not_110_to_199)
        self.assertNotIn(110, not_110_to_199)
        self.assertIn(200, not_110_to_199)
        self.assertNotIn(199, not_110_to_199)
        self.assertIn(328, not_110_to_199)
        self.assertNotIn(329, not_110_to_199)
        self.assertIn(1, not_110_to_199)
        self.assertNotIn(0, not_110_to_199)

if __name__ == "__main__":

    unittest.main()
