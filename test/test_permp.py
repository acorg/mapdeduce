#!/usr/bin/env python

"""Tests for permp function that computes empirical p-values"""

from MapDeduce.permp import permp
import numpy as np

try:
    import unittest2 as unittest
except ImportError:
    import unittest
from rpy2.rinterface import RRuntimeError


class Permp(unittest.TestCase):
    """Tests for permp function"""

    def test_expected_result1_(self):
        """Compare python result to R result"""
        p = permp(1, 1000, 3, 997)[0]
        self.assertTrue(np.abs(p - 0.001997999) < 1e-10)

    def test_too_many_combinations_for_exact(self):
        """When there are too many permutations, expec an RRuntimeError"""
        with self.assertRaises(RRuntimeError):
            permp(1, 1000, 500, 500, method="exact")

    def test_approximate_method(self):
        """
        When there are too many permutations for exact, approximate shold not
        raise an error.
        """
        permp(1, 1000, 500, 500, method="approximate")

    def test_returns_np_array(self):
        """Should return an np array"""
        p = permp(1, 1000, 3, 997)
        self.assertIsInstance(p, np.ndarray)


if __name__ == "__main__":
    unittest.main()
