#!/usr/bin/env python

"""Tests for permp function that computes empirical p-values"""

import unittest
import warnings

import numpy as np
from rpy2.rinterface_lib.embedded import RRuntimeError

from mapdeduce.permp import permp

# Permp.test_too_many_combinations_for_exact raises this warning
warnings.filterwarnings(
    action="ignore",
    module="rpy2",
    message="Error in 1:total.nperm : result would be too " "long a vector",
)


class Permp(unittest.TestCase):
    """Tests for permp function"""

    def test_expected_result1_(self):
        """Compare python result to R result"""
        p = permp(1, 1000, 3, 997)[0]
        self.assertTrue(np.abs(p - 0.001997999) < 1e-10)

    def test_too_many_combinations_for_exact(self):
        """When there are too many permutations, expect an RRuntimeError"""
        with self.assertRaises(RRuntimeError):
            permp(1, 1000, 500, 500, method="exact")

    def test_approximate_method(self):
        """
        When there are too many permutations for exact, approximate should not
        raise an error.
        """
        permp(1, 1000, 500, 500, method="approximate")

    def test_returns_np_array(self):
        """Should return an np array"""
        p = permp(1, 1000, 3, 997)
        self.assertIsInstance(p, np.ndarray)


if __name__ == "__main__":
    unittest.main()
