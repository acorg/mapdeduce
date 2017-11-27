#!/usr/bin/env python

"""Tests for data"""

import unittest2 as unittest

import numpy as np
import pandas as pd

import MapSeq
from MapSeq.hwas import HwasLmm


class HwasLmmCrossValidation(unittest.TestCase):
    """Tests for HwasLmm.cross_validate"""

    def test_list_returned(self):
        """Should return a list containing folds"""
        np.random.seed(1234)
        N, S, P = 4, 3, 2
        snps = pd.DataFrame(np.random.randint(2, size=N * S).reshape((N, S)))
        pheno = pd.DataFrame(np.random.randn(N, P))
        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.cross_validate(n_splits=2)
        self.assertIsInstance(hwas.folds, list)

    def test_fold_type(self):
        """Items in list should be tuples"""
        np.random.seed(1234)
        N, S, P = 4, 3, 2
        snps = pd.DataFrame(np.random.randint(2, size=N * S).reshape((N, S)))
        pheno = pd.DataFrame(np.random.randn(N, P))
        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.cross_validate(n_splits=2)
        fold = hwas.folds[0]
        self.assertIsInstance(fold, tuple)

    def test_items_in_list(self):
        """Test types of items in each fold"""
        np.random.seed(1234)
        N, S, P = 4, 3, 2
        snps = pd.DataFrame(np.random.randint(2, size=N * S).reshape((N, S)))
        pheno = pd.DataFrame(np.random.randn(N, P))
        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.cross_validate(n_splits=2)
        fold = hwas.folds[0]
        self.assertIsInstance(fold[0], MapSeq.hwas.HwasLmm)
        self.assertIsInstance(fold[1], pd.core.frame.DataFrame)
        self.assertIsInstance(fold[2], pd.core.frame.DataFrame)

    def test_phenotype_offset(self):
        """Test phenotype offset does not"""
        np.random.seed(1234)
        N, S, P = 4, 3, 2
        snps = pd.DataFrame(np.random.randint(2, size=N * S).reshape((N, S)))
        pheno = pd.DataFrame(np.random.randn(N, P) + 100)
        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.cross_validate(n_splits=2)
        # Include all snps
        pn = hwas.cross_validation_predictions(p_grid=[1, ])

        # Predictions by chance should be small (pheno drawn from normal dist,
        # with scale = 1, loc = 0.
        # If pheno offset not handled correctly mean error would be ~100
        self.assertLess(pn.mean().mean()[1], 5)

if __name__ == "__main__":

    unittest.main()
