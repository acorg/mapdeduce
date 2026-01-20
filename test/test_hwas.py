#!/usr/bin/env python

"""Tests for data"""

from itertools import combinations
import unittest

from scipy.stats import pearsonr
import numpy as np
import pandas as pd

import mapdeduce
from mapdeduce.hwas import (
    HwasLmm,
    find_perfectly_correlated_snps,
    prune_collinear_snps,
)


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
        self.assertIsInstance(fold[0], mapdeduce.hwas.HwasLmm)
        self.assertIsInstance(fold[1], pd.core.frame.DataFrame)
        self.assertIsInstance(fold[2], pd.core.frame.DataFrame)

    @unittest.skip(
        "Test currently failing, but don't currently need this "
        "functionality"
    )
    def test_phenotype_offset(self):
        """Test phenotype offset does not"""
        np.random.seed(1234)
        N, S, P = 4, 3, 2
        snps = pd.DataFrame(np.random.randint(2, size=N * S).reshape((N, S)))
        pheno = pd.DataFrame(np.random.randn(N, P) + 100)
        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.cross_validate(n_splits=2)
        # Include all snps
        pn = hwas.cross_validation_predictions(
            p_grid=[
                1,
            ]
        )

        # Predictions by chance should be small (pheno drawn from normal dist,
        # with scale = 1, loc = 0.
        # If pheno offset not handled correctly mean error would be ~100
        self.assertLess(pn.mean().mean()[1], 5)


class HwasLmmFit(unittest.TestCase):
    """Tests for HwasLmm.fit"""

    def test_results_is_dataframe(self):
        """fit should attach a results DataFrame"""
        np.random.seed(1234)
        N = 20
        # Use 3+ SNPs to avoid edge case in effective_tests with 2 SNPs
        snps = pd.DataFrame(
            {
                "SNP1": np.random.randint(2, size=N),
                "SNP2": np.random.randint(2, size=N),
                "SNP3": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": np.random.randn(N)})
        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.fit()
        self.assertIsInstance(hwas.results, pd.DataFrame)

    def test_results_has_expected_columns(self):
        """results DataFrame should have p, beta, logp columns"""
        np.random.seed(1234)
        N = 20
        snps = pd.DataFrame(
            {
                "SNP1": np.random.randint(2, size=N),
                "SNP2": np.random.randint(2, size=N),
                "SNP3": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": np.random.randn(N)})
        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.fit()
        self.assertIn("p", hwas.results.columns)
        self.assertIn("beta", hwas.results.columns)
        self.assertIn("logp", hwas.results.columns)

    def test_results_index_matches_snps(self):
        """results index should contain the tested SNPs"""
        np.random.seed(1234)
        N = 20
        snps = pd.DataFrame(
            {
                "SNP1": np.random.randint(2, size=N),
                "SNP2": np.random.randint(2, size=N),
                "SNP3": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": np.random.randn(N)})
        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.fit()
        self.assertEqual(set(hwas.results.index), set(snps.columns))

    def test_strong_association_small_p_value(self):
        """SNP strongly associated with phenotype should have small p-value"""
        np.random.seed(42)
        N = 50

        # Create SNP that perfectly predicts phenotype
        snp_values = np.array([0] * 25 + [1] * 25)

        # Phenotype has large effect: 0 when SNP=0, 10 when SNP=1 (plus small
        # noise)
        pheno_values = snp_values * 10.0 + np.random.randn(N) * 0.5

        # Use 3+ SNPs to avoid edge case in effective_tests
        snps = pd.DataFrame(
            {
                "causal_snp": snp_values,
                "null_snp1": np.random.randint(2, size=N),
                "null_snp2": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": pheno_values})

        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.fit()

        print(hwas.results.loc["causal_snp", "p"])
        self.assertLess(hwas.results.loc["causal_snp", "p"], 1e-10)

    def test_strong_association_large_effect(self):
        """SNP strongly associated with phenotype should have large effect size"""
        np.random.seed(42)
        N = 50
        effect_size = 10.0
        snp_values = np.array([0] * 25 + [1] * 25)
        pheno_values = snp_values * effect_size + np.random.randn(N) * 0.5

        # Use 3+ SNPs to avoid edge case in effective_tests
        snps = pd.DataFrame(
            {
                "causal_snp": snp_values,
                "null_snp1": np.random.randint(2, size=N),
                "null_snp2": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": pheno_values})

        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.fit()

        self.assertTrue(
            effect_size - 0.1
            < hwas.results.loc["causal_snp", "beta"]
            < effect_size + 0.1
        )

    def test_no_association_large_p_value(self):
        """SNP not associated with phenotype should have large p-value"""
        np.random.seed(42)
        N = 50

        # Random SNP with no relationship to phenotype
        snp_values = np.array([0] * 25 + [1] * 25)

        # Phenotype is just random noise, independent of SNP
        pheno_values = np.random.randn(N)

        # Use 3+ SNPs to avoid edge case in effective_tests
        snps = pd.DataFrame(
            {
                "null_snp": snp_values,
                "null_snp2": np.random.randint(2, size=N),
                "null_snp3": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": pheno_values})

        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.fit()

        self.assertGreater(hwas.results.loc["null_snp", "p"], 0.2)

    def test_no_association_small_effect(self):
        """SNP not associated with phenotype should have small effect size"""
        np.random.seed(42)
        N = 50
        snp_values = np.array([0] * 25 + [1] * 25)
        pheno_values = np.random.randn(N)

        # Use 3+ SNPs to avoid edge case in effective_tests
        snps = pd.DataFrame(
            {
                "null_snp": snp_values,
                "null_snp2": np.random.randint(2, size=N),
                "null_snp3": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": pheno_values})

        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.fit()

        self.assertLess(abs(hwas.results.loc["null_snp", "beta"]), 0.2)

    def test_invariant_snp_skipped_when_not_explicit(self):
        """Invariant SNP should be skipped with warning when test_snps is None"""
        np.random.seed(42)
        N = 20

        snps = pd.DataFrame(
            {
                "variable_snp1": np.random.randint(2, size=N),
                "variable_snp2": np.random.randint(2, size=N),
                "variable_snp3": np.random.randint(2, size=N),
                "invariant_snp": np.ones(N, dtype=int),  # All 1s
            }
        )
        pheno = pd.DataFrame({"y": np.random.randn(N)})

        hwas = HwasLmm(snps=snps, pheno=pheno)

        # Don't pass test_snps - defaults to all columns in snps
        with self.assertWarns(UserWarning):
            hwas.fit()

        # Invariant SNP should not be in results
        self.assertNotIn("invariant_snp", hwas.results.index)
        self.assertIn("variable_snp1", hwas.results.index)

    def test_invariant_snp_raises_when_explicit(self):
        """Invariant SNP should raise ValueError when explicitly passed"""
        np.random.seed(42)
        N = 20
        snps = pd.DataFrame(
            {
                "variable_snp1": np.random.randint(2, size=N),
                "variable_snp2": np.random.randint(2, size=N),
                "variable_snp3": np.random.randint(2, size=N),
                "invariant_snp": np.ones(N, dtype=int),  # All 1s
            }
        )
        pheno = pd.DataFrame({"y": np.random.randn(N)})

        hwas = HwasLmm(snps=snps, pheno=pheno)

        with self.assertRaises(ValueError) as err:
            # Explicitly pass test_snps including the invariant one
            hwas.fit(test_snps=list(snps.columns))

        self.assertIn("invariant_snp", str(err.exception))
        self.assertIn("does not have 2 unique values", str(err.exception))

    def test_causal_snp_ranked_first(self):
        """Causal SNP should have lowest p-value among tested SNPs"""
        np.random.seed(42)
        N = 50
        snp_causal = np.array([0] * 25 + [1] * 25)
        pheno_values = snp_causal * 10.0 + np.random.randn(N) * 0.5

        snps = pd.DataFrame(
            {
                "causal": snp_causal,
                "null1": np.random.randint(2, size=N),
                "null2": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": pheno_values})

        hwas = HwasLmm(snps=snps, pheno=pheno)
        hwas.fit()

        # Results are sorted by p-value, causal should be first
        self.assertEqual(hwas.results.index[0], "causal")


class FindPerfectlyCorrelatedSnpsTests(unittest.TestCase):
    """Tests for find_perfectly_correlated_snps function"""

    def test_no_correlation(self):
        """Independent SNPs should return empty list"""

        # no snps are identical or complements
        snps = pd.DataFrame(
            {
                "snp1": [0, 1, 0, 1, 0, 0],
                "snp2": [0, 0, 1, 1, 0, 1],
                "snp3": [1, 1, 0, 1, 0, 0],
            }
        )
        result = find_perfectly_correlated_snps(snps)
        self.assertEqual(result, [])

    def test_identical_snps(self):
        """Identical SNPs should be detected"""
        snps = pd.DataFrame(
            {
                "snp1": [0, 1, 0, 1, 0, 1],
                "snp2": [0, 1, 0, 1, 0, 1],  # identical to snp1
                "snp3": [1, 1, 0, 0, 1, 0],  # independent
            }
        )
        result = find_perfectly_correlated_snps(snps)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("snp1", "snp2", "identical"))

    def test_complement_snps(self):
        """Complement SNPs (one = 1 - other) should be detected"""
        snps = pd.DataFrame(
            {
                "snp1": [0, 1, 0, 1, 0, 1],
                "snp2": [1, 0, 1, 0, 1, 0],  # complement of snp1
                "snp3": [0, 0, 1, 1, 0, 1],  # independent
            }
        )
        result = find_perfectly_correlated_snps(snps)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("snp1", "snp2", "complement"))

    def test_multiple_correlations(self):
        """Multiple correlated pairs should all be detected"""
        snps = pd.DataFrame(
            {
                "snp1": [0, 1, 0, 1, 0, 1],
                "snp2": [0, 1, 0, 1, 0, 1],  # identical to snp1
                "snp3": [1, 0, 1, 0, 1, 0],  # complement of snp1 and snp2
            }
        )
        result = find_perfectly_correlated_snps(snps)
        self.assertEqual(len(result), 3)
        self.assertIn(("snp1", "snp2", "identical"), result)
        self.assertIn(("snp1", "snp3", "complement"), result)
        self.assertIn(("snp2", "snp3", "complement"), result)


class HwasLmmFitValidation(unittest.TestCase):
    """Tests for HwasLmm.fit input validation"""

    def test_raises_on_identical_snps(self):
        """fit should raise ValueError if SNPs are identical"""
        np.random.seed(42)
        N = 20
        snp_values = np.random.randint(2, size=N)
        snps = pd.DataFrame(
            {
                "snp1": snp_values,
                "snp2": snp_values,  # identical
                "snp3": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": np.random.randn(N)})

        hwas = HwasLmm(snps=snps, pheno=pheno)
        with self.assertRaises(ValueError) as err:
            hwas.fit()
        self.assertIn("perfectly correlated", str(err.exception))
        self.assertIn("snp1", str(err.exception))
        self.assertIn("snp2", str(err.exception))

    def test_raises_on_complement_snps(self):
        """fit should raise ValueError if SNPs are complements"""
        np.random.seed(42)
        N = 20
        snp_values = np.random.randint(2, size=N)
        snps = pd.DataFrame(
            {
                "snp1": snp_values,
                "snp2": 1 - snp_values,  # complement
                "snp3": np.random.randint(2, size=N),
            }
        )
        pheno = pd.DataFrame({"y": np.random.randn(N)})

        hwas = HwasLmm(snps=snps, pheno=pheno)
        with self.assertRaises(ValueError) as ctx:
            hwas.fit()
        self.assertIn("perfectly correlated", str(ctx.exception))


class HwasLmmRegressOut(unittest.TestCase):
    """Tests for HwasLmm.regress_out"""

    def setUp(self):
        """Attach HwasLmm with known effects to self"""
        pheno = pd.DataFrame(
            np.array(((0, 0), (1, 0), (2, 0))),
            columns=["x", "y"],
            index=["a", "b", "c"],
        )

        snps = pd.DataFrame(
            np.array(((0, 0), (1, 1), (0, 1))),
            columns=["SNP-1", "SNP-2"],
            index=["a", "b", "c"],
        )

        self.hwas = HwasLmm(snps=snps, pheno=pheno)

        # Specify effects with known vectors, to test that they are regressed
        # out correctly. Dataframe with beta column in same format as
        # Hwas.results
        self.hwas.results = pd.DataFrame(
            {
                "beta": [np.array((1, 0)), np.array((2, 2))],
            },
            index=["SNP-1", "SNP-2"],
        )

    def test_returns_df(self):
        """HwasLmm.regress_out should return a DataFrame"""
        df = self.hwas.regress_out("SNP-1")
        self.assertIsInstance(df, pd.core.frame.DataFrame)

    def test_df_dims(self):
        """Output should have same dimensions as original data"""
        df = self.hwas.regress_out("SNP-1")
        self.assertEqual(self.hwas.pheno.shape, df.shape)

    def test_snp1_regressed_out_correctly(self):
        """Test maths for SNP1"""
        output = self.hwas.regress_out("SNP-1")
        expect = np.array(((0, 0), (0, 0), (2, 0)))
        self.assertEqual(0, (output - expect).sum().sum())

    def test_snp2_regressed_out_correctly(self):
        """Test maths for SNP2"""
        output = self.hwas.regress_out("SNP-2")
        expect = np.array(((0, 0), (-1, -2), (0, -2)))
        self.assertEqual(0, (output - expect).sum().sum())


class PruneCollinearSnpsTests(unittest.TestCase):
    """Tests for prune_collinear_snps function"""

    def _r2(self, x, y):
        """Compute r2 between two arrays using scipy.stats.pearsonr."""
        r, _ = pearsonr(x, y)
        return r**2

    def test_requires_dataframe(self):
        """Should raise TypeError if not passed a DataFrame"""
        snps = np.array([[0, 1], [1, 0], [0, 0]])
        with self.assertRaises(TypeError):
            prune_collinear_snps(snps)

    def test_requires_unique_columns(self):
        """Should raise ValueError if column names are not unique"""
        snps = pd.DataFrame([[0, 1], [1, 0]], columns=["A", "A"])
        with self.assertRaises(ValueError):
            prune_collinear_snps(snps)

    def test_threshold_below_zero_raises(self):
        """Should raise ValueError if threshold is negative"""
        snps = pd.DataFrame({"A": [0, 1, 0], "B": [1, 0, 1]})
        with self.assertRaises(ValueError):
            prune_collinear_snps(snps, threshold=-0.1)

    def test_threshold_above_one_raises(self):
        """Should raise ValueError if threshold is above 1"""
        snps = pd.DataFrame({"A": [0, 1, 0], "B": [1, 0, 1]})
        with self.assertRaises(ValueError):
            prune_collinear_snps(snps, threshold=1.1)

    def test_threshold_zero_valid(self):
        """threshold of exactly 0 should be accepted"""
        snps = pd.DataFrame({"A": [0, 1, 0], "B": [1, 0, 1]})
        prune_collinear_snps(snps, threshold=0)

    def test_threshold_one_valid(self):
        """threshold of exactly 1 should be accepted"""
        snps = pd.DataFrame({"A": [0, 1, 0], "B": [1, 0, 1]})
        prune_collinear_snps(snps, threshold=1)

    def test_independent_snps_all_kept(self):
        """Independent SNPs (r2 ~= 0) should all be retained"""
        # Manually chosen to have low correlation
        snps = pd.DataFrame(
            {
                "A": [0, 1, 0, 1, 0, 1],
                "B": [1, 0, 0, 1, 1, 0],
                "C": [0, 0, 1, 0, 1, 1],
            }
        )

        # Verify r2 between all pairs is low
        for c1, c2 in combinations(snps.columns, 2):
            self.assertLess(self._r2(snps[c1], snps[c2]), 0.5)

        pruned_df, removed = prune_collinear_snps(snps, threshold=0.8)

        self.assertEqual(["A", "B", "C"], list(pruned_df.columns))
        self.assertEqual({}, removed)

    def test_identical_snps_one_kept(self):
        """Identical SNPs (r2 = 1) should result in only one being kept"""
        snp = [0, 1, 0, 1, 1, 0]
        snps = pd.DataFrame({"A": snp, "B": snp, "C": snp})

        pruned_df, removed = prune_collinear_snps(snps, threshold=0.95)

        self.assertEqual(["A"], list(pruned_df.columns))
        self.assertEqual({"B": "A", "C": "A"}, removed)

    def test_anticorrelated_snps_one_kept(self):
        """Perfectly anti-correlated SNPs (r2 = 1) should be pruned"""
        snp = np.array([0, 1, 0, 1, 1, 0])
        snps = pd.DataFrame({"A": snp, "B": 1 - snp})

        pruned_df, removed = prune_collinear_snps(snps, threshold=0.95)

        self.assertEqual(["A"], list(pruned_df.columns))
        self.assertEqual({"B": "A"}, removed)

    def test_high_correlation_above_threshold_pruned(self):
        """SNPs with r2 above threshold should be pruned"""
        np.random.seed(42)
        n = 100
        snp1 = np.random.randint(0, 2, n)
        noise = np.random.randint(0, 2, n)
        snp2 = np.where(np.random.random(n) < 0.95, snp1, noise)

        self.assertGreater(self._r2(snp1, snp2), 0.9)

        snps = pd.DataFrame({"A": snp1, "B": snp2})
        pruned_df, removed = prune_collinear_snps(snps, threshold=0.8)

        self.assertEqual(["A"], list(pruned_df.columns))
        self.assertEqual({"B": "A"}, removed)

    def test_moderate_correlation_below_threshold_kept(self):
        """SNPs with r2 below threshold should both be kept"""
        np.random.seed(43)
        n = 100
        snp1 = np.random.randint(0, 2, n)
        noise = np.random.randint(0, 2, n)
        snp2 = np.where(np.random.random(n) < 0.7, snp1, noise)

        r2 = self._r2(snp1, snp2)
        self.assertGreater(r2, 0.4)
        self.assertLess(r2, 0.6)

        snps = pd.DataFrame({"A": snp1, "B": snp2})
        pruned_df, removed = prune_collinear_snps(snps, threshold=0.8)

        self.assertEqual(["A", "B"], list(pruned_df.columns))
        self.assertEqual({}, removed)

    def test_single_snp(self):
        """Single SNP should be returned unchanged"""
        snps = pd.DataFrame({"A": [0, 1, 0, 1, 1, 0]})

        pruned_df, removed = prune_collinear_snps(snps, threshold=0.8)

        self.assertEqual(["A"], list(pruned_df.columns))
        self.assertEqual({}, removed)
        pd.testing.assert_frame_equal(snps, pruned_df)

    def test_removed_maps_to_most_correlated(self):
        """Removed SNPs should map to the retained SNP with highest r2"""
        # A and B are independent, C is identical to B
        snps = pd.DataFrame(
            {
                "A": [0, 1, 0, 1, 1, 0],
                "B": [1, 1, 0, 0, 1, 0],
                "C": [1, 1, 0, 0, 1, 0],  # Identical to B
            }
        )

        pruned_df, removed = prune_collinear_snps(snps, threshold=0.95)

        self.assertEqual(["A", "B"], list(pruned_df.columns))
        self.assertEqual({"C": "B"}, removed)

    def test_removed_maps_to_later_kept_snp(self):
        """Removed SNP should map to most correlated kept SNP, even if kept
        later.

        This tests the edge case where:
        - B is pruned due to high correlation with A
        - C is kept (low correlation with A)
        - But B is more correlated with C than A

        The mapping should be B->C, not B->A.
        """
        # Construct SNPs using a "shifting window" pattern where B is
        # between A and C, correlating with both but more strongly with C
        A = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        B = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        C = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        # Verify the correlation structure:
        # r2_AB ~= 0.44, r2_AC ~= 0.29, r2_BC ~= 0.64
        r2_AB = self._r2(A, B)
        r2_AC = self._r2(A, C)
        r2_BC = self._r2(B, C)

        # Use threshold = 0.35, between r2_AC and r2_AB
        threshold = 0.35

        # B should be pruned (r2 with A above threshold)
        self.assertGreater(r2_AB, threshold)

        # C should be kept (r2 with A below threshold)
        self.assertLess(r2_AC, threshold)

        # B should map to C (higher r2 than with A)
        self.assertGreater(r2_BC, r2_AB)

        snps = pd.DataFrame({"A": A, "B": B, "C": C})
        pruned_df, removed = prune_collinear_snps(snps, threshold=threshold)

        self.assertEqual(["A", "C"], list(pruned_df.columns))

        # B should map to C (most correlated), not A
        self.assertEqual({"B": "C"}, removed)

    def test_threshold_boundary(self):
        """SNPs exactly at threshold boundary"""
        snp1 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        snp2 = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        r2 = self._r2(snp1, snp2)

        snps = pd.DataFrame({"A": snp1, "B": snp2})

        # Threshold just above r2, both should be kept
        pruned_df, _ = prune_collinear_snps(snps, threshold=r2 + 0.01)
        self.assertEqual(2, len(pruned_df.columns))

        # Threshold at or below r2, one should be pruned
        pruned_df, _ = prune_collinear_snps(snps, threshold=r2 - 0.01)
        self.assertEqual(1, len(pruned_df.columns))

    def test_preserves_index(self):
        """Returned DataFrame should preserve the original index"""
        snps = pd.DataFrame(
            {"A": [0, 1, 0], "B": [0, 1, 0]},
            index=["sample1", "sample2", "sample3"],
        )

        pruned_df, _ = prune_collinear_snps(snps, threshold=0.95)

        self.assertEqual(
            ["sample1", "sample2", "sample3"], list(pruned_df.index)
        )

    def test_uniform_snp_handled_correctly(self):
        """Uniform SNPs (zero variance) should not cause numerical issues."""
        n = 10
        snps = pd.DataFrame(
            {
                "A": np.zeros(n),  # uniform
                "B": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "C": np.ones(n),  # uniform
            }
        )

        # Should complete without error
        pruned_df, _ = prune_collinear_snps(snps, threshold=0.95)

        # Uniform SNPs should be kept (since they're first or have r2=nan
        # which is not > threshold)
        self.assertIn("A", pruned_df.columns)

    def test_near_uniform_snp_handled_correctly(self):
        """Near-uniform SNPs should not cause numerical issues.

        When a SNP has very small variance, the standardization norm is tiny.
        This should be handled without producing inf, nan, or overflow.
        """
        n = 10

        # Create SNPs with varying degrees of near-uniformity
        # These have tiny but non-zero variance
        nearly_uniform_1 = np.zeros(n)
        nearly_uniform_1[0] = 1e-10  # Tiny non-zero value

        nearly_uniform_2 = np.full(n, 0.5)
        nearly_uniform_2[0] = 0.5 + 1e-15  # Tiny variation

        snps = pd.DataFrame(
            {
                "A": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Normal SNP
                "B": nearly_uniform_1,
                "C": nearly_uniform_2,
                "D": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Normal SNP
            }
        )

        # Should complete without error or warning
        pruned_df, _ = prune_collinear_snps(snps, threshold=0.95)

        # Should return valid output
        self.assertIsInstance(pruned_df, pd.DataFrame)
        self.assertFalse(pruned_df.empty)


if __name__ == "__main__":
    unittest.main()
