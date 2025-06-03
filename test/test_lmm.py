import unittest

import numpy as np
import pandas as pd

import mapdeduce as md


class TestMvLMM(unittest.TestCase):
    def test_cant_pass_dummies_and_phenotypes_with_different_rows(self):
        """
        Shouldn't be able to pass dummies and phenotypes with a different number of rows.
        """
        # (10, 3) dummies
        dummies = pd.DataFrame(np.random.randint(0, 2, size=30).reshape(10, 3))

        # (10, 2) phenotypes
        phenotypes = np.random.randn(9, 2)

        msg = "dummies and phenotypes have different numbers of rows"
        with self.assertRaisesRegex(ValueError, msg):
            md.MvLMM(dummies=dummies, phenotypes=phenotypes)

    def test_dummies_only_contains_zero_one(self):
        """
        Dummies should only contain values of zero and one.
        """
        dummies = pd.DataFrame(np.arange(9).reshape(3, 3))
        phenotypes = np.arange(9).reshape(3, 3)
        msg = "dummies must contain 0s and 1s"
        with self.assertRaisesRegex(ValueError, msg):
            md.MvLMM(dummies=dummies, phenotypes=phenotypes)

    def test_cannot_pass_multiple_aaps(self):
        """
        Passing multiple aaps should raise an error.
        """
        np.random.seed(42)
        dummies = pd.DataFrame(np.random.randint(0, 2, size=9).reshape(3, 3))
        phenotypes = np.arange(9).reshape(3, 3)
        at = md.MvLMM(dummies, phenotypes)
        with self.assertRaisesRegex(ValueError, "can only pass a single aap"):
            at.test_aap([1, 2])

    def test_single_aap(self):
        """
        Testing a single aap should return a dict.
        """
        n = 10  # number of viruses
        p = 2  # number of phenotypes
        d = 20  # number of dummies
        np.random.seed(45)
        dummies = pd.DataFrame(np.random.randint(0, 2, size=n * d).reshape(n, d))
        phenotypes = np.random.randn(n, p)
        at = md.MvLMM(dummies, phenotypes)
        results = at.test_aap(0)
        self.assertIsInstance(results, dict)
        self.assertIsInstance(results["p_value"], float)

    def test_multiple_aaps(self):
        """
        Testing multiple aaps returns a DataFrame.
        """
        n = 10  # number of viruses
        p = 2  # number of phenotypes
        d = 20  # number of dummies
        np.random.seed(45)
        dummies = pd.DataFrame(np.random.randint(0, 2, size=n * d).reshape(n, d))
        phenotypes = np.random.randn(n, p)
        at = md.MvLMM(dummies, phenotypes)
        df = at.test_aaps([0, 1, 2, 3])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.loc[0, "p_value"], float)

    def test_non_unique_columns(self):
        """
        If dummies contains repeated columns then a ValueError should be raised.
        """
        # Create dummies with duplicate columns
        dummies = pd.DataFrame(
            {"col1": [0, 1, 0], "col2": [1, 0, 1], "col3": [0, 1, 0]}  # Same as col1
        )
        phenotypes = np.random.randn(3, 2)

        msg = "dummies contains duplicate columns"
        with self.assertRaisesRegex(ValueError, msg):
            md.MvLMM(dummies=dummies, phenotypes=phenotypes)
