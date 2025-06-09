from typing import Iterable, Hashable

import scipy.stats
import scipy.linalg
import pandas as pd
import numpy as np
import numpy.typing as npt
from glimix_core.lmm import Kron2Sum
from tqdm import tqdm


class MvLMM:
    def __init__(self, dummies: pd.DataFrame, phenotypes: npt.ArrayLike) -> None:
        """
        Conduct association tests between sequences encoded as dummy variables and
        multidimensional phenotypes.

        Args:
            dummies: A DataFrame where columns contain sequence dummies and rows
                contain samples (e.g. individuals).
            phenotypes: Columns contain each phenotype, rows contain samples.
        """
        self.Y = phenotypes
        self.dummies = dummies
        self.a = dummies.shape[1]
        self.n, self.p = self.Y.shape  # n viruses, n phenotypes
        self.A = np.eye(self.p)  # trait by trait covariance matrix

        if set(np.array(dummies).ravel()) != {0, 1}:
            raise ValueError("dummies must contain 0s and 1s")

        if self.dummies.shape[0] != self.Y.shape[0]:
            raise ValueError("dummies and phenotypes have different numbers of rows")

        if len(self.dummies.columns) != len(set(self.dummies.columns)):
            raise ValueError("dummies contains duplicated column names")

        if self.dummies.T.duplicated().any():
            dupes = (
                self.dummies.T.duplicated(keep=False)
                .replace(False, np.nan)
                .dropna()
                .index
            )
            raise ValueError(f"dummies contains duplicate columns: {dupes}")

    def test_aap(self, aap: str) -> dict[str, float]:
        """
        Run an association test on an amino acid polymorphism.

        Args:
            aap: Amino acid polymorphism to test. Must be a column in self.seqs.dummies.
        """
        if not isinstance(self.dummies[aap], pd.Series):
            raise ValueError("can only pass a single aap")

        G = self.dummies.drop(columns=aap)

        if G.shape[1] != (self.dummies.shape[1] - 1):
            raise ValueError("G should have one less column that dummies")

        kwds = dict(Y=self.Y, G=G, A=self.A, restricted=False)

        # covariates for null hypothesis comprise only a constant
        X_h0 = np.ones(self.n).reshape((self.n, 1))

        # covariates for alternate hypothesis comprise constant and single fixed effect
        # of aap
        X_h1 = np.stack((np.ones(self.n), self.dummies[aap])).T

        h0 = Kron2Sum(X=X_h0, **kwds)  # null hypothesis
        h1 = Kron2Sum(X=X_h1, **kwds)  # alternate hypothesis

        h1.fit(verbose=False)
        h0.fit(verbose=False)

        h1_ll = h1.lml()
        h0_ll = h0.lml()

        lrts = compute_likelihood_ratio_test_statistic(h0_ll=h0_ll, h1_ll=h1_ll)

        betas = h1.beta.reshape(self.p, 2)[:, 1]  # Effect size of fixed effect
        p_value = compute_p_value(lrts, df=1)

        return dict(
            aap=aap,
            p_value=p_value,
            p_value_log10=-np.inf if p_value == 0 else np.log10(p_value),
            prop_with_aap=self.dummies[aap].mean(),
            n_with_aap=int(self.dummies[aap].sum()),
            beta_joint=scipy.linalg.norm(betas),
            **{f"beta_{i}": beta for i, beta in enumerate(betas)},
        )

    def test_aaps(self, aaps: Iterable[Hashable]) -> pd.DataFrame:
        """
        Run association tests on aaps.

        Args:
            aaps: Iterable containing column labels in self.dummies.
        """
        aaps = set(aaps)

        if non_existent := aaps - set(self.dummies.columns):
            raise ValueError(f"these aaps don't exist in dummies: {non_existent}")

        aaps = list(aaps)

        df = {}
        for aap in tqdm(aaps):
            df[aap] = self.test_aap(aap)

        df = pd.DataFrame.from_dict(df, orient="index").sort_values("p_value")

        try:
            n_tests = effective_tests(self.dummies[aaps])
        except np.linalg.LinAlgError:
            n_tests = len(aaps)

        # Keep record of number of aaps and effective tests.
        df.attrs["n_aaps"] = len(aaps)
        df.attrs["effective_tests"] = n_tests

        df["p_value_corrected"] = df["p_value"] * n_tests

        # p-values should not exceed 1, but may do after multiplying by n_tests
        gt_one_mask = df["p_value_corrected"] > 1
        df.loc[gt_one_mask, "p_value_corrected"] = 1

        df["p_value_corrected_log10"] = np.log10(df["p_value_corrected"])

        return df

    def variable_aaps(self, threshold: float) -> list:
        """
        Lookup aaps whose variability is above some threshold.

        AAPs are encoded as 0/1. The mean value across many viruses therefore tells you
        the proportion of viruses with that AAP.

        If the threshold is set at 0.05, then any AAP which is present in a proportion
        of viruses greater than or equal to 0.05 and less than or equal to 0.95 is
        returned.

        Args:
            threshold: Must be a number between 0 and 0.5. A threshold of 0 would return
                all AAPs, a threshold of 0.5 would return only AAPs that are present
                in precisely half of viruses.
        """
        if not 0 <= threshold <= 0.5:
            raise ValueError("threshold must be between 0 and 0.5")

        variable = self.dummies.mean(axis=0).apply(
            lambda x: threshold <= x <= (1 - threshold)
        )

        return variable[variable].index.tolist()

    def test_variable_aaps(self, threshold: float) -> pd.DataFrame:
        """
        Run association tests of aaps that are more variable than a threshold.

        Args:
            threshold: See MvLmm.variable_aaps.
        """
        aaps = self.variable_aaps(threshold=threshold)
        return self.test_aaps(aaps)

    def blup(self, test_dummies: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Best linear unbiased predictions (BLUP).

        Args:
            test_dummies: AAPs of strains to make predictions for. Must have identical columns to
                self.dummies.
            verbose: Passed to Kron2Sum.fit.
        """
        if len(test_dummies.columns) != len(self.dummies.columns):
            raise ValueError(
                "test_dummies has a different number of AAPs to self.dummies"
            )

        if not (test_dummies.columns == self.dummies.columns).all():
            raise ValueError(
                "test_dummies has different AAPs to self.dummies or they are in a different order"
            )

        # Fit Kron2Sum
        Y_train = self.Y.values  # Train coordinates (n, p)
        G = self.dummies.values  # Train AAPs (n, a)
        X = np.ones(self.n).reshape(self.n, 1)  # Intercept  (n, 1)
        k2s = Kron2Sum(Y=Y_train, G=G, A=self.A, X=X)
        k2s.fit(verbose=verbose)

        # BLUP
        K = cov(G)  # Train set AAP covariance (n, n)

        # Inverse covariance (n, n)
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)

        G_test = test_dummies.values  # Test AAPs (n_test, a)
        n_test = G_test.shape[0]
        G_all = np.vstack([G, G_test])  # Concatenated AAPs  (n_test + n, a)
        assert G_all.shape == (self.n + n_test, self.a)

        K_all = cov(G_all)  # Covariance of train and test AAPs
        assert K_all.shape == (self.n + n_test, self.n + n_test)

        K_cross = K_all[self.n :, : self.n]  # Cross-covariance of train and test AAPs
        assert K_cross.shape == (n_test, self.n)

        Y_test = K_cross @ K_inv @ Y_train @ k2s.C1

        return pd.DataFrame(Y_test, index=test_dummies.index, columns=self.Y.columns)


def compute_likelihood_ratio_test_statistic(h0_ll: float, h1_ll: float) -> float:
    """
    Compute the likelihood ratio test statistic.

    Args:
        h0_ll: Log-likelihood of the null hypothesis.
        h1_ll: Log-likelihood of the alternate hypothesis.
    """
    return -2 * (h0_ll - h1_ll)


def compute_p_value(lrts: float, df: int) -> float:
    """
    Compute a p-value given a likelihood ratio test statistic and the degrees of freedom.

    Args:
        lrts: Likelihood ratio test statistic.
        df: Degrees of freedom.
    """
    return 1 - scipy.stats.chi2(df=df).cdf(lrts)


def cov(arr: npt.ArrayLike) -> np.ndarray:
    """
    Compute the covariance matrix of an array.

    Args:
        arr:
    """
    return np.dot(arr, arr.T) / float(arr.shape[0])


def effective_tests(dummies: npt.ArrayLike) -> float:
    """
    Compute the effective number of tests, given correlation between snps.
    For 1 SNP return 1.

    Args:
        dummies: (n, s) dummy variables containing 0/1 indicating absence/presence of an
            AAP.
    """
    if dummies.shape[1] == 1:
        return 1.0

    corr, _ = scipy.stats.spearmanr(dummies)

    try:
        eigenvalues, _ = np.linalg.eigh(corr)

    except np.linalg.LinAlgError as err:
        if (dummies.shape[1] == 2) and (corr == -1):
            # if only two SNPs in snps and are perfectly negatively correlated
            return 1
        else:
            raise err

    # Prevent values that should be zero instead being tiny and negative
    eigenvalues += 1e-12
    return (np.sum(np.sqrt(eigenvalues)) ** 2) / np.sum(eigenvalues)
