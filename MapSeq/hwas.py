"""Classes and functions for running Hemagglutinin wide association studies"""

import numpy as np
import scipy
from limix.qtl import qtl_test_lmm
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def pd_qtl_test_lmm(**kwargs):
    """
    Pandas wrapper for qtl_test_lmm. qtl_test_lmm "snps" and "pheno" kwargs
    now have to be pd.DataFrame intances. Checks that the indexes for both
    those dfs are the same. Attaches the p-values, effect sizes and standard
    errors as a pd.Panel to the lmm object.

    @param kwargs: Passed to qtl_test_lmm
    """
    snps = kwargs.pop("snps", None)
    pheno = kwargs.pop("pheno", None)
    assert (snps.index != snps.index).sum() == 0
    lmm = qtl_test_lmm(snps.values, pheno.values, **kwargs)

    try:
        minor_axis = snps.columns
    except AttributeError:
        minor_axis = [snps.name, ]

    lmm.results = pd.Panel(
        data=[lmm.getPv(), lmm.getBetaSNP(), lmm.getBetaSNPste()],
        items=["p-values", "effect-size", "std-error"],
        major_axis=pheno.columns,       # phenotypes
        minor_axis=minor_axis,          # snps
    )
    return lmm


def cov(m):
    """
    Compute the covariance matrix of m

    @param m: ndarry / dataframe
    """
    return np.dot(m, m.T) / float(m.shape[0])


def qq_plot(results, label_top_n=5, effect_size=True, **kwargs):
    """
    Plot a quantile-quantile comparison plot of p-values

    @param results pd.Panel: Like pd.Panel returned by pd_qtl_test_lmm.
        items: p-values, effect-size, std-error
        major axis: Phenotypes
        minor axis: SNPs
    @param label_top_n int: Label the index of the top n substitutions with
        the lowest p values
    @param effect_size: Bool. Plot the effect size with error bars

    Optional kwargs

    @param phenotype: Specify the phenotype to plot. Default is the first
        phenotype in self.pheno.columns
    """
    ax = plt.gca()
    phenotype = kwargs.pop("phenotype", results.major_axis[0])

    # Get 2D DataFrame contianing pvalues and effect sizes for this
    # phenotype
    df = results[:, phenotype, :]
    df.sort_values("p-values", inplace=True)

    # Parameters for the qq plot
    n = df.shape[0]
    x = -1 * np.log10(np.linspace(1 / float(n), 1, n))
    y = -1 * np.log10(df["p-values"])
    ax.scatter(x, y, zorder=20, label="-log10(P-value)", edgecolor="white")
    ax.scatter(x, np.abs(df.loc[:, "effect-size"]),
               label="Abs. effect size",
               edgecolor="white",
               zorder=15)
    ax.errorbar(x=x,
                y=np.abs(df.loc[:, "effect-size"]),
                yerr=df.loc[:, "std-error"],
                lw=0,
                elinewidth=1,
                ecolor="grey",
                label="Effect size std. err",
                zorder=10)
    # Finalise plot and label top SNPs
    for i in range(label_top_n):
        ax.text(x=x[i] + 0.02, y=y[i] - 0.01, s=df.index[i], ha="left",
                va="top", zorder=15, fontsize=10, rotation=90)
    ax.set_xlabel(r"Null $-log_{10}$(p-value)")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.plot((0, 50), (0, 50), c="white", zorder=10)  # x = y
    return x, y, ax


class HwasLmm(object):
    """
    Linear mixed models to look for associations between amino acid - position
    combinations on antigenic phenotypes.
    """

    def __init__(self, snps, pheno):
        """
        @param snps: df. (N, S). S snps for N individuals
        @param pheno: df. (N, P). P phenotypes for N individuals
        """
        if snps.shape[0] != pheno.shape[0]:
            msg = "snps and pheno have different number of individuals"
            raise ValueError(msg)

        self.N = snps.shape[0]   # Number of individuals
        self.S = snps.shape[1]   # Number of snps
        self.P = pheno.shape[1]  # Number of phenotypes
        self.snps = snps
        self.pheno = pheno

        # Covariance matrix of all the snps
        self.K = cov(snps)

    def compute_k_leave_each_snp_out(self):
        """
        Leave each snp out of self.snps and compute a covariance matrix.
        This attaches a K_leave_out attribute which is a dict. Keys are the
        snp left out. Values are the corresponding covariance matrix.
        """
        self.K_leave_out = {
            s: cov(self.snps.drop(s, axis=1)) for s in self.snps.columns
        }

    def effective_tests(self):
        """
        Compute the effective number of tests, given correlation between
        snps
        """
        corr, p = scipy.stats.spearmanr(self.snps)
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvalues += 1e-12
        return (np.sum(np.sqrt(eigenvalues)) ** 2) / np.sum(eigenvalues)

    def lmm(self, recompute_K=False, **kwargs):
        """
        Run lmm on the snps.

        @param recompute_K. Bool. For each snp, use a covariance matrix
            computed with that snp ommitted.

        Optional kwargs:

        @param snps. df (N, S). N individuals, S snps
        """
        snps = kwargs.pop("snps", self.snps)

        if recompute_K:
            results = {}

            if not hasattr(self, "K_leave_out"):
                self.compute_k_leave_each_snp_out()

            for snp in tqdm(snps.columns):
                lmm = pd_qtl_test_lmm(snps=snps.loc[:, snp],
                                      pheno=self.pheno,
                                      K=self.K_leave_out[snp])
                results[snp] = lmm.results[:, :, snp]
            results = pd.Panel.from_dict(results, orient="minor")

        else:
            lmm = pd_qtl_test_lmm(snps=snps, pheno=self.pheno, K=self.K)
            results = lmm.results

        return results

    def lmm_permute(self, n, recompute_K=False, **kwargs):
        """
        Run lmm on n shuffled permutations of snps.

        @param n. Int. Number of permutations.
        @param recompute_K. Bool. For each snp, use a covariance matrix
            computed with that snp ommitted.

        Optional kwargs:

        @param snps. df (N, S). N individuals, S snps.
        """
        pvalues = np.empty((n, self.S))
        snps = kwargs.pop("snps", self.snps)

        for i in range(n):
            pvalues[i, :] = self.lmm(
                snps=shuffle(snps), recompute_K=recompute_K
            )

        return np.array(pvalues)

    def snp_stripplot(self, snp, **kwargs):
        """
        Stripplot showing the value of the phenotype for the two values of the
        snp

        @param snp: Str. Column name of the snp to plot

        @param kwargs: Passed to sns.stripplot.
        """
        ax = plt.gca()
        x, y = snp, "Phenotype"
        df = pd.DataFrame({
            y: self.pheno.values[:, 0],
            x: self.snps.loc[:, snp].values
        })
        sns.stripplot(data=df, x=x, y=y, color="black", ax=ax, **kwargs)
        # Plot the means of the groups
        means = np.empty((2, 2))
        for i, (x, idx) in enumerate(df.groupby(snp).groups.iteritems()):
            means[i, 0] = x
            means[i, 1] = df.loc[idx, y].mean()
        ax.plot(means[:, 0], means[:, 1], c="darkgrey")
        return ax
