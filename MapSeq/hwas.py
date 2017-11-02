"""Classes and functions for running Hemagglutinin wide association studies"""

import numpy as np
import scipy
import sklearn
from limix.qtl import qtl_test_lmm
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from permp import permp
from warnings import warn


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
    n_tests = kwargs.pop("n_tests", 1)
    assert (snps.index != snps.index).sum() == 0
    lmm = qtl_test_lmm(snps.values, pheno.values, **kwargs)

    lmm.n_tests = n_tests
    corrected = lmm.getPv() * n_tests
    corrected[corrected > 1] = 1  # No p value should exceed 1

    lmm.results = pd.Panel(
        data=[
            lmm.getPv(),
            lmm.getBetaSNP(),
            lmm.getBetaSNPste(),
            corrected,
        ],
        items=[
            "p-values",
            "effect-size",
            "std-error",
            "p-values-corrected",
        ],
        major_axis=pheno.columns,
        minor_axis=snps.columns,
    )

    return lmm


def shuffle_values(nperm, values):
    """Return an ndarray containing n shuffles of values

    @param nperm: Int. Number of shuffles
    @param arr: ndarray

    @returns (N, nperm) ndarray
    """
    assert values.ndim == 1
    arr = np.empty((values.shape[0], nperm))
    for i in xrange(nperm):
        arr[:, i] = sklearn.utils.shuffle(values)
    return arr


def cov(m):
    """
    Compute the covariance matrix of m

    @param m: ndarry / dataframe
    """
    return np.dot(m, m.T) / float(m.shape[0])


def effective_tests(snps):
    """
    Compute the effective number of tests, given correlation between snps.
    For 1 SNP return 1.

    @param snps: pd.DataFrame
    """
    if snps.shape[1] == 1:
        return 1
    corr, p = scipy.stats.spearmanr(snps)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
    except np.linalg.LinAlgError:
        # if only two SNPs in snps and are perfectly negatively correlated 
        if corr == -1:
            return 1
        else:
            raise np.linalg.LinAlgError()

    # Prevent values that should be zero instead being tiny and negative
    eigenvalues += 1e-12
    return (np.sum(np.sqrt(eigenvalues)) ** 2) / np.sum(eigenvalues)


def qq_plot(results, effect_size=True, snps=None, **kwargs):
    """
    Plot a quantile-quantile comparison plot of p-values

    @param results pd.Panel: Like pd.Panel returned by pd_qtl_test_lmm.
        items: p-values, effect-size, std-error
        major axis: Phenotypes
        minor axis: SNPs
    @param effect_size: Bool. Plot the effect size with error bars
    @param snps: List. Plot only these snps

        Optional kwargs

    @param phenotype: Specify the phenotype to plot. Default is the first
        phenotype in self.pheno.columns
    @param larger: List. SNPs to plot larger.
    @param very_large: List. SNPs to plot very large.
    """
    ax = plt.gca()
    phenotype = kwargs.pop("phenotype", results.major_axis[0])
    larger = kwargs.pop("larger", None)
    very_large = kwargs.pop("very_large", None)

    # Get 2D DataFrame contianing pvalues and effect sizes for this
    # phenotype
    df = results[:, phenotype, :]
    if snps is not None:
        print "Only plotting substitutions at these positions:\n{}".format(
            ",".join(map(str, snps))
        )
        df = pd.concat(map(lambda x: df.filter(regex=str(x), axis=0), snps))
    df.sort_values("p-values", inplace=True)

    if larger is not None:
        s = pd.Series(np.array([i in larger for i in df.index]) * 125 + 25,
                      index=df.index)
    else:
        s = pd.Series(np.repeat(50, df.shape[0]), index=df.index)

    if very_large is not None:
        for vl in very_large:
            s[vl] = 325

    # Parameters for the qq plot
    n = df.shape[0]
    x = pd.Series(-1 * np.log10(np.linspace(1 / float(n), 1, n)),
                  index=df.index)
    scatter_kwds = dict(x=x, edgecolor="white", s=s)
    ax.scatter(y=-1 * np.log10(df["p-values"]),
               zorder=20,
               label="-log10(p-value)",
               c="#c7eae5",
               **scatter_kwds)
    ax.scatter(y=-1 * np.log10(df["p-values-corrected"]),
               zorder=20,
               label="-log10(Corrected p-value)",
               c="#35978f",
               **scatter_kwds)
    if "p-empirical" in df.columns:
        ax.scatter(y=-1 * np.log10(df["p-empirical"]),
                   zorder=18,
                   label="-log10(Empirical p-value",
                   c="#003c30",
                   **scatter_kwds)
    ax.scatter(y=np.abs(df.loc[:, "effect-size"]),
               label="Abs. effect size",
               c="#a6611a",
               zorder=15,
               **scatter_kwds)
    ax.errorbar(x=x,
                y=np.abs(df.loc[:, "effect-size"]),
                yerr=df.loc[:, "std-error"],
                lw=0,
                elinewidth=1,
                ecolor="grey",
                label="Effect size std. err",
                zorder=10)
    # Label larger SNPs
    if larger is not None:
        y = -1 * np.log10(df["p-values-corrected"])
        for snp in larger:
            ax.text(x=x[snp],
                    y=y[snp] + 0.05,
                    s=snp,
                    ha="center",
                    va="bottom",
                    zorder=20,
                    fontsize=10,
                    rotation=90)
    ax.set_xlabel(r"Null $-log_{10}$(p-value)")
    xmax, ymax = ax.get_xlim()[1], ax.get_ylim()[1]
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.plot((0, 50), (0, 50), c="white", zorder=10)  # x = y
    return ax


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
        if (snps.index != pheno.index).sum() != 0:
            raise ValueError("snps and pheno have different indexes")
        self.N = snps.shape[0]   # n individuals
        self.S = snps.shape[1]   # n snps
        self.P = pheno.shape[1]  # n phenotypes
        self.snps = snps
        self.pheno = pheno
        self.K = cov(snps)  # Covariance matrix of snps
        self.n_tests = effective_tests(snps)

    def compute_k_leave_each_snp_out(self):
        """Leave each snp out of self.snps and compute a covariance matrix.
        This attaches a K_leave_out attribute which is a dict. Keys are the
        snp left out. Values are the corresponding covariance matrix.
        """
        self.K_leave_out = {
            s: cov(self.snps.drop(s, axis=1)) for s in self.snps.columns
        }

    def lmm(self, K_without_snp=False, **kwargs):
        """Run lmm on the snps.

        @param K_without_snp. Bool. For each snp, use a covariance matrix
            computed with that snp ommitted.

            Optional kwargs:

        @param snps. df (N, S). N individuals, S snps
        """
        snps = kwargs.pop("snps", self.snps)

        if K_without_snp:
            results = {}

            if not hasattr(self, "K_leave_out"):
                self.compute_k_leave_each_snp_out()

            # Compute the number of tests
            n_tests = 1 if self.S == 1 else float(effective_tests(snps))

            for snp in tqdm(snps.columns):
                lmm = pd_qtl_test_lmm(snps=snps.loc[:, [snp, ]],
                                      pheno=self.pheno,
                                      K=self.K_leave_out[snp],
                                      n_tests=n_tests)
                results[snp] = lmm.results[:, :, snp]
            results = pd.Panel.from_dict(results, orient="minor")

        else:
            lmm = pd_qtl_test_lmm(snps=snps, pheno=self.pheno, K=self.K)
            results = lmm.results

        return results

    def lmm_permute(self, n, K_without_snp=False, **kwargs):
        """Run lmm on n shuffled permutations of snps.

        @param n. Int. Number of permutations.
        @param K_without_snp. Bool. For each snp, use a covariance matrix
            computed with that snp ommitted.

            Optional kwargs:

        @param snps. df (N, S). N individuals, S snps.

        @returns df. Columns are snps. 1 row for each perumutation. Values
            are the p-value for that permutation
        """
        pvalues = np.empty((n, self.S))
        snps = kwargs.pop("snps", self.snps)

        for i in range(n):
            results = self.lmm(snps=sklearn.utils.shuffle(snps),
                               K_without_snp=K_without_snp)
            pvalues[i, :] = results.loc["p-values", :, :]

        df = pd.DataFrame(pvalues)
        df.columns = snps.columns

        return df

    def empirical_p(self, results, cutoff=0.1, nperm=int(1e3)):
        """Compute empirical p-values for SNPs with a p-value lower than cutoff

        @param results. pd.Panel like that returned by pd_qtl_test_lmm which
            contains standard p-values
        """
        if self.pheno.shape[1] > 1:
            warn("Only implemented for univariate phenotypes")
        pheno = self.pheno.columns[0]
        if "p-empirical" in results.items:
            print "empirical pvalues already in results will be overwritten:"
            ser = results.loc["p-empirical", pheno, :]
            print ser[ser.notnull()]

        pvalues = results.loc["p-values-corrected", pheno, :]
        snps_below_cutoff = pvalues.index[pvalues < cutoff]
        empirical_pvalues = {}
        for snp in tqdm(snps_below_cutoff):
            arr = shuffle_values(nperm=nperm,
                                 values=self.snps.loc[:, snp].values)
            lmm = qtl_test_lmm(snps=arr,
                               pheno=self.pheno.values,
                               K=self.K_leave_out[snp])

            # Adjust pvalues by effective number of tests
            perm_pvalues = lmm.getPv() * self.n_tests

            # After adjusting for multiple tests ensure the maximum value
            # for any p-value is 1
            perm_pvalues[perm_pvalues > 1] = 1

            # Now compute the empirical p value
            x = (perm_pvalues <= pvalues[snp]).sum()
            n1, n2 = self.snps.loc[:, snp].value_counts().values
            empirical_pvalues[snp] = permp(x=x,
                                           nperm=nperm,
                                           n1=n1,
                                           n2=n2,
                                           total_nperm=None,
                                           method="auto")[0]
        results.loc["p-empirical", pheno, :] = pd.Series(empirical_pvalues)
        return results

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
