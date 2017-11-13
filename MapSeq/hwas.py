"""Classes and functions for running Hemagglutinin wide association studies"""

import numpy as np
import scipy
import sklearn

from limix.qtl import qtl_test_lmm, qtl_test_lmm_kronecker
from limix.vardec import VarianceDecomposition

from plotting import plot_arrow, map_setup

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from permp import permp
import warnings


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


def qq_plot(results, snps=None, **kwargs):
    """
    Plot a quantile-quantile comparison plot of p-values

    @param results pd.DataFrame: Like pd.Panel returned by pd_qtl_test_lmm.
        columns must contain "p" and can also contain the following:
                    p-corrected
                    beta
                    std-error
                    p-empirical
        DataFrame indexes are SNPs.
    @param snps: List. Plot only these snps

        Optional kwargs

    @param larger: List. SNPs to plot larger.
    @param very_large: List. SNPs to plot very large.
    """
    ax = plt.gca()
    larger = kwargs.pop("larger", None)
    very_large = kwargs.pop("very_large", None)

    # Get 2D DataFrame contianing pvalues and effect sizes for this
    # phenotype
    df = results
    if snps is not None:
        print "Only plotting substitutions at these positions:\n{}".format(
            ",".join(map(str, snps))
        )
        df = pd.concat(map(lambda x: df.filter(regex=str(x), axis=0), snps))
    df.sort_values("p", inplace=True)

    if larger is not None:
        s = pd.Series(np.array([i in larger for i in df.index]) * 125 + 25,
                      index=df.index)
    else:
        s = pd.Series(np.repeat(50, df.shape[0]), index=df.index)

    if very_large is not None:
        for vl in very_large:
            s[vl] = 325

    # qq plot parameters
    n = df.shape[0]
    x = pd.Series(
        -1 * np.log10(np.linspace(1 / float(n), 1, n)),
        index=df.index
    )
    scatter_kwds = dict(
        x=x,
        edgecolor="white",
        s=s
    )

    ax.scatter(
        y=df["logp"],
        zorder=15,
        label="-log10(p-value)",
        c="#c7eae5",
        **scatter_kwds
    )

    try:
        ax.scatter(
            y=df["logp-corrected"],
            zorder=15,
            label="-log10(Corrected p-value)",
            c="#35978f",
            **scatter_kwds
        )
    except KeyError:
        pass

    try:
        ax.scatter(
            y=-1 * np.log10(df["logp-empirical"]),
            zorder=10,
            label="-log10(Empirical p-value",
            c="#003c30",
            **scatter_kwds
        )
    except KeyError:
        pass

    try:
        ax.scatter(
            y=df.loc[:, "joint-effect"],
            label="Joint-effect",
            c="#a6611a",
            zorder=10,
            **scatter_kwds
        )
    except KeyError:
        pass

    # Label larger SNPs
    if very_large is not None:

        y = df["logp"]

        for snp in very_large:

            try:
                ax.text(
                    x=x[snp],
                    y=y[snp] + 0.05,
                    s=snp,
                    ha="center",
                    va="bottom",
                    zorder=20,
                    fontsize=10,
                    rotation=90
                )
            except KeyError:
                warnings.warn("{} not labelled".format(snp))

    ax.set_xlabel(r"Null $-log_{10}$(p-value)")

    ax.set_xlim(
        left=0,
        right=ax.get_xlim()[1]
    )

    ax.set_ylim(
        bottom=0,
        top=ax.get_ylim()[1]
    )

    ax.plot(
        (0, 50),
        (0, 50),
        c="white",
        zorder=10
    )

    ax.legend(
        bbox_to_anchor=(1, 1),
        loc="upper left",
    )

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

        if (len(snps.index) != len(set(snps.index))):
            raise ValueError("snps indices aren't all unique")
        if (len(snps.columns) != len(set(snps.columns))):
            raise ValueError("snps columns aren't all unique")
        if (len(pheno.index) != len(set(pheno.index))):
            raise ValueError("pheno indices aren't all unique")
        if (len(pheno.columns) != len(set(pheno.columns))):
            raise ValueError("pheno columns aren't all unique")

        self.snps = snps
        self.pheno_shift = pheno.mean()
        self.pheno = pheno - self.pheno_shift

        self.N = snps.shape[0]   # n individuals
        self.S = snps.shape[1]   # n snps
        self.P = pheno.shape[1]  # n phenotypes
        self.P0 = pheno.columns[0]
        self.K = cov(snps)
        self.n_tests = 1 if self.S == 1 else effective_tests(self.snps)
        if self.P > 1:
            self.Asnps = np.eye(self.P)
            self.P1 = pheno.columns[1]

    def compute_k_leave_each_snp_out(self, test_snps=None):
        """Leave each snp out of self.snps and compute a covariance matrix.
        This attaches a K_leave_out attribute which is a dict. Keys are the
        snp left out. Values are the corresponding covariance matrix.

        @param test_snps: List. Only compute covariance matrix without snps
            for these snps.
        """
        test_snps = self.snps.columns if test_snps is None else test_snps
        self.K_leave_out = {
            s: cov(self.snps.drop(s, axis=1)) for s in test_snps
        }

    def lmm(self, test_snps=None):
        """Run LMM

        @param test_snps: List. Only test for associations with these snps.
        @returns pd.DataFrame. Containing p-values and effect sizes of the snps
            tested.
        """
        if not hasattr(self, "K_leave_out"):
            self.compute_k_leave_each_snp_out(test_snps=test_snps)

        if test_snps is None:
            test_snps = self.snps.columns

        results = {}

        for snp in tqdm(test_snps):
            print snp
            if self.P == 1:

                lmm = qtl_test_lmm(
                    snps=self.snps.loc[:, [snp, ]].values,
                    pheno=self.pheno.values,
                    K=self.K_leave_out[snp]
                )

                beta = lmm.getBetaSNP()[0, 0]

            else:

                try:
                    lmm, pv = qtl_test_lmm_kronecker(
                        snps=self.snps.loc[:, [snp, ]].values,
                        phenos=self.pheno.values,
                        Asnps=self.Asnps,
                        K1r=self.K_leave_out[snp]
                    )

                except AssertionError:

                    warnings.warn("Doing manual VarianceDecomposition")

                    vs = VarianceDecomposition(
                        Y=self.pheno.values
                    )

                    vs.addFixedEffect(
                        F=self.snps.loc[:, [snp, ]].values,
                        A=self.Asnps
                    )

                    vs.addRandomEffect(
                        K=self.K_leave_out[snp]
                    )

                    vs.addRandomEffect(
                        is_noise=True
                    )

                    conv = vs.optimize()

                    if conv:
                        lmm, pv = qtl_test_lmm_kronecker(
                            snps=self.snps.loc[:, [snp, ]].values,
                            phenos=self.pheno.values,
                            Asnps=self.Asnps,
                            K1r=self.K_leave_out[snp],
                            K1c=vs.getTraitCovar(0),
                            K2c=vs.getTraitCovar(1)
                        )

                    else:
                        raise ValueError("Variance decom. didn't optimize")

                # lmm.getBetaSNP() returns (P, S) array of effect sizes
                # Only tested 1 snp
                beta = lmm.getBetaSNP()[:, 0]

            results[snp] = {
                "p": lmm.getPv()[0, 0],
                "beta": beta,
            }

        df = pd.DataFrame.from_dict(results, orient="index")
        df.sort_values("p", inplace=True)

        corrected = df["p"] * self.n_tests
        corrected[corrected > 1] = 1
        df["p-corrected"] = corrected

        df["logp"] = np.log10(df["p"]) * -1
        df["logp-corrected"] = np.log10(df["p-corrected"]) * -1

        if self.P > 1:
            df["joint-effect"] = df["beta"].apply(np.linalg.norm)

        return df

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
            pvalues[i, :] = results.loc["p", :, :]

        df = pd.DataFrame(pvalues)
        df.columns = snps.columns

        return df

    def empirical_p(self, results, cutoff=0.1, nperm=int(1e3)):
        """Compute empirical p-values for SNPs with a p-value lower than cutoff

        @param results. pd.Panel like that returned by pd_qtl_test_lmm which
            contains standard p-values
        """
        if self.pheno.shape[1] > 1:
            warnings.warn("Only implemented for univariate phenotypes")
        pheno = self.pheno.columns[0]
        if "p-empirical" in results.items:
            print "empirical pvalues already in results will be overwritten:"
            ser = results.loc["p-empirical", pheno, :]
            print ser[ser.notnull()]

        pvalues = results.loc["p-corrected", pheno, :]
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

    def plot_multi_effects(self, results, color_dict=None, max_groups=8,
                           label_arrows=False, min_effect=0, max_p=1,
                           plot_similar_together=False, test_pos=None,
                           plot_antigens=True):
        """
        @param results: pd.DataFrame like that returned from HwasLmm.lmm()
        @param color_dict: Dictionary containing colors for antigens
        @param max_groups: Number. Maximum number of groups to show.
        @param label_arrows: Bool. Attach labels to the arrows
        @param min_effect: Number. Only show snps with a joint effect
             > min_effect
        @param max_p: Number. Only show snps with a p value < max_p
        @param plot_similar_together. Bool. Plot snps with similar effects
            and p-values with the same arrow. This rounds the effect sizes and
            logp values to 2 decimal places, and performs a groupby on these
            columns.
        @param test_pos. List. HA positions that are being tested. There may
            be snps at positions that aren't being tested that have the same
            profile as one that does. In that case the un-wanted position will
            be in the dummy name. Remove positions that aren't being tested
            from the dummy names
        @param plot_antigens: Bool. Plot the antigens as well.
        """
        if plot_antigens:
            if color_dict is not None:
                c = [color_dict[i] for i in self.pheno.index]
            else:
                c = "black"

            ax = self.pheno.plot.scatter(
                x=self.P0,
                y=self.P1,
                c=c,
                zorder=20,
                s=60,
                lw=0.25,
                edgecolor="white",
            )

        df = results["beta"].apply(pd.Series)
        df.columns = "b0", "b1"
        df["joint"] = results["beta"].apply(np.linalg.norm)
        df["snp"] = df.index
        df["logp"] = results["logp"]
        df = df[results["p"] < max_p]  # Now may not be same len as results
        df = df[df["joint"] > min_effect]
        df.sort_values(by=["logp", "snp"])

        # Find starts, ends, and labels for arrows

        arrows = []

        if plot_similar_together:

            df = np.round(df, decimals=2)

            grouped = df.groupby(
                by=["logp", "b0", "b1"],
                sort=False
            )

            for (logp, b0, b1), group in tuple(grouped)[:max_groups]:

                snp = group.index[0]  # A representative snp
                end = self.pheno[self.snps.loc[:, snp] == 1].mean()
                start = end - np.array([b0, b1])

                snps_sorted = "\n            ".join(group.index.sort_values())
                pv = "{:.4F}".format(results.loc[snp, "p"])
                j = "{:.2F}".format(group.loc[snp, "joint"])
                label = "{} {} {}".format(pv, j, snps_sorted)

                arrows.append({
                    "end": end,
                    "start": start,
                    "label": label,
                    "logp": logp,
                })

        else:

            for dummy, row in df.iterrows():

                mask = self.snps.loc[:, dummy] == 1
                end = self.pheno[mask].mean().values
                start = end - row[["b0", "b1"]].values

                if test_pos is None:
                    snps_sorted = "\n            ".join(dummy.split("|"))

                else:
                    store = []
                    for i in dummy.split("|"):
                        pos = int(i[:-1])
                        if pos in test_pos:
                            store.append(i)
                    snps_sorted = "\n            ".join(store)

                pv = "{:.4F}".format(results.loc[dummy, "p"])
                j = "{:.2F}".format(row["joint"])
                label = "{} {} {}".format(pv, j, snps_sorted)

                arrows.append({
                    "end": end,
                    "start": start,
                    "label": label,
                    "logp": row["logp"],
                })

        # Plot the arrows

        color = iter(sns.color_palette("Set1", len(arrows)))
        leg_artists, leg_labels = [], []

        for a in arrows:

            label = a["label"] if label_arrows else ""
            leg_labels.append(a["label"])

            leg_artists.append(
                plot_arrow(
                    start=a["start"],
                    end=a["end"],
                    color=next(color),
                    lw=a["logp"],
                    zorder=5,
                    label=label,
                )
            )

        ax = plt.gca()
        ax.legend(
            leg_artists,
            leg_labels,
            bbox_to_anchor=(1, 0.5),
            loc="center left",
        )

        map_setup(ax)

        return ax
