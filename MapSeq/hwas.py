"""Classes and functions for running Hemagglutinin wide association studies"""

import numpy as np
import pandas as pd

import scipy
import sklearn

from limix.qtl import qtl_test_lmm, qtl_test_lmm_kronecker
from limix.vardec import VarianceDecomposition

import warnings
from tqdm import tqdm
from permp import permp

from plotting import plot_arrow, map_setup
from dataframes import CoordDf

import seaborn as sns
import matplotlib.pyplot as plt


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

        @param subtract_pheno_mean: Bool. Mean centre the phenotype.
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
        self.pheno = pheno

        self.N = snps.shape[0]   # n individuals
        self.S = snps.shape[1]   # n snps
        self.P = pheno.shape[1]  # n phenotypes
        self.P0 = pheno.columns[0]
        self.K = cov(snps)

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

    def fit(self, test_snps=None, progress_bar=False):
        """
        Run LMM.

        Results are attached as a results attribute on self.

        @param test_snps: List. Test for associations with these snps.

        @param progress_bar: Bool. Visualise tqdm progress bar.
        """
        if test_snps is None:
            test_snps = self.snps.columns

        if not hasattr(self, "K_leave_out"):
            self.compute_k_leave_each_snp_out(test_snps=test_snps)

        results = {}

        iterable = tqdm(test_snps) if progress_bar else test_snps

        for snp in iterable:

            snp_profile = self.snps.loc[:, [snp, ]].values

            if np.unique(snp_profile).shape[0] != 2:

                warnings.warn(
                    "{} does not have 2 unique values. Skipping.".format(snp)
                )

                continue

            if self.P == 1:

                lmm = qtl_test_lmm(
                    snps=snp_profile,
                    pheno=self.pheno.values,
                    K=self.K_leave_out[snp]
                )

                beta = lmm.getBetaSNP()[0, 0]

            else:

                try:
                    lmm, pv = qtl_test_lmm_kronecker(
                        snps=snp_profile,
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
                        F=snp_profile,
                        A=self.Asnps
                    )

                    vs.addRandomEffect(
                        K=self.K_leave_out[snp]
                    )

                    vs.addRandomEffect(
                        is_noise=True
                    )

                    try:
                        conv = vs.optimize()
                    except np.linalg.LinAlgError:
                        warnings.warn("{} raised LinAlgError".format(snp))
                        continue

                    if conv:
                        lmm, pv = qtl_test_lmm_kronecker(
                            snps=snp_profile,
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
        df["logp"] = np.log10(df["p"]) * -1

        n_tests = effective_tests(self.snps.loc[:, test_snps])
        corrected = df["p"] * n_tests
        df["p-corrected"] = corrected
        df["logp-corrected"] = np.log10(corrected) * -1

        # p-values can't exceed 1
        mask = df["p-corrected"] > 1
        df.loc[mask, "p-corrected"] = 1
        df.loc[mask, "logp-corrected"] = 0

        if self.P > 1:
            df["joint-effect"] = df["beta"].apply(np.linalg.norm)

        df.index.name = "AAP"

        self.results = df

    def regress_out(self, snp, summary_plot=False):
        """
        Regress out the effects of snp from the phenotype. Returns the residual
        phenotype.

        @param snp. Must be in the index of self.snps

        @param summary_plot: Bool. Visualise the phenotype shift.

        @returns residual_pheno: pd.DataFrame. Same shape as self.pheno (N, P)
        """
        beta = self.results.loc[snp, "beta"].reshape(1, -1)
        profile = self.snps.loc[:, snp].values.reshape(-1, 1)
        residual = self.pheno - profile.dot(beta)

        if summary_plot:
            ax = self.pheno.plot.scatter(
                x="x",
                y="y",
                s=5,
                c="black",
                label="Original"
            )

            residual.plot.scatter(
                x="x",
                y="y",
                ax=ax,
                s=10,
                label="Residual"
            )

            joined = residual.join(
                self.pheno,
                lsuffix="resid",
                rsuffix="orig"
            )

            for strain, row in joined.iterrows():
                xmatch = row["xresid"] != row["xorig"]
                ymatch = row["yresid"] != row["yorig"]

                if xmatch and ymatch:
                    ax.plot(
                        (row["xresid"], row["xorig"]),
                        (row["yresid"], row["yorig"]),
                        c="black",
                        zorder=1,
                        lw=0.3
                    )
            map_setup()

            return residual, ax

        else:

            return residual

    def cross_validate(self, n_splits=5, progress_bar=False):
        """
        Conduct K-fold cross validation. Split data into n_splits training and
        testing splits. Train on each training split.

        Attaches list containing the results of the cross validation as a
        self.folds attribute.

        @param n_splits: Int. Number of folds.

        @param progress_bar: Bool. Show tqdm progress bar for each fold.
        """
        if hasattr(self, "folds"):
            raise AttributeError("HwasLmm already has folds attribute.")

        kf = sklearn.model_selection.KFold(
            n_splits=n_splits,
            random_state=1234,
        )

        folds = []
        append = folds.append

        for train, test in kf.split(self.snps):

            train_snps_i = self.snps.iloc[train, :].copy()
            train_pheno_i = self.pheno.iloc[train, :].copy()

            # Train on diverse snps (not all either 0 or 1)
            means = train_snps_i.mean()
            mask = (means > 0) & (means < 1)
            diverse_snps_i = mask.index[mask]

            hwas_i = HwasLmm(
                snps=train_snps_i,
                pheno=train_pheno_i,
            )

            hwas_i.fit(
                test_snps=diverse_snps_i,
                progress_bar=progress_bar,
            )

            test_snps_i = self.snps.iloc[test, :].copy()
            test_pheno_i = self.pheno.iloc[test, :].copy()

            append((hwas_i, test_pheno_i, test_snps_i))

        self.folds = folds

    def cross_validation_predictions(self, p_grid):
        """
        Predict phenotypes for cross validation folds. Include only SNPs that
        have a p value lower than that of each element in p_grid.

        @param p_grid: np.ndarray. p value thresholds.

        @returns pd.Panel: Items are p-values, major axis are strains, minor
            axis are folds. Run pn.apply(np.mean, axis=1) to get fold means
        """
        dists = {}

        for p in p_grid:

            dists[p] = {}

            for i, (hwas, test_pheno, test_snps) in enumerate(self.folds):

                pheno_predict = hwas.predict(
                    snps=test_snps,
                    max_p=p,
                )

                cdf = CoordDf(pheno_predict)

                dists[p][i] = pd.Series(cdf.paired_distances(test_pheno))

        return pd.Panel(data=dists)

    def predict(self, snps, max_p=1, min_effect=0, df=None):
        """
        Predict phenotype values for each individual in SNPs

        @param snps: M x S ndarray. M individuals, S snps

        @param max_p: Number. Only include SNPs that have a p-value less than
            max_p

        @param min_effect: Number. Only include SNPs that have an effect size
            greater than min_effect

        @param df: pd.DataFrame. Optional. Provide df containing effects for
            each snp.
        """
        if df is None:
            df = self.summarise_joint(
                max_p=max_p,
                min_effect=min_effect
            )

        if df.empty:
            raise ValueError(
                "No SNPs with p-value < {:.2E} and effect size > {:.2E}"
                "".format(max_p, min_effect)
            )

        effects = df.filter(regex="b[0-9]", axis=1)

        if effects.empty:
            raise ValueError(
                "df returned from self.summarise_joint does not contain "
                "effects"
            )

        predictors = effects.index & snps.columns

        if predictors.empty:
            raise ValueError(
                "No SNPs predictors to use.\n\n"
                "effects: {effects}\n\n"
                "snps: {snps}".format(
                    effects=effects.index,
                    snps=snps.columns,
                )
            )

        snps = snps.loc[:, predictors]
        effects = effects.loc[predictors, :]

        return pd.DataFrame(
            data=np.dot(snps, effects),
            index=snps.index,
            columns=effects.columns
        )

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

    def plot_antigens(self, color_dict=None, **kwargs):
        """
        2D scatter plot of antigens

        @param color_dict: Dict / None. Values are mpl color for each antigen.
            Overrides c, if c passed as a kwarg.

        @param **kwargs. Passed to self.pheno.plot.scatter
        """
        if color_dict is not None:
            c = [color_dict[i] for i in self.pheno.index]

        else:
            c = kwargs.pop("c", "black")

        return self.pheno.plot.scatter(
            x=self.P0,
            y=self.P1,
            c=c,
            s=kwargs.pop("s", 60),
            lw=kwargs.pop("lw", 0.25),
            edgecolor=kwargs.pop("edgecolor", "white"),
            ax=plt.gca(),
            **kwargs
        )

    def summarise_joint(self, min_effect=0, max_p=1):
        """Make a summary dataframe of the joint effects. Columns comprise
        effect sizes in each dimension individually, the joint effect size, the
        p-value of the association, and -1 x log10(p-value).

        @param min_effect: Number. Only include snps with a joint effect size >
            min_effect

        @param max_p: Number. Only inlucde snps with a p-value < max_p.

        @returns. pd.DataFrame. Containing the summary.
        """
        df = self.results["beta"].apply(pd.Series)
        df.columns = ["b{}".format(i) for i in range(df.shape[1])]
        df["joint"] = self.results["beta"].apply(np.linalg.norm)
        df["snp"] = df.index
        df["logp"] = self.results["logp"]
        df["p"] = self.results["p"]

        df = df[df["p"] < max_p]
        df = df[df["joint"] > min_effect]

        df.sort_values(by=["logp", "snp"])
        df.drop("snp", axis=1, inplace=True)

        return df

    def plot_antigens_with_snp(self, snp, jitter=0, randomz=None, **kwargs):
        """Plot antigens that have a snp.

        @param snp. Must specify a column in self.snps

        @param jitter. Number. Add jitter to the antigen positions. Random
            uniform jitter is generated in the interval -1, 1, multiplyed by
            the value of jitter, and added to the values that are visualised.

        @param randomz. None / Number. If not None, then each point gets a
            random z value within +/- 0.5 units of randomz
        """
        ax = kwargs.pop("ax", plt.gca())

        mask = self.snps.loc[:, snp] == 1
        n = mask.sum()

        offsets = np.random.uniform(
            low=-1,
            high=1,
            size=n * 2,
        ) * jitter

        df = self.pheno[mask] + offsets.reshape(n, 2)

        if randomz:

            df["z"] = np.random.uniform(
                low=-0.5,
                high=0.5,
                size=n
            ) + randomz

            for idx, row in df.iterrows():
                ax.scatter(
                    x=row[self.P0],
                    y=row[self.P1],
                    zorder=row["z"],
                    **kwargs
                )

        else:
            df.plot.scatter(
                x=self.P0,
                y=self.P1,
                ax=ax,
                **kwargs
            )

    def plot_multi_effects(self, min_effect=0, max_p=1, snps=None,
                           label_arrows=False, plot_strains_with_snps=True,
                           colors=None, plot_similar_together=False,
                           max_groups=8, test_pos=None, lw_factor=1):
        """
        Visualisation of 2D joint effects detected.

        Arrows are plotted that correspond to the joint effect vector. The
        arrow tip points towards the mean position of strains with the SNP.
        Arrow width is proportional to -1 x log10(p-value), so that SNPs that
        are 10x more significant twice the width.

        @param min_effect: Number. Only show snps with a joint effect
             > min_effect

        @param max_p: Number. Only show snps with a p value < max_p

        @param snps: List. Show these snps. Overrides max_p and min_effect.

        @param label_arrows: Bool. Attach labels to the arrows

        @param plot_strains_with_snps: Bool. Mark which strains have which SNPs

        @param colors: List of mpl colors to use for arrows. Should be at at
            least as long as how many arrows will be plotted.

        @param plot_similar_together. Bool. Plot snps with similar effects
            and p-values with the same arrow. This rounds the effect sizes and
            logp values to 2 decimal places, and performs a groupby on these
            columns.

        @param max_groups: Number. Maximum number of groups to show if plotting
            similar together.

        @param test_pos. List. Only show SNPs at these positions. There may
            be snps at positions that aren't being tested that have the same
            profile as one that does. In that case the un-wanted position will
            be in the dummy name. Remove positions that aren't being tested
            from the dummy names

        @param lw_factor: Number. Arrow linewidths are:

                     -1 * log10(p-value) * lw_factor

        @returns ax: Matplotlib ax
        """
        df = self.summarise_joint(
            min_effect=0,
            max_p=1,
        )

        if snps is not None:

            df = df.loc[snps, :]

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
                pv = "{:.4F}".format(self.results.loc[snp, "p"])
                j = "{:.2F}".format(group.loc[snp, "joint"])
                label = "{} {} {}".format(pv, j, snps_sorted)

                arrows.append({
                    "end": end,
                    "start": start,
                    "label": label,
                    "logp": logp,
                    "snp": snp,
                })

        else:

            for dummy, row in df.iloc[:max_groups, :].iterrows():

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

                pv = "{:.4F}".format(self.results.loc[dummy, "p"])
                j = "{:.2F}".format(row["joint"])
                label = "{} {} {}".format(pv, j, snps_sorted)

                arrows.append({
                    "end": end,
                    "start": start,
                    "label": label,
                    "logp": row["logp"],
                    "snp": dummy,
                })

        # Plotting

        ax = plt.gca()

        if colors is None:
            colors = sns.color_palette("Set1", len(arrows))

        for a, c in zip(arrows, colors):
            a["color"] = c

        if ax.get_legend():
            leg_artists, leg_labels = ax.get_legend_handles_labels()
        else:
            leg_artists, leg_labels = [], []

        for a in arrows:

            label = a["label"] if label_arrows else ""
            leg_labels.append(a["label"])

            leg_artists.append(
                plot_arrow(
                    start=a["start"],
                    end=a["end"],
                    color=a["color"],
                    lw=a["logp"] * lw_factor,
                    zorder=20,
                    label=label,
                    ax=ax
                )
            )

            if plot_strains_with_snps:

                self.plot_antigens_with_snp(
                    snp=a["snp"],
                    jitter=0.1,
                    c=a["color"],
                    edgecolor="white",
                    s=40,
                    randomz=1,
                    alpha=0.5,
                    lw=1,
                    ax=ax,
                )

        ax.legend(
            leg_artists,
            leg_labels,
            bbox_to_anchor=(1, 0.5),
            loc="center left",
        )

        map_setup(ax)

        return ax
