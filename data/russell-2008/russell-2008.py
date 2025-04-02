import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import mapdeduce as md
    return md, np, pd, plt


@app.cell
def _(plt):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = 8, 4
    return


@app.cell
def _(pd):
    df = pd.read_csv("layout-seq.csv")
    df.index = df["Strain"] + "_" + df["Accession"]
    return (df,)


@app.cell
def _(df, pd):
    seq_df = df["Sequence"].str.split("").apply(pd.Series)[range(109, 302)]
    seq_df.columns = [str(col) for col in seq_df.columns]
    return (seq_df,)


@app.cell
def _(df):
    coord_df = df[["x", "y"]]
    return (coord_df,)


@app.cell
def _(coord_df, md, seq_df):
    oms = md.OrderedMapSeq(seq_df=seq_df, coord_df=coord_df)
    return (oms,)


@app.cell
def _():
    sites_of_interest = [145, 225, 193, 227, 226, 189, 159]
    return (sites_of_interest,)


@app.cell
def _(oms, plt, sites_of_interest):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))
    for site, _ax in zip(sites_of_interest, fig.axes):
        oms.plot_amino_acids_at_site(str(site), ax=_ax)
    axes[-1, -1].axis("off")
    plt.show()
    return axes, fig, site


@app.cell
def _(oms):
    oms.plot_amino_acids_at_site("145")
    return


@app.cell
def _(oms):
    oms.filter(remove_invariant=True, get_dummies=True, merge_duplicate_dummies=True)
    return


@app.cell
def _(md, oms):
    at = md.AssociationTest(dummies=oms.seqs.dummies, phenotypes=oms.coord.df)
    return (at,)


@app.cell
def _(mo):
    mo.md(r"""## Common AAPs""")
    return


@app.cell
def _(oms):
    aap_counts = oms.seqs.dummies.sum().sort_values()
    aap_counts.plot(ylabel="AAP frequency", xlabel="AAP", figsize=(8, 3), ylim=(0, len(oms.seqs.df)), clip_on=False, zorder=10)
    return (aap_counts,)


@app.cell
def _(aap_counts, oms):
    min_aap_count = 5
    n = len(oms.seqs.dummies)
    common_aaps = aap_counts.index[(5 <= aap_counts) & (aap_counts <= (n - min_aap_count))]
    print(len(common_aaps))
    print(common_aaps)
    return common_aaps, min_aap_count, n


@app.cell
def _():
    # with mo.persistent_cache(".marimo_cache"):
    #     df_test = at.test_aaps(common_aaps)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## FU02 - CA04 substitutions

        Just look at FU02 - CA04 cluster difference substitutions and compare the p-values and effect sizes estimated by the original limix.
        """
    )
    return


@app.cell
def _():
    fu02_ca04_aaps = ["145N", "145K", "159F", "226V", "159Y", "189S", "226I", "227S", "227P", "189N"]
    return (fu02_ca04_aaps,)


@app.cell
def _(at, fu02_ca04_aaps, mo):
    with mo.persistent_cache(".marimo_cache"):
        df_test_fu02_ca04 = at.test_aaps(fu02_ca04_aaps)
    return (df_test_fu02_ca04,)


@app.cell
def _(pd):
    df_prior_fu02_ca04 = pd.DataFrame(
        [
            ["145N", 2.59e-07, 9.6e-07, 2],
            ["145K", 3.43e-05, 0.000127, 1.51],
            ["159F", 0.0137, 0.0508, 1.35],
            ["226V", 0.0221, 0.0821, 1.28],
            ["159Y", 0.101, 0.374, 0.842],
            ["189S", 0.294, 1, 0.854],
            ["226I", 0.465, 1, 0.587],
            ["227S", 0.808, 1, 0.242],
            ["227P", 0.808, 1, 0.242],
            ["189N", 0.867, 1, 0.264],
        ],
        columns=["aap", "p_value", "p_value_corrected", "beta_joint"],
    ).set_index("aap")
    return (df_prior_fu02_ca04,)


@app.cell
def _(df_prior_fu02_ca04, df_test_fu02_ca04):
    df_comb = df_prior_fu02_ca04.join(df_test_fu02_ca04, lsuffix="_prior")
    return (df_comb,)


@app.cell
def _(plt):
    def compare(df, x, y, ax=None, logscale: bool=False):
        ax = ax or plt.gca()
        df.plot.scatter(x, y, ax=ax)
        if logscale:
            ax.set(yscale="log", xscale="log")
        ax.axline((0.1, 0.1), (1, 1), c="lightgrey", lw=1)
        ax.set(aspect=1)
        return ax
    return (compare,)


@app.cell
def _(compare, df_comb, plt):
    _, _axes = plt.subplots(ncols=3, figsize=(10, 4))

    compare(df_comb, "p_value_prior", "p_value", logscale=True, ax=_axes[0])
    compare(df_comb, "p_value_corrected_prior", "p_value_corrected", logscale=True, ax=_axes[1])
    compare(df_comb, "beta_joint_prior", "beta_joint", ax=_axes[2])

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
