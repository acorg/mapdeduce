import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import mapdeduce as md
    import matplotlib.pyplot as plt
    import numpy as np
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
def _(oms):
    dummy_proportions = oms.seqs.dummies.mean().sort_values()
    dummy_proportions.plot()
    return (dummy_proportions,)


@app.cell
def _(dummy_proportions):
    common_aaps = dummy_proportions.index[(0.01 <= dummy_proportions) & (dummy_proportions <= 0.99)]
    common_aaps
    return (common_aaps,)


@app.cell
def _(at, common_aaps, mo):
    with mo.persistent_cache(name=".marimo_cache"):
        df_test = at.test_aaps(common_aaps)
    return (df_test,)


@app.cell
def _(df_test, np, plt):
    plt.scatter(np.linspace(0, 1, len(df_test)), df_test["p_value"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
