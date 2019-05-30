"""Classes for handling DataFrames containing coordinates and sequences."""

from __future__ import print_function
from builtins import range, object

from itertools import combinations

import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import quantile_transform

from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt

from .helper import expand_sequences, site_consensus
from .munging import df_from_fasta


class CoordDf(object):

    def __init__(self, df):
        """Coordinate data.

        Args:
            df (pd.DataFrame): Must contain x and y columns.
        """
        self.df = df

    def __repr__(self):
        return "CoordDf with {} samples and {} dimensions:\n{}".format(
            *self.df.shape, repr(self.df))

    def rotate(self, a, inplace=True):
        """Rotate points a degrees around the origin anticlockwise.

        Args:
            a (Number): Arc degrees to rotate the dataframe by.
            inplace (bool): Rotate the data inplace, or return a rotated
                copy of the data.
        """
        theta = np.radians(a)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        arr = np.matmul(self.df, R)
        df = pd.DataFrame(arr, index=self.df.index, columns=self.df.columns)

        if inplace:
            self.df = df
        else:
            return CoordDf(df=df)

    def points_in_patch(self, patch):
        """Points in sel.df contained in a matplotlib patch.

        Args:
            patch (matplotlib.patches.Patch)

        Returns:
            Strains in patch.
        """
        fig, ax = plt.subplots()
        self.df.plot.scatter(x="x", y="y", ax=ax)
        ax.add_artist(patch)
        path = patch.get_transform().transform_path(patch.get_path())
        mask = path.contains_points(ax.transData.transform(self.df))
        plt.close()
        return mask

    def pca_rotate(self, inplace=True):
        """Rotate coordinates along first and second principal components.

        Args:
            inplace (bool): Rotate the data inplace, or return a rotated
                copy of the data.
        """
        n_components = self.df.shape[1]
        coord_pca = PCA(n_components=n_components).fit(self.df)
        df = pd.DataFrame(coord_pca.transform(self.df))
        df.columns = ["PC{}".format(i + 1) for i in range(n_components)]
        df.index = self.df.index
        if inplace:
            self.df = df
        else:
            return CoordDf(df=df)

    def quantile_transform(self, inplace=True):
        """Transform features using quantile information.

        Notes:
            http://scikit-learn.org/stable/modules/generated/
            sklearn.preprocessing.quantile_transform.html#sklearn.
                preprocessing.quantile_transform

        Args:
            inplace (bool)
        """
        arr = quantile_transform(self.df, output_distribution="normal")
        df = pd.DataFrame(arr, index=self.df.index, columns=self.df.columns)

        if inplace:
            self.df = df
        else:
            return CoordDf(df=df)

    def paired_distances(self, other):
        """Compute euclidean distances between points in self.df and paired
        points in another dataframe. The other dataframe must have the same
        dimensions as self.df

        Args:
            other (pd.DataFrame)

        Returns:
            (ndarray): Euclidean distances.
        """
        if self.df.index.shape != other.index.shape:
            raise ValueError("Index lengths mismatched.")

        if self.df.columns.shape != other.columns.shape:
            raise ValueError("Column lengths mismatched.")

        n = self.df.shape[0]

        distances = np.empty(n)

        for i in range(n):

            try:
                distances[i] = euclidean(
                    u=self.df.iloc[i, :],
                    v=other.iloc[i, :])

            except ValueError:
                distances[i] = np.nan

        return distances


class SeqDf(object):

    def __init__(self, df):
        """DataFrames containing amino acid sequences.

        Args:
            df (pd.DataFrame): Columns are amino acid positions, rows are
                samples, cells contain amino acids.
        """
        self.df = df

    def __repr__(self):
        return "SeqDf with {} samples and {} sites\n{}:".format(
            *self.df.shape, repr(self.df))

    def __str__(self):
        return str(self.df)

    @classmethod
    def from_fasta(cls, path):
        """Make a SeqDf from a fasta file.

        Args:
            path (str): Path to fasta file.

        Returns:
            (SeqDf)
        """
        return cls(df_from_fasta(path=path, positions="infer"))

    @classmethod
    def from_series(cls, series):
        """Make SeqDf from a series.

        Args:
            series (pd.Series): Each element in series is a string. See
                mapdeduce.helper.expand_sequences for more details.

        Returns:
            (SeqDf)
        """
        return cls(expand_sequences(series))

    def remove_invariant(self, inplace=True):
        """Remove positions (columns) that contain only one amino acid.

        Args:
            inplace (bool)
        """
        mask = self.df.apply(lambda x: pd.unique(x).shape[0] > 1)
        n = (~mask).sum()
        print("Removed {} invariant sequence positions".format(n))
        new = self.df.loc[:, self.df.columns[mask]]

        if inplace:
            self.df = new
        else:
            return new

    def get_dummies(self, inplace=True):
        """Get dummy representation of the sequences.

        Args:
            inplace (bool)
        """
        d = pd.get_dummies(self.df, prefix_sep="").astype(float)

        if inplace:
            self.dummies = d
        else:
            return d

    def shuffle_dummies(self, n_shuffles, c):
        """Return a DataFrame containing n shuffles of the data in column c

        Args:
            n_shuffles (int): Number of shuffles.
            c (str): Must be column in self.dummies.

        Returns:
            (ndarray): Shape [N, n_shuffles].
        """
        values = self.dummies.loc[:, c].values
        arr = np.empty((values.shape[0], n_shuffles))
        for i in range(n_shuffles):
            arr[:, i] = sklearn.utils.shuffle(values)
        return arr

    def get_dummies_at_positions(self, positions):
        """Return set of dummy variable names at HA positions.

        Notes:
            Dummy variable names are either singles (e.g. 135K), or compound
            (e.g. 7D|135K|265E). For compound dummy variable names return the
            entire compound name if any constituent SNP is in positions.

        Args:
            positions (iterable) containing positions.

        Returns:
            (set) containing dummy variable names.
        """
        dummies = set()
        add = dummies.add

        for dummy in self.dummies.columns:

            for c in dummy.split("|"):
                pos = int(c[: -1])

                if pos in positions:
                    add(dummy)
                    break

        return dummies

    def merge_duplicate_dummies(self, inplace=False):
        """Merge SNPs that are identical in all strains.

        Args:
            inplace (bool)
        """
        grouped = self.dummies.T.groupby(by=self.dummies.index.tolist())

        df = pd.DataFrame(
            data={"|".join(g.index): n for n, g in grouped},
            index=self.dummies.index)

        if inplace:
            self.dummies = df
        else:
            return df

    def consensus(self):
        """Compute the consensus sequence.

        Returns:
            (pd.Series)
        """
        return self.df.apply(site_consensus, axis=0)

    def merge_duplicate_strains(self, inplace=False):
        """Replace all strains that have the same index with a single
        consensus strain.

        Args:
            inplace (bool).

        Returns:
            (mapdeduce.dataframes.SeqDf) if inplace=False.
        """
        vc = self.df.index.value_counts()
        dupes = (vc[vc > 1]).index
        data = {d: SeqDf(self.df.loc[d]).consensus() for d in dupes}
        cons = pd.DataFrame.from_dict(data, orient="index")
        df = pd.concat([self.df.drop(dupes, axis=0), cons])

        if inplace:
            self.df = df
        else:
            return SeqDf(df)

    def to_fasta(self, path):
        """Write the sequences in fasta format.

        Args:
            path (str): Path to write file.
        """
        if not path.lower().endswith(".fasta"):
            path += ".fasta"
        with open(path, "w") as handle:
            for row in self.df.iterrows():
                handle.write(">{}\n".format(row.name))
                handle.write("{}\n".format("".join(row)))

    def groupby_amino_acid_at_site(self, p):
        """Lookup groups of strains that have the same amino acid at site p.

        Args:
            p (int): Site. Must be a column in self.df.

        Returns:
            (dict): Maps amino acid -> set containing strain names.
        """
        if p not in self.df.columns:
            raise ValueError("{} not in self.df.columns".format(p))
        return {amino_acid: set(group.index)
                for amino_acid, group in self.df.groupby(self.df.loc[:, p])}

    def substitutions_at_site(self, p, min_strains=0):
        """Find substitutions that occur at site p.

        Args:
            p (int). Must be in self.df.
            min_strains (int). Minimum number of strains that must posses a
                given amino acid to be included. Default=0 to include all
                strains.

        Returns:
            (dict): Maps substituion -> pd.Series containing profile of
                substitution. Strains with 0 have the aa0. Strains with 1 have
                the aa1.
        """
        groups = self.groupby_amino_acid_at_site(p)
        rv = {}
        for aa0, aa1 in combinations(groups, 2):
            aa0, aa1 = sorted((aa0, aa1))
            if len(groups[aa0]) < min_strains:
                continue
            elif len(groups[aa1]) < min_strains:
                continue
            else:
                data = [0.0] * len(groups[aa0]) + [1.0] * len(groups[aa1])
                index = list(groups[aa0]) + list(groups[aa1])
                series = pd.Series(data=data, index=index)
                series.name = "{}{}{}".format(aa0, str(p), aa1)
                rv[str(series.name)] = series
        return rv
