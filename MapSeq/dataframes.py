"""Classes for handling DataFrames containing coordinates and sequences"""

import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import quantile_transform

import shapely
import shapely.affinity as affinity
from shapely.geometry.point import Point
from shapely.geometry.multipoint import MultiPoint

import matplotlib.pyplot as plt


class CoordDf(object):
    """Simple class for Dataframes containing x, y coordinates"""

    def __init__(self, df):
        """@param df: pd.DataFrame. Must contain x and y columns."""
        self.df = df

    def rotate(self, a, inplace=True):
        """
        Rotate points specified by the xy coordinates in the DataFrame
        a degrees around the origin anticlockwise.

        This method is very slow with the number of points I typically have
        in datasets (~5000). Instantiating this many Point instances is very
        slow.

        @param a: Number. Arc degrees to rotate the dataframe by
        @param inplace: Bool. Rotate the data inplace, or return a rotated
            copy of the data.
        """
        multipoint = MultiPoint(self.df.apply(Point, axis=1))
        rotated = affinity.rotate(multipoint, angle=a)
        array = np.array(shapely.geometry.base.dump_coords(rotated))
        df = pd.DataFrame(array[:, 0])
        df.columns = self.df.columns
        df.index = self.df.index
        if inplace:
            self.df = df
        else:
            return CoordDf(df=df)

    def points_in_patch(self, patch):
        """
        Return points in df contained in a matplotlib patch

        @param patch. matplotlib.patches.Patch instance
        """
        fig, ax = plt.subplots()
        self.df.plot.scatter(x="x", y="y", ax=ax)
        ax.add_artist(patch)
        path = patch.get_transform().transform_path(patch.get_path())
        mask = path.contains_points(ax.transData.transform(self.df))
        plt.close()
        return mask

    def pca_rotate(self, inplace=True):
        """
        Rotate the coordinates along first and second principal components.

        @param inplace: Bool. Rotate the data inplace, or return a rotated
            copy of the data.
        """
        coord_pca = PCA(n_components=2).fit(self.df)
        df = pd.DataFrame(coord_pca.transform(self.df))
        df.columns = "PC1", "PC2"
        df.index = self.df.index
        if inplace:
            self.df = df
        else:
            return CoordDf(df=df)

    def quantile_transform(self, inplace=True):
        """
        Transform features using quantile information. See
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.
        quantile_transform.html#sklearn.preprocessing.quantile_transform

        @param inplace: Bool.
        """
        arr = quantile_transform(self.df, output_distribution="normal")
        df = pd.DataFrame(arr)
        df.columns = self.df.columns
        df.index = self.df.index
        if inplace:
            self.df = df
        else:
            return CoordDf(df=df)


class SeqDf(object):
    """Class for dataframes containing amino acid sequences"""

    def __init__(self, df):
        """@param df: pd.DataFrame. Columns are amino acid positions"""
        self.df = df

    def remove_invariant(self, inplace=True):
        """Remove positions (columns) that contain only one amino acid

        @param inplace: Bool.
        """
        mask = self.df.apply(lambda x: pd.unique(x).shape[0] > 1)
        print "Removed {} sequence positions that were invariant".format(
            (~mask).sum())
        new = self.df.loc[:, self.df.columns[mask]]

        if inplace:
            self.df = new
        else:
            return new

    def get_dummies(self, inplace=True):
        """Get dummy variable represenation of the sequences

        @param inplace: Bool.
        """
        d = pd.get_dummies(self.df, prefix_sep="").astype(float)

        if inplace:
            self.dummies = d
        else:
            return d

    def shuffle_dummies(self, n_shuffles, c):
        """Return a DataFrame containing n shuffles of the data in column c

        @param n_shuffles: Int. Number of shuffles
        @param c: Must lookup column in self.dummies

        @returns (N, n_shuffles) ndarray
        """
        values = self.dummies.loc[:, c].values
        arr = np.empty((values.shape[0], n_shuffles))
        for i in xrange(n_shuffles):
            arr[:, i] = sklearn.utils.shuffle(values)
        return arr

    def get_dummies_at_positions(self, positions):
        """Return a list of dummy variable names at positions

        @param positions: List of integer positions
        """
        dummies = []
        for c in self.dummies.columns:
            pos = int(c[: -1])
            if pos in positions:
                dummies.append(c)
        return dummies

    def merge_duplicate_dummies(self, inplace=False):
        """Merge SNPs that are identical in all strains.

        @param inplace: Bool.
        """
        grouped = self.dummies.T.groupby(by=self.dummies.index.tolist())

        df = pd.DataFrame(
            data={"|".join(g.index): n for n, g in grouped},
            index=self.dummies.index
        )

        if inplace:
            self.dummies = df
        else:
            return df
