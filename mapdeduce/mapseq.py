"""Contains the main class to represent maps with sequences."""

from __future__ import print_function, division

from builtins import zip, map, str, range, object

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import warnings

import numpy as np
import spm1d

from scipy import spatial

import sklearn
from sklearn import neighbors

from operator import and_
import itertools

import tqdm

from .plotting import setup_ax, amino_acid_colors, add_ellipses, \
    combination_label, point_size, map_setup
from .munging import handle_duplicate_sequences
from .data import amino_acids
from .dataframes import CoordDf, SeqDf
from .helper import is_not_amino_acid
from functools import reduce


class MapSeq(object):

    def __init__(self, seq_df, coord_df, map=None):
        """An antigenic map with sequence data.

        Args:
            seq_df (pd.DataFrame): Indexes are strains. Columns are amino acid
                positions.
            coord_df (pd.DataFrame): Indexes are strains. Columns are "x" and
                "y" coordinates.
            map (int): 2009 or 2017. Optional. Specify a known map to configure
                plotting boundaries.

        Attributes:
            sequence_df (pd.DataFrame): Columns are sites. Rows are strains.
                Cells contain amino acids.
            coord_df (pd.DataFrame): Columns are coordinate dimensions. rows
                are strains. Cells contain numbers.
            strains_with_seq (set): Strains that have a sequence.
            strains_with_coords (set): Strains that have antigenic coordinates.
            strains_with_both (set): Strains that have both antigenic
                coordinates and a sequence.
            strains_with_only_coords (set): Strains that have coordinates, but
                don't have a sequence.
            seq_in_both (pd.DataFrame): Like self.sequence_df, but only for
                strains that have a sequence and coordinates.
            coords_in_both (pd.DataFrame): Like self.coord_df, but only for
                strains that have a sequence and coordinates.
            variant_positions (set): Column names in sequence_df that contain
                more than one unqique element.
            map (int): Optional. Integer encoding which antigenic map this is.
        """
        self.sequence_df = handle_duplicate_sequences(seq_df)
        self.coord_df = coord_df.copy()
        self.map = map

        # Replace any unknown amino acids with NaN
        cond = self.sequence_df.applymap(lambda x: x not in amino_acids)
        self.sequence_df.mask(cond, inplace=True)

        # Remove any rows that contain NaN coords
        mask = self.coord_df.notnull().any(axis=1)
        self.coord_df = self.coord_df.loc[mask, :]

        # Strains in the sequences and coordinates dataframes
        self.strains_with_seq = set(self.sequence_df.index)
        self.strains_with_coords = set(self.coord_df.index)

        # Strains in both sequence and coordinate dataframes
        self.strains_with_both = \
            self.strains_with_seq & self.strains_with_coords

        # Strains that only have coordinates
        self.strains_with_only_coords = \
            self.strains_with_coords - self.strains_with_seq

        # Coordinates and sequences of strains in both
        self.seq_in_both = self.sequence_df.loc[self.strains_with_both, :]
        self.coords_in_both = self.coord_df.loc[self.strains_with_both, :]

        self.coords_of_strains_without_sequence = \
            self.coord_df.loc[self.strains_with_only_coords]

    @property
    def variant_positions(self):
        """Positions that have different amino acids."""
        _variant_positions = set()
        for p in self.seq_in_both.columns:
            if len(self.seq_in_both.loc[:, p].unique()) != 1:
                _variant_positions.add(p)
        return _variant_positions

    def plot_with_without(self, ax=None, **kwds):
        """Plot indicating which antigens do and do not have sequences.

        Args:
            ax (matplotlib ax): Optional matplotlib ax.
            **kwds passed to pd.DataFrame.plot.scatter.
        """
        ax = plt.gca() if ax is None else ax
        kwds = dict(ax=ax, x="x", y="y")
        n_without_sequence = self.coords_of_strains_without_sequence.shape[0]
        self.coords_of_strains_without_sequence.plot.scatter(
            color="darkgrey",
            label="Without sequence ({})".format(n_without_sequence),
            **kwds)
        n_with_sequence = self.coords_in_both.shape[0]
        self.coords_in_both.plot.scatter(
            color="#b72467",
            label="With sequence ({})".format(n_with_sequence),
            **kwds)
        setup_ax(map=self.map)
        map_setup()
        return ax

    def variant_proportions(self, p):
        """Of antigens with sequences, compute the proportion of amino acids
        at each position, p.

        Args:
            p (int): HA position

        Returns:
            pd.Series
        """
        series = self.seq_in_both.loc[:, p]
        value_counts = series.value_counts()
        return (value_counts / value_counts.sum()).sort_values()

    def plot_amino_acids_at_site(self, p, randomz=True, ellipses=True,
                                 title=True, ax=None, **kwds):
        """Plot map colored by amino acids at site p.

        Args:
            p (int): HA site.
            randomz (bool): Given points random sites in z. This is
                slower because marks have to plotted individually.
            ellipses (bool): Demark clusters with ellipses.
            title (bool): Add a title to the figure.
            ax (matplotlib ax): Plot figure on this ax.
            **kwds passed to ax.scatter for the colored strains.

        Returns:
            (matplotlib ax)
        """
        ax = plt.gca() if ax is None else ax

        # Antigens without a known sequence
        if not self.coords_of_strains_without_sequence.empty:
            self.coords_of_strains_without_sequence.plot.scatter(
                ax=ax, x="x", y="y", s=5, color="darkgrey",
                label="Unknown sequence")

        # Antigens with a known sequence
        kwds = dict(
            lw=1, edgecolor="white",
            s=3 * point_size(self.seq_in_both.shape[0]), **kwds)

        proportions = self.variant_proportions(p=p) * 100
        seq_grouped = self.seq_in_both.groupby(p)

        for amino_acid, seq_group in seq_grouped:

            coord_group = self.coords_in_both.loc[seq_group.index, :]

            try:
                kwds["color"] = amino_acid_colors[amino_acid]

            except KeyError:
                kwds["color"] = amino_acid_colors["X"]

            label = "{} {:.1f}%".format(amino_acid, proportions[amino_acid])

            if randomz:
                zorders = np.random.rand(seq_group.shape[0]) + 5

                for z, (strain, row) in zip(zorders, coord_group.iterrows()):
                    ax.scatter(x=row["x"], y=row["y"], zorder=z, **kwds)

                # Plot the final point twice, but add a label for the group
                ax.scatter(x=row["x"], y=row["y"], label=label, **kwds)

            else:
                coord_group.plot.scatter(
                    label=label, x='x', y='y', ax=ax, **kwds)

        if ellipses and self.map is not None:
            add_ellipses(self.map)

        setup_ax(self.map)

        ax.legend()
        if title:
            ax.text(x=0.5, y=1, s=p, fontsize=25, va="top",
                    transform=ax.transAxes, ha="center")

        map_setup(ax)
        return ax

    def plot_single_substitution(self, sub, ellipses=True, filename=None,
                                 connecting_lines=True, **kwds):
        """Plot map showing strains that differ by sub.

        Args:
            sub (tuple): Substitution. E.g. ("N", 145, "K")
            ellipses (bool): Demark clusters with ellipses.
            filename (str): Passed to plt.savefig. None does not save figure.
            connecting_lines (bool): Plot lines between each points that
                differ by the substitution.
            **kwds passed to self.single_substitutions. Use to restrict
                to particular sites.
        """
        # Strains that differ by only the substituion sub
        combinations = self.single_substitutions(sub, **kwds)

        label = "".join(map(str, sub))

        if not combinations:
            print("No pairs of strains with {}".format(label))
            return

        # Collect x, y of points to plot, and lines between
        aas = sub[0], sub[2]
        for i, pairs in enumerate(combinations):

            fig, ax = plt.subplots()

            # Antigens without a known sequence
            self.coords_of_strains_without_sequence.plot.scatter(
                ax=ax, x="x", y="y", s=5, color="lightgrey",
                label="Unknown sequence")

            # Antigens with a known sequence
            self.coords_in_both.plot.scatter(
                ax=ax, x="x", y="y", s=10, color="darkgrey",
                label="Known sequence")

            # More transparent lines when there are more points
            alpha = 0.95 ** len(pairs)
            alpha = 0.3 if alpha < 0.3 else alpha

            # Plot strains that have aa0 and aa1
            samples = [None, None]
            for j in 0, 1:
                strains = set(pair[j] for pair in pairs)

                samples[j] = self.coords_in_both.loc[strains, :]

            # Plot the group with more samples first
            # Prevents over plotting
            for sample in sorted(samples, key=lambda x: len(x))[::-1]:
                sample.plot.scatter(
                    x="x", y="y", s=150, c=amino_acid_colors[aas[j]],
                    edgecolor="white", linewidth=1, zorder=20, ax=ax,
                    label="{}{}".format(aas[j], sub[1]))

            # Compute Hotelling's T-squared statistic on the two samples
            if len(samples[0]) > 1 and len(samples[1]) > 1:
                h = spm1d.stats.hotellings2(*samples)
                h_report = "p = {:.2E}\nz = {:.3f}\ndf = {:d}, {:d}".format(
                    h.inference().p, h.z, *list(map(int, h.df)))
            else:
                h_report = "[Insufficient data]"

            ax.text(
                x=0, y=1, ha="left", va="top", transform=ax.transAxes,
                s=r"2 sample Hotelling's T$^2$" + "\n" + h_report)

            if connecting_lines:
                for pair in pairs:
                    # May be multiple strains with the same name that have
                    # different coordinates. Plot lines between all
                    # combinations
                    pair_coords = self.coords_in_both.loc[pair, :]

                    segments = []

                    if pair_coords.shape[0] > 2:
                        # Some strains are repated in the map
                        # Look for all combinations of groups of strains
                        groups = [group for name, group in
                                  pair_coords.groupby(pair_coords.index)]

                        for idx, series_1 in groups[0].iterrows():
                            for idx, series_2 in groups[1].iterrows():
                                segments.append(
                                    (series_1.values, series_2.values)
                                )

                    elif pair_coords.shape[0] == 2:
                        segments.append(pair_coords.values)

                    else:
                        raise ValueError("This 'pair' indexes less that 2 "
                                         "strains\n{}".format(pair))

                    ax.add_collection(
                        LineCollection(
                            segments=segments, lw=1, color="black",
                            alpha=alpha, zorder=10, label=""))

            if ellipses:
                add_ellipses(self.map)

            setup_ax(self.map)
            ax.legend()
            title = "{} ({})".format(label, i + 1)
            ax.set_title(title)
            map_setup()
            if filename is not None:
                plt.tight_layout()
                plt.savefig(filename.format(title.replace(" ", "")))

    def plot_variant_positions_colored_by_amino_acid(self, filename, **kwds):
        """Call plot_amino_acids_at_site for all variant positions

        Args:
            filename (str): A format string with one field to fill in. Each
                position will be substituted in. E.g.:
                "img/melb-h3-2009-coloured-by-pos/{}.png"

            **kwds: Keyword arguments passed to plot_amino_acids_at_site
        """
        # Save text file containing html of the variant positions. 1 sorted by
        # primary structure the second by most common variant
        sorted_primary = sorted(self.variant_positions)
        sorted_most_common = \
            self.variant_positions_sorted_by_most_common_variant()

        img_tag = '<img src="{}" class="map" />\n'

        with open(".by_primary.txt", "w") as fobj:
            for p in sorted_primary:
                fobj.write(img_tag.format(filename.format(p)))

        with open(".by_most_common.txt", "w") as fobj:
            for p in sorted_most_common:
                fobj.write(img_tag.format(filename.format(p)))

        print("Wrote .by_primary.txt and .by_most_common.txt.")

        n = len(self.variant_positions)
        print("There are {} variant positions.".format(n))
        print("Doing", end="")

        for i, p in enumerate(self.variant_positions):
            print("{}".format(p), end="")

            fig, ax = plt.subplots()
            self.scatter_colored_by_amino_acid(p, **kwds)
            map_setup()
            plt.tight_layout()
            plt.savefig(filename.format(p), bbox_inches="tight")
            plt.close()

    @property
    def variant_positions_sorted_by_most_common_variant(self):
        """Lookup variant positions.

        Returns:
            tuple. Contains variant positions, sorted by most common variant.
        """
        proportions_all_variants = list(
            map(self.variant_proportions, self.variant_positions))
        proportions_all_variants_sorted = sorted(
            proportions_all_variants, key=lambda x: x[-1])
        return tuple(p.name for p in proportions_all_variants_sorted)

    def strains_with_combinations(self, combinations, without=False):
        """Lookup strains that have combinations of amino acids at particular
        positions.

        Args:
            combinations (dict): Keys are positions, values are amino
                acids:  E.g. {145: "N", 189: "S"}
            without (bool): Lookup strains without the combination.

        Returns:
            pd.DataFrame. Contains strains with the amino acid combinations.
        """
        for k in list(combinations.keys()):
            if k not in self.seq_in_both.columns:
                raise ValueError("Position {} is unknown.")

        masks = (self.seq_in_both.loc[:, k] == v
                 for k, v in combinations.items())

        mask = reduce(and_, masks)

        if without:
            mask = ~ mask

        df = self.seq_in_both[mask]

        if df.empty:

            label = "+".join(
                sorted(
                    "{}{}".format(k, v) for k, v in combinations.items()))

            warnings.warn("No strains with {}".format(label))

        return df

    def duplicate_sequences(self, **kwds):
        """Find groups of duplicate sequences.

        Any element in self.seq_in_both that is not one of the standard 20
        1-letter amino acid abbreviations is ignored. (It is treated as NaN,
        see: http://pandas.pydata.org/pandas-docs/stable/
            missing_data.html#na-values-in-groupby)

        Args:
            positions (iterable of ints). Optional. Lookup groups of sequences
                identical at these positions. Default is all positions.

        Returns:
            pd.GroupBy
        """
        positions = kwds.pop("positions", self.seq_in_both.columns.tolist())
        df = self.seq_in_both
        return df.mask(df.applymap(is_not_amino_acid)).groupby(positions)

    def plot_strains_with_combinations(self, combinations, without=False,
                                       plot_other=True, ax=None, **kwds):
        """Plot a map highlighting strains with combinations of amino acids
        at particular positions.

        Args:
            combinations (dict): Keys are positions, values are amino
                acids:  E.g. {145: "N", 189: "S"}.
            without (bool): Plot strains without the combination.
            plot_other (bool): Plot other antigens (those without the
                combinations).
            ax (matplotlib ax): Optional.
            **kwds passed to the plt.scatter.

        Returns:
            (matplotlib ax)
        """
        ax = plt.gca() if ax is None else ax
        df = self.strains_with_combinations(combinations, without=without)

        label = "+".join(sorted("{}{}".format(k, v)
                                for k, v in combinations.items()))
        label = "Without {}".format(label) if without else label

        if df.empty:
            raise ValueError("No strains with {}".format(label))

        else:
            strains = df.index

            if plot_other:
                other = set(coord_df.index) - set(strains)
                self.coord_df.loc[list(other), :].plot.scatter(
                    x="x", y="y", s=10, color="darkgrey", ax=ax)

            ax.scatter(
                self.coord_df.loc[strains, "x"],
                self.coord_df.loc[strains, "y"],
                label=label, **kwds)
            ax.legend()

            if self.map:
                add_ellipses(self.map)
                setup_ax(self.map)

            map_setup(ax)
            return ax

    def plot_strains_with_combinations_kde(self, combinations, c=0.9,
                                           color="black", ax=None,
                                           label=False, clabel_kwds=dict(),
                                            **kwds):
        """Plot a countour that contains c percent of the density of a KDE of
        strains with combinations of AAPs specified in combinations.

        Args:
            combinations (dict): Dictionary specifying combinations. E.g.:
                {145: "N", 133: "D"}
            c (number): Range 0-1. The contour contains c percent of the KDE
                density.
            color (matlplotlib colour): Colour to plot the contour line
            ax (matplotlib ax): Optional.
            label (bool): Label the contour.
            clabel_kwds (dict): Optional kwds for ax.clabel if label is True.
            **kwds passed to plt.contour.

        Returns:
            (matplotlib ax)
        """
        ax = plt.gca() if ax is None else ax
        df = self.strains_with_combinations(combinations)
        strains = df.index

        dataset = self.coord_df.loc[strains, :]

        grid = sklearn.model_selection.GridSearchCV(
            estimator=neighbors.KernelDensity(kernel="gaussian"),
            param_grid=dict(bandwidth=np.linspace(0.01, 2, 20)),
            cv=3
        )

        kde = grid.fit(dataset).best_estimator_

        xmin, ymin = dataset.min() - 2
        xmax, ymax = dataset.max() + 2

        xnum = (xmax - xmin) * 5
        ynum = (ymax - ymin) * 5

        Xgrid, Ygrid = np.meshgrid(
            np.linspace(xmin, xmax, num=xnum),
            np.linspace(ymin, ymax, num=ynum))

        Z = np.exp(
            kde.score_samples(np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T))

        zsort = np.sort(Z)[::-1]
        dens = zsort[np.argmax(np.cumsum(zsort) > Z.sum() * c)]

        contour_set = ax.contour(
            Xgrid, Ygrid, Z.reshape(Xgrid.shape), levels=[dens, ],
            colors=color, **kwds)

        if label:
            # fmt arg of clabel can be dict mapping level value to str
            fmt = {contour_set.levels[0]: combination_label(combinations)}
            ax.clabel(contour_set, fmt=fmt, inline=True, use_clabeltext=True,
                      **clabel_kwds)

        map_setup(ax)

        return ax

    def differ_by_n(self, n):
        """Lookup pairs of strains that differ by n positions.

        Args:
            n (int)

        Returns:
            set containing 2-tuples
        """
        keep = set()
        for a, b in itertools.combinations(self.seq_in_both.index, 2):
            df = self.seq_in_both
            if (df.loc[a, :] != df.loc[b, :]).sum() == n:
                keep.add((a, b))
        return keep

    def single_substitutions_one_random_aa(self, p, aa):
        """Find pairs of strains that differ only at position p, and where one
        has the amino acid aa at that position.

        Args:
            p (int): Position
            aa (str): Amino acid. Must be in amino_acids.

        Returns:
            list of pd.Series containing the amino acids
                and corresponding strain names.
        """
        keep = list()
        differ_by_1 = self.differ_by_n(1)
        for pair in differ_by_1:
            aas = self.seq_in_both.loc[pair, p]
            different = aas.unique().shape[0] > 1
            aa_present = (aa == aas).sum()
            nan_absent = aas.isnull().sum() < 1
            if different and aa_present and nan_absent:
                keep.append(aas)
        return keep

    def single_substitutions(self, sub, **kwds):
        """Find pairs of strains that differ by only the substitution 'sub'.

        Args:
            sub (tuple): ("N", 145, "K") like. First and last elements are
                strings referring to amino acids. Middle element is int
            exclude (list). Optional. Sequences only have to be identical at
                positions in this list.

        Returns:
            (set)
        """
        assert len(sub) == 3

        aa0 = sub[0]
        aa1 = sub[2]
        pos = sub[1]

        assert aa0 in amino_acids
        assert aa1 in amino_acids
        assert pos in self.seq_in_both.columns

        # Drop unwanted positions
        df = self.seq_in_both.drop(kwds.pop("exclude", list()), axis=1)

        # Drop the position of the substitution
        if pos in df.columns:
            df = df.drop(pos, axis=1)

        # Groupby all columns to get groups with the same sequence
        grouped = df.groupby(df.columns.tolist())

        # In each group find all combinations of sequences that differ by
        # aa0-aa1 at pos
        pairs = set()
        for name, group in grouped:

            # Only consider groups that contain more than one sequence
            if len(group) < 2:
                continue

            # Lookup amino acid for the group at pos
            group_pos = self.seq_in_both.loc[group.index, pos]

            # These sequences have aa0 at pos
            aa0_pos = group_pos[group_pos == aa0].index
            aa1_pos = group_pos[group_pos == aa1].index

            if aa0_pos.any() and aa1_pos.any():
                pairs.add(tuple(itertools.product(aa0_pos, aa1_pos)))

        return pairs

    def identical_sequences(self, positions=None):
        """Lookup strains with identical sequences.

        Args:
            positions (list containing ints). Only consider these positions.
                The default, None, considers all positions.

        Returns:
            (pd.DataFrame) containing the identical sequences.
        """
        identical = []
        if positions is None:
            positions = self.seq_in_both.columns.tolist()
        groupby = self.seq_in_both.groupby(positions)
        for sequence, strains in groupby.groups.items():
            if len(strains) > 1:
                identical.append(strains)
        return identical

    def error(self, positions=None):
        """
        Assuming in a perfect system (sequencing, laboratory, cartography)
        genetically identical strains should be in an identical position in the
        map. This method returns the distribution of pairwise distances between
        genetically identical strains.

        Some groups of genetically identical strains are larger than others.
        E.g. some groups can have ~30 strains, whilst lots of others have 2-5.
        This methods computes the mean and median pairwise distance for each
        group, so that one group does not dominate the distribution.

        (There are (30^2-30)/2 = 435 pairwise distances between 30 points,
        compared to (5^2-5)/2 = 10 for 10 points).

        Args:
            positions (list containing ints) Only consider these positions when
                looking for genetically identical strains. Default, None, uses
                all positions.

        Returns:
            (dict) containing mean and median distanes.
        """
        identical_sequences = self.identical_sequences(positions=positions)
        n = len(identical_sequences)
        means, medians = np.empty(n), np.empty(n)
        for i, names in enumerate(identical_sequences):
            # Each i is a list containing the indexes of identical sequences
            coords = self.coords_in_both.loc[names, :].values
            distances = spatial.distance.pdist(X=coords,
                                               metric="euclidean",
                                               p=2)
            means[i] = np.mean(distances)
            medians[i] = np.median(distances)
        return {"means": means, "medians": medians}

    def plot_error(self, positions=None):
        """Plot the distribution of means and median pairwise antigenic
        distances between genetically strains, calculated by self.error()

        Args:
            positions (list of ints). Only consider these positions when
                looking for genetically identical strains. Default, None, uses
                all positions.

        Returns:
            (matplotlib ax)
        """
        error = self.error(positions=positions)
        max_error = max(list(map(max, list(error.values()))))
        step = 0.5
        bins = np.arange(0, np.ceil(max_error) + step, step)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 10),
                               sharex=True, sharey=True)

        ax[0].hist(error["means"], bins=bins, ec="white")
        ax[0].set_title("Means")
        ax[0].set_ylabel("Frequency")

        ax[1].hist(error["medians"], bins=bins, ec="white")
        ax[1].set_title("Medians")
        ax[1].set_ylabel("Frequency")

        ax[1].set_xlabel("Antigenic distance (AU)")

        return ax

    def find_single_substitutions(self, cluster_diff_df, filename=None,
                                  **kwds):
        """
        Find strains that differ by one substitution out of the combinations
        defined in cluster_combinations

        Args:
            cluster_diff_df (pd.DataFrame): Specifies cluster difference
                substitutions. E.g.:

                            CA04 FU02
                        145    N    K
                        159    F    Y
                        189    N    S
                        226    I    V
                        227    P    S

            filename (str or None): If not None, save a plot with filename.
                Should be a format string with room to substitute in a label
                describing the substitutions found.

            **kwds passed to plot_strains_with_combinations.
        """
        clusters = cluster_diff_df.columns
        positions = cluster_diff_df.index

        for c0, c1 in itertools.permutations(clusters, 2):
            for p in positions:
                # Base combinations on c0
                combinations = cluster_diff_df.loc[:, c0].to_dict()

                # Alter position p to be c1-like
                combinations[p] = cluster_diff_df.loc[p, c1]

                # Find strains that have this combination of amino acids
                # and plot them
                if self.plot_strains_with_combinations(combinations, **kwds):
                    # The c1 amino acid
                    label = "{}-like_{}{}".format(c1, str(p), combinations[p])

                    # The c0 amino acids
                    combinations.pop(p, None)
                    c0_aas = combination_label(combinations)
                    label += ",_{}-like_{}".format(c0, c0_aas)

                    plt.title(label.replace("_", " "))
                    plt.tight_layout()
                    plt.savefig(filename.format(label).replace(",", "_"))
                    plt.close()

        # Finally, plot the viruses with the full combinations of each cluster
        for c in clusters:
            combinations = cluster_diff_df.loc[:, c].to_dict()
            aas = combination_label(combinations)
            if self.plot_strains_with_combinations(combinations, **kwds):
                label = "{}-like_{}".format(c, aas)
                plt.title(label.replace("_", " "))
                plt.tight_layout()
                plt.savefig(filename.format(label))
                plt.close()

    def plot(self, ax=None, **kwds):
        """Plot the map.

        Args:
            ax (matplotlib ax): Optional.
            **kwds passed to ax.scatter

        Returns:
            (matplotlib ax)
        """
        ax = plt.gca() if ax is None else ax
        x = self.coord_df.iloc[:, 0]
        y = self.coord_df.iloc[:, 1]
        ax.scatter(x, y, **kwds)
        map_setup(ax)
        return ax

    def find_double_substitutions(self, cluster_diff_df, filename=None,
                                  **kwds):
        """Find strains that differ by two substitutions out of the
        combinations defined in cluster_combinations.

        Args:
            cluster_diff_df (pd.DataFrame): Contains cluster difference
                substitutions. E.g.:
                            CA04 FU02
                        145    N    K
                        159    F    Y
                        189    N    S
                        226    I    V
            filename (str or None): If not None, save a plot with filename.
                Should be a format string with room to substitute in a label
                describing the substitutions found.
            **kwds passed to plot_strains_with_combinations.
        """
        clusters = cluster_diff_df.columns
        positions = cluster_diff_df.index

        for c0, c1 in itertools.permutations(clusters, 2):
            for p0, p1 in itertools.combinations(positions, 2):
                # Base combinations on c0
                combinations = cluster_diff_df.loc[:, c0].to_dict()

                # Alter positions to be c1-like
                for p in p0, p1:
                    combinations[p] = cluster_diff_df.loc[p, c1]

                # Find strains that have this combination of amino acids
                # and plot them
                if self.plot_strains_with_combinations(combinations, **kwds):
                    # Label
                    subs = "+".join(sorted("{}{}".format(str(p),
                                                         combinations[p])
                                           for p in (p0, p1)))
                    label = "Double_{}-like_{}".format(c1, subs)
                    plt.title(label.replace("_", " "))
                    plt.tight_layout()
                    plt.savefig(filename.format(label))
                    plt.close()


class OrderedMapSeq(MapSeq):

    def __init__(self, seq_df, coord_df):
        """Like MapSeq, but the order of the indexes of the dataframes
        containing the coordinates and sequences are identical.

        Notes:
            Positions with no amino acid diversity are removed.

        Args:
            seq_df (pd.DataFrame): Indexes are strains. Columns are amino acid
                positions.
            coord_df (pd.DataFrame): Indexes are strains. Columns are "x" and
                "y" coordinates.

        Attributes:
            coord (mapdeduce.dataframes.CoordDf): Contains coordinates.
            seq (mapdeduce.dataframes.SeqDf): Contains sequences.
        """
        super().__init__(seq_df, coord_df)

        # Join the two dataframes, so they share indexes
        combined = self.coords_in_both.join(self.seq_in_both)

        # Remove strains that have any NaN entries
        mask = combined.notnull().any(axis=1)
        n_with_nan = (~mask).sum()
        if n_with_nan:
            tqdm.write("Removed {} strains with NaN values".format(n_with_nan))
            combined = combined[mask]

        phenotypes = self.coords_in_both.columns

        self.coord = CoordDf(combined.loc[:, phenotypes])
        self.seqs = SeqDf(combined.drop(phenotypes, axis=1))

    def filter(self, patch=None, plot=True, remove_invariant=True,
               get_dummies=True, merge_duplicate_dummies=False,
               rename_idx=False):
        """Remove data where the x y coordinates are outside a matplotlib
        patch.

        Notes:
            Updates self.coord and self.seq inplace.

        Args:
            patch (matplotlib.patches.Patch): Remove strains not contained
                in patch.
            plot (bool): Show points that have been included / excluded
                and the patch. (Only if a patch is specified).
            remove_invariant (bool): Remove sequence positions that only
                contain a single amino acid.
            get_dummies (bool): Attach dummy variable representation of the
                sequences.
            merge_duplicate_dummies (bool): Merge dummy variables that have
                the same profile (identical for all strains).
            rename_idx (bool): Rename strains in the format strain-X where X
                is an integer. This is necessary for merging duplicate dummies
                if there are duplicate strain names (which can occur if a
                strain was titrated multiple times). Attaches a strain_names
                attribute to self which is a dict containing the new name to
                old name mapping.

        Returns:
            (matplotlib ax)
        """
        # Strain removal operations first
        if patch is not None:

            if plot:
                ax = self.coord.df.plot.scatter(
                    x="x", y="y", label="All points", c="black", s=5)

            mask = self.coord.points_in_patch(patch=patch)

            self.coord.df = self.coord.df[mask]
            self.seqs.df = self.seqs.df[mask]

        if remove_invariant:
            self.seqs.remove_invariant(inplace=True)

        if rename_idx:
            old_idx = self.coord.df.index
            new_idx = ["strain-{}".format(i) for i in range(old_idx.shape[0])]
            self.strain_names = dict(list(zip(new_idx, old_idx)))
            self.coord.df.index = new_idx
            self.seqs.df.index = new_idx

        if get_dummies:
            self.seqs.get_dummies(inplace=True)

        if merge_duplicate_dummies:
            self.seqs.merge_duplicate_dummies(inplace=True)

        if plot and patch is not None:
            self.coord.df.plot.scatter(x="x", y="y", label="Included", ax=ax)
            map_setup(ax)
            return ax
