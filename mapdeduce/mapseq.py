"""Contains the main class to represent maps with sequences."""

import gzip
import itertools
import json
import warnings
from functools import reduce
from io import StringIO
from operator import and_
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import spm1d
import tqdm
from matplotlib.collections import LineCollection
from scipy import spatial

from .data import amino_acids
from .dataframes import CoordDf, SeqDf
from .helper import is_not_amino_acid
from .munging import handle_duplicate_sequences
from .plotting import (
    add_ellipses,
    amino_acid_colors,
    combination_label,
    make_ax_a_map,
    point_size,
    setup_ax,
)


class MapSeq:
    def __init__(
        self,
        seq_df: pd.DataFrame,
        coord_df: pd.DataFrame,
        map: Optional[int] = None,
    ) -> None:
        """
        An antigenic map with sequence data.

        Args:
            seq_df (pd.DataFrame): Indexes are strains. Columns are amino acid
                positions.
            coord_df (pd.DataFrame): Indexes are strains. Columns are "x" and
                "y" coordinates.
            map (int): 2009 or 2017. Optional. Specify a known map to configure
                plotting boundaries.
        """
        self.all_seqs = handle_duplicate_sequences(seq_df)
        self.all_coords = coord_df.copy()
        self.map = map

        # Replace any unknown amino acids with NaN
        cond = self.all_seqs.map(lambda x: x not in amino_acids)
        self.all_seqs.mask(cond, inplace=True)

        # Remove any rows that contain NaN coords
        mask = self.all_coords.notnull().any(axis=1)
        self.all_coords = self.all_coords.loc[mask, :]

        # Strains in the sequences and coordinates dataframes
        self.strains_in_seq = set(self.all_seqs.index)
        self.strains_in_coords = set(self.all_coords.index)

        # Strains in both sequence and coordinate dataframes
        self.strains_in_both = self.strains_in_seq & self.strains_in_coords

        # Strains exclusively in sequence df
        self.strains_excl_to_seq = self.strains_in_seq - self.strains_in_coords

        # Strains exclusively in coords df
        self.strains_excl_to_coords = (
            self.strains_in_coords - self.strains_in_seq
        )
        self.coords_excl_to_coords = self.all_coords.loc[
            list(self.strains_excl_to_coords)
        ]

        # Coordinates and sequences of strains in both
        self.seq_in_both = self.all_seqs.loc[list(self.strains_in_both)]
        self.coords_in_both = self.all_coords.loc[list(self.strains_in_both)]

        # Find variant positions (i.e. which positions have substitutions)
        self.variant_positions = set()
        for p in self.seq_in_both.columns:
            if len(self.seq_in_both.loc[:, p].unique()) != 1:
                self.variant_positions.add(p)

    def to_disk(self, path: str) -> None:
        """
        Save MapSeq to a compressed JSON file.

        Args:
            path: Path to save the compressed JSON file.
        """
        data = {
            "version": 1,
            "map": self.map,
            "seq_df": json.loads(self.all_seqs.to_json(orient="split")),
            "coord_df": json.loads(self.all_coords.to_json(orient="split")),
        }

        with gzip.open(path, "wt") as f:
            json.dump(data, f, separators=(",", ":"))

    @classmethod
    def from_disk(cls, path: str) -> "MapSeq":
        """
        Load MapSeq from a compressed JSON file.

        Args:
            path: Path to the compressed JSON file.

        Returns:
            MapSeq instance.
        """
        with gzip.open(path, "rt") as f:
            data = json.load(f)

        seq_df = pd.read_json(
            StringIO(json.dumps(data["seq_df"])), orient="split"
        )
        coord_df = pd.read_json(
            StringIO(json.dumps(data["coord_df"])), orient="split"
        )

        return cls(seq_df=seq_df, coord_df=coord_df, map=data.get("map"))

    def scatter_with_without(self, **kwds) -> plt.Axes:
        """
        Plot indicating which antigens do and do not have sequences.

        Args:
            **kwds. Passed to pd.DataFrame.plot.scatter
        """
        ax = plt.gca()
        kwds = dict(ax=ax, x="x", y="y")
        n_without_sequence = self.coords_excl_to_coords.shape[0]
        self.coords_excl_to_coords.plot.scatter(
            color="darkgrey",
            label="Without sequence ({})".format(n_without_sequence),
            **kwds,
        )
        n_with_sequence = self.coords_in_both.shape[0]
        self.coords_in_both.plot.scatter(
            color="#b72467",
            label="With sequence ({})".format(n_with_sequence),
            **kwds,
        )
        setup_ax(map=self.map)
        return ax

    def variant_proportions(self, p: int) -> pd.Series:
        """
        Of antigens with sequences, compute the proportion of amino acids
        at each position, p.

        Args:
            p (int): HA position

        Returns:
            pd.Series
        """
        series = self.seq_in_both.loc[:, p]
        value_counts = series.value_counts()
        return (value_counts / value_counts.sum()).sort_values()

    def scatter_colored_by_amino_acid(
        self, p: int, randomz: bool = True, ellipses: bool = True, **kwargs
    ) -> plt.Axes:
        """
        Plot map colored by amino acids at position p.

        Args:
            p (int): HA position
            randomz (bool): Given points random positions in z. This is
                slower because marks have to plotted individually.
            ellipses (bool). Demark clusters with ellipses.
            **kwargs. Passed to ax.scatter for the colored strains.

        Returns:
            matplotlib ax
        """
        ax = plt.gca()

        # Antigens without a known sequence
        if not self.coords_excl_to_coords.empty:
            self.coords_excl_to_coords.plot.scatter(
                ax=ax,
                x="x",
                y="y",
                s=5,
                color="darkgrey",
                label="Unknown sequence",
            )

        # Antigens with a known sequence
        kwds = dict(
            lw=1,
            edgecolor="white",
            s=3 * point_size(self.seq_in_both.shape[0]),
            **kwargs,
        )

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
                    label=label, x="x", y="y", ax=ax, **kwds
                )

        # Set axis limits from all coordinates so plots for different sites
        # have identical extent, even if some strains have NaN as an amino acid
        # at this position.
        all_x = self.all_coords["x"]
        all_y = self.all_coords["y"]
        ax.set_xlim(all_x.min(), all_x.max())
        ax.set_ylim(all_y.min(), all_y.max())

        if ellipses:
            add_ellipses(self.map)

        setup_ax(self.map)

        ax.legend()
        ax.text(x=0.5, y=1, s=p, fontsize=25, va="top", transform=ax.transAxes)

        return ax

    def scatter_single_substitution(
        self,
        sub: tuple,
        ellipses: bool = True,
        filename: str = None,
        connecting_lines: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot map colored by amino acids at position p.

        Args:
            sub (tuple): Substitution. E.g. ("N", 145, "K")
            ellipses (bool): Demark clusters with ellipses.
            filename (str): Passed to plt.savefig. None does not save figure.
            connecting_lines (bool): Plot lines between each points that
                differ by the substitution.
            **kwargs. Passed to self.single_substitutions. Use to restrict
                to particular positions.
        """
        # Strains that differ by only the substitution sub
        combinations = self.single_substitutions(sub, **kwargs)

        label = "".join(map(str, sub))

        if not combinations:
            print("No pairs of strains with {}".format(label))
            return

        # Collect x, y of points to plot, and lines between
        aas = sub[0], sub[2]
        for i, pairs in enumerate(combinations):
            fig, ax = plt.subplots()

            # Antigens without a known sequence
            self.coords_excl_to_coords.plot.scatter(
                ax=ax,
                x="x",
                y="y",
                s=5,
                color="lightgrey",
                label="Unknown sequence",
            )

            # Antigens with a known sequence
            self.coords_in_both.plot.scatter(
                ax=ax,
                x="x",
                y="y",
                s=10,
                color="darkgrey",
                label="Known sequence",
            )

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
                    x="x",
                    y="y",
                    s=150,
                    c=amino_acid_colors[aas[j]],
                    edgecolor="white",
                    linewidth=1,
                    zorder=20,
                    ax=ax,
                    label="{}{}".format(aas[j], sub[1]),
                )

            # Compute Hotelling's T-squared statistic on the two samples
            if len(samples[0]) > 1 and len(samples[1]) > 1:
                h = spm1d.stats.hotellings2(*samples)
                h_report = "p = {:.2E}\nz = {:.3f}\ndf = {:d}, {:d}".format(
                    h.inference().p, h.z, *list(map(int, h.df))
                )
            else:
                h_report = "[Insufficient data]"

            ax.text(
                x=0,
                y=1,
                ha="left",
                va="top",
                transform=ax.transAxes,
                s=r"2 sample Hotelling's T$^2$" + "\n" + h_report,
            )

            if connecting_lines:
                for pair in pairs:
                    # May be multiple strains with the same name that have
                    # different coordinates. Plot lines between all
                    # combinations
                    pair_coords = self.coords_in_both.loc[pair, :]

                    segments = []

                    if pair_coords.shape[0] > 2:
                        # Some strains are repeated in the map
                        # Look for all combinations of groups of strains
                        groups = [
                            group
                            for _, group in pair_coords.groupby(
                                pair_coords.index
                            )
                        ]

                        for _, series_1 in groups[0].iterrows():
                            for _, series_2 in groups[1].iterrows():
                                segments.append(
                                    (series_1.values, series_2.values)
                                )

                    elif pair_coords.shape[0] == 2:
                        segments.append(pair_coords.values)

                    else:
                        raise ValueError(
                            "This 'pair' indexes less that 2 "
                            "strains\n{}".format(pair)
                        )

                    ax.add_collection(
                        LineCollection(
                            segments=segments,
                            lw=1,
                            color="black",
                            alpha=alpha,
                            zorder=10,
                            label="",
                        )
                    )

            if ellipses:
                add_ellipses(self.map)

            setup_ax(self.map)
            ax.legend()
            title = "{} ({})".format(label, i + 1)
            ax.set_title(title)
            if filename is not None:
                plt.tight_layout()
                plt.savefig(filename.format(title.replace(" ", "")))

    def scatter_variant_positions_colored_by_amino_acid(
        self, filename: str, **kwds
    ) -> None:
        """
        Call scatter_colored_by_amino_acid for all variant positions

        Args:
            filename (str): A format string with one field to fill in. Each
                position will be substituted in. E.g.:
                "img/melb-h3-2009-coloured-by-pos/{}.png"

            **kwds: Keyword arguments passed to scatter_colored_by_amino_acid
        """
        # Save text file containing html of the variant positions. 1 sorted by
        # primary structure the second by most common variant
        sorted_primary = sorted(self.variant_positions)
        sorted_most_common = (
            self.variant_positions_sorted_by_most_common_variant()
        )

        img_tag = '<img src="{}" class="map" />\n'

        with open(".by_primary.txt", "w") as fobj:
            for pos in sorted_primary:
                fobj.write(img_tag.format(filename.format(pos)))

        with open(".by_most_common.txt", "w") as fobj:
            for pos in sorted_most_common:
                fobj.write(img_tag.format(filename.format(pos)))

        print("Wrote .by_primary.txt and .by_most_common.txt.")

        n = len(self.variant_positions)
        print("There are {} variant positions.".format(n))
        print("Doing", end="")

        for pos in self.variant_positions:
            print("{}".format(pos), end="")

            plt.subplots()
            self.scatter_colored_by_amino_acid(pos, **kwds)
            make_ax_a_map()
            plt.tight_layout()
            plt.savefig(filename.format(pos), bbox_inches="tight")
            plt.close()

    def variant_positions_sorted_by_most_common_variant(self) -> tuple:
        """
        Lookup variant positions.

        Returns:
            tuple. Contains variant positions, sorted by most common variant.
        """
        proportions_all_variants = list(
            map(self.variant_proportions, self.variant_positions)
        )
        proportions_all_variants_sorted = sorted(
            proportions_all_variants, key=lambda x: x[-1]
        )
        return tuple(p.name for p in proportions_all_variants_sorted)

    def strains_with_combinations(
        self,
        combinations: dict,
        without: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Lookup strains that have combinations of amino acids at particular
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
                raise ValueError(f"Position {k} is unknown.")

        masks = (
            self.seq_in_both.loc[:, k] == v for k, v in combinations.items()
        )

        mask = reduce(and_, masks)

        if without:
            mask = ~mask

        df = self.seq_in_both[mask]

        if df.empty:
            label = "+".join(
                sorted("{}{}".format(k, v) for k, v in combinations.items())
            )

            if verbose:
                warnings.warn("No strains with {}".format(label))

        return df

    def duplicate_sequences(
        self, **kwargs
    ) -> pd.core.groupby.DataFrameGroupBy:
        """
        Find groups of duplicate sequences.

        Any element in self.seq_in_both that is not one of the standard 20
        1-letter amino acid abbreviations is ignored. (It is treated as NaN,
        see: http://pandas.pydata.org/pandas-docs/stable/
            missing_data.html#na-values-in-groupby)

        Args:

            positions (iterable of ints). Optional. Lookup groups of sequences
                identical at these positions. Default is all positions.

        Returns:
            pd.core.groupby.DataFrameGroupBy
        """
        positions = kwargs.pop("positions", self.seq_in_both.columns.tolist())
        df = self.seq_in_both
        return df.mask(df.map(is_not_amino_acid)).groupby(positions)

    def plot_strains_with_combinations(
        self,
        combinations: dict[int, str],
        without: bool = False,
        plot_other: bool = True,
        **kwargs,
    ):
        """
        Plot a map highlighting strains with combinations of amino acids
        at particular positions.

        Args:
            combinations (dict): Keys are positions, values are amino
                acids:  E.g. {145: "N", 189: "S"}
            without (bool): Plot strains without the combination.
            plot_other (bool): Plot other antigens (those without the
                combinations).
            **kwargs: Optional keywords passed to the scatter function of the
                strains with particular combinations.
        """
        df = self.strains_with_combinations(
            combinations=combinations, without=without
        )

        label = "+".join(
            sorted("{}{}".format(k, v) for k, v in combinations.items())
        )

        if without:
            label = "Without " + label

        if df.empty:
            print("No strains with {}".format(label))

        else:
            ax = kwargs.pop("ax", False)

            if not ax:
                fig, ax = plt.subplots()

            strains = df.index
            print("{} strains with {}".format(len(strains), label))

            if plot_other:
                self.all_coords.plot.scatter(
                    x="x", y="y", s=10, color="darkgrey", ax=ax
                )

            # self.all_coords.loc[strains, :] may be a Series, hence ax.scatter
            plt.scatter(
                self.all_coords.loc[strains, "x"],
                self.all_coords.loc[strains, "y"],
                label=label,
                **kwargs,
            )
            plt.legend()

            if self.map:
                add_ellipses(self.map)
                setup_ax(self.map)

    def plot_strains_with_combinations_kde(
        self,
        combinations: dict[int, str],
        c: float = 0.9,
        color: str = "black",
        **kwargs,
    ):
        """
        Plot the contour corresponding to the region that contains c
        percent of the density of a KDE over strains with combinations of
        amino acid polymorphisms specified in combinations.

        Args:
            combinations (dict): Dictionary specifying combinations. E.g.:
                {145: "N", 133: "D"}
            c (float / int)
            color (matplotlib colour): Colour to plot the contour line
            **kwargs. Passed to plt.contour
        """
        df = self.strains_with_combinations(combinations)
        strains = df.index

        dataset = self.all_coords.loc[strains, :]

        grid = sklearn.model_selection.GridSearchCV(
            estimator=sklearn.neighbors.KernelDensity(kernel="gaussian"),
            param_grid=dict(bandwidth=np.linspace(0.01, 2, 20)),
            cv=3,
        )

        kde = grid.fit(dataset).best_estimator_

        xmin, ymin = dataset.min() - 2
        xmax, ymax = dataset.max() + 2

        xnum = (xmax - xmin) * 5
        ynum = (ymax - ymin) * 5

        Xgrid, Ygrid = np.meshgrid(
            np.linspace(xmin, xmax, num=xnum),
            np.linspace(ymin, ymax, num=ynum),
        )

        Z = np.exp(
            kde.score_samples(np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T)
        )

        zsort = np.sort(Z)[::-1]
        dens = zsort[np.argmax(np.cumsum(zsort) > Z.sum() * c)]

        plt.contour(
            Xgrid,
            Ygrid,
            Z.reshape(Xgrid.shape),
            levels=[
                dens,
            ],
            colors=color,
            **kwargs,
        )

    def differ_by_n(self, n: int) -> set[tuple[str, str]]:
        """
        Lookup pairs of strains that differ by n positions.

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

    def single_substitutions_one_random_aa(
        self, pos: int, aa: str
    ) -> list[pd.Series]:
        """
        Find pairs of strains that differ only at position p, and where one
        has the amino acid aa at that position.

        Args:
            pos (int): Position.
            aa (str): Amino acid. Must be in amino_acids.

        Returns:
            list of pd.Series containing the amino acids
                and corresponding strain names.
        """
        keep = list()
        differ_by_1 = self.differ_by_n(1)
        for pair in differ_by_1:
            aas = self.seq_in_both.loc[pair, pos]
            different = aas.unique().shape[0] > 1
            aa_present = (aa == aas).sum()
            nan_absent = aas.isnull().sum() < 1
            if different and aa_present and nan_absent:
                keep.append(aas)
        return keep

    def single_substitutions(self, sub: tuple, **kwargs) -> set:
        """
        Find pairs of strains that differ by only the substitution 'sub'.

        Args:
            sub (tuple): ("N", 145, "K") like. First and last elements are
                strings referring to amino acids. Middle element is int
            exclude (list). Optional. Sequences only have to be identical at
                positions in this list.

        Returns:
            set
        """
        assert len(sub) == 3

        aa0 = sub[0]
        aa1 = sub[2]
        pos = sub[1]

        assert aa0 in amino_acids
        assert aa1 in amino_acids
        assert pos in self.seq_in_both.columns

        # Drop unwanted positions
        df = self.seq_in_both.drop(kwargs.pop("exclude", list()), axis=1)

        # Drop the position of the substitution
        if pos in df.columns:
            df = df.drop(pos, axis=1)

        # Groupby all columns to get groups with the same sequence
        grouped = df.groupby(df.columns.tolist())

        # In each group find all combinations of sequences that differ by
        # aa0-aa1 at pos
        pairs = set()
        for _, group in grouped:

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

    def identical_sequences(
        self, positions: Optional[list[int]] = None
    ) -> list:
        """
        Lookup strains with identical sequences.

        Args:
            positions (list containing ints). Only consider these positions.
                The default, None, considers all positions.

        Returns:
            list containing the identical sequences.
        """
        identical = []
        if positions is None:
            positions = self.seq_in_both.columns.tolist()
        groupby = self.seq_in_both.groupby(positions)
        for _, strains in groupby.groups.items():
            if len(strains) > 1:
                identical.append(strains)
        return identical

    def error(self, positions: Optional[list[int]] = None) -> dict:
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
            dict containing mean and median distances.
        """
        identical_sequences = self.identical_sequences(positions=positions)
        n = len(identical_sequences)
        means, medians = np.empty(n), np.empty(n)
        for i, names in enumerate(identical_sequences):
            # Each i is a list containing the indexes of identical sequences
            coords = self.coords_in_both.loc[names, :].values
            distances = spatial.distance.pdist(
                X=coords, metric="euclidean", p=2
            )
            means[i] = np.mean(distances)
            medians[i] = np.median(distances)
        return {"means": means, "medians": medians}

    def plot_error(self, positions: Optional[list[int]] = None) -> plt.Axes:
        """
        Plot the distribution of means and median pairwise antigenic
        distances between genetically strains, calculated by self.error()

        Args:
            positions (list of ints). Only consider these positions when
                looking for genetically identical strains. Default, None, uses
                all positions.

        Returns:
            matplotlib ax
        """
        error = self.error(positions=positions)
        max_error = max(list(map(max, list(error.values()))))
        step = 0.5
        bins = np.arange(0, np.ceil(max_error) + step, step)

        _, ax = plt.subplots(
            nrows=2, ncols=1, figsize=(7, 10), sharex=True, sharey=True
        )

        ax[0].hist(error["means"], bins=bins, ec="white")
        ax[0].set_title("Means")
        ax[0].set_ylabel("Frequency")

        ax[1].hist(error["medians"], bins=bins, ec="white")
        ax[1].set_title("Medians")
        ax[1].set_ylabel("Frequency")

        ax[1].set_xlabel("Antigenic distance (AU)")

        return ax

    def find_single_substitutions(
        self,
        cluster_diff_df: pd.DataFrame,
        filename: Optional[str] = None,
        **kwargs,
    ):
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

            **kwargs: Passed to plot_strains_with_combinations
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
                if self.plot_strains_with_combinations(combinations, **kwargs):
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
            if self.plot_strains_with_combinations(combinations, **kwargs):
                label = "{}-like_{}".format(c, aas)
                plt.title(label.replace("_", " "))
                plt.tight_layout()
                plt.savefig(filename.format(label))
                plt.close()

    def find_double_substitutions(
        self,
        cluster_diff_df: pd.DataFrame,
        filename: Optional[str] = None,
        **kwargs,
    ):
        """
        Find strains that differ by two substitutions out of the
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
            **kwargs passed to plot_strains_with_combinations
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
                if self.plot_strains_with_combinations(combinations, **kwargs):
                    # Label
                    subs = "+".join(
                        sorted(
                            "{}{}".format(str(p), combinations[p])
                            for p in (p0, p1)
                        )
                    )
                    label = "Double_{}-like_{}".format(c1, subs)
                    plt.title(label.replace("_", " "))
                    plt.tight_layout()
                    plt.savefig(filename.format(label))
                    plt.close()


class OrderedMapSeq(MapSeq):
    def __init__(self, seq_df: pd.DataFrame, coord_df: pd.DataFrame) -> None:
        """
        Like MapSeq, but the order of the indexes of the dataframes
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

    def filter(
        self,
        patch: Optional[matplotlib.patches.Patch] = None,
        plot: bool = True,
        remove_invariant: bool = True,
        get_dummies: bool = True,
        merge_duplicate_dummies: bool = False,
        prune_collinear_dummies: Optional[float] = None,
        rename_idx: bool = False,
    ):
        """
        Remove data where the x y coordinates are outside a matplotlib
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
            prune_collinear_dummies (float or None): If set, prune
                near-collinear dummies (r² > threshold) after merging
                duplicates. Requires get_dummies=True.
            rename_idx (bool): Rename strains in the format strain-X where X
                is an integer. This is necessary for merging duplicate dummies
                if there are duplicate strain names (which can occur if a
                strain was titrated multiple times). Attaches a strain_names
                attribute to self which is a dict containing the new name to
                old name mapping.

        Returns:
            matplotlib ax
        """
        # Strain removal operations first
        if patch is not None:
            if plot:
                ax = self.coord.df.plot.scatter(
                    x="x", y="y", label="All points", c="black", s=5
                )

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

        if prune_collinear_dummies is not None:
            self.seqs.prune_collinear_dummies(
                threshold=prune_collinear_dummies, inplace=True
            )

        if plot and patch is not None:
            self.coord.df.plot.scatter(x="x", y="y", label="Included", ax=ax)

            make_ax_a_map(ax)

            return ax
