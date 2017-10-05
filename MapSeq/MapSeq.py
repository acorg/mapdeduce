"""Contains the main class to represent maps with sequences."""

import matplotlib.pyplot as plt

import numpy as np

from operator import and_
import itertools

from plotting import setup_ax, amino_acid_colors, add_ellipses


class MapSeq(object):
    """Class for handling antigenic maps with sequence data."""

    def __init__(self, seq_df, coord_df, map):
        """
        @param seq_df: pd.DataFrame. Indexes are strains. Columns are
            amino acid positions.
        @param coord_df: pd.DataFrame. Indexes are strains. Columns are
            "x" and "y" coordinates.
        @param map: Int. 2009 or 2017. Specify which map is being worked on
            to setup plotting cluster boundaries and plot ranges.
        """
        self.all_seqs = seq_df.copy()
        self.all_coords = coord_df.copy()
        self.map = map

        # Strains in the sequences and coordinates dataframes
        self.strains_in_seq = set(self.all_seqs.index)
        self.strains_in_coords = set(self.all_coords.index)

        # Strains in both sequence and coordinate dataframes
        self.strains_in_both = self.strains_in_seq & self.strains_in_coords

        # Strains exclusively in sequence df
        self.strains_excl_to_seq = self.strains_in_seq - self.strains_in_coords

        # Strains exclusively in coords df
        self.strains_excl_to_coords = \
            self.strains_in_coords - self.strains_in_seq
        self.coords_excl_to_coords = \
            self.all_coords.loc[self.strains_excl_to_coords, :]

        # Coordinates and sequences of strains in both
        self.seq_in_both = self.all_seqs.loc[self.strains_in_both, :]
        self.coords_in_both = self.all_coords.loc[self.strains_in_both, :]

        # Find variant positions (i.e. which positions have substitutions)
        self.variant_positions = set()
        for p in self.seq_in_both.columns:
            if len(self.seq_in_both.loc[:, p].unique()) != 1:
                self.variant_positions.add(p)

    def scatter_with_without(self):
        """Plot indicating which antigens do and do not have sequences."""
        ax = plt.gca()
        kwds = dict(ax=ax, x="x", y="y")
        self.coords_excl_to_coords.plot.scatter(color="darkgrey",
                                                label="Without sequence",
                                                **kwds)
        self.coords_in_both.plot.scatter(color="#b72467",
                                         label="With sequence",
                                         **kwds)
        setup_ax(map=self.map)
        return ax

    def variant_proportions(self, p):
        """
        Return the proportion of variants at position p from the antigens that
        have sequences.

        @param p: Int. HA position
        """
        series = self.seq_in_both.loc[:, p]
        value_counts = series.value_counts()
        return (value_counts / value_counts.sum()).sort_values()

    def scatter_colored_by_amino_acid(self, p, randomz=True, ellipses=True):
        """
        Plot map colored by amino acids at position p.

        @param p. Int. HA position
        @param randomz. Bool. Given points random positions in z. This is
            slower because marks have to plotted individually.
        @param ellipses. Bool. Demark clusters with ellipses.
        """
        ax = plt.gca()

        # Plot antigens without a known sequence
        self.coords_excl_to_coords.plot.scatter(ax=ax,
                                                x="x",
                                                y="y",
                                                s=5,
                                                color="darkgrey",
                                                label="Unknown sequence")

        # Plot antigens with a known sequence
        kwds = dict(edgecolor="white", s=100)
        proportions = self.variant_proportions(p=p) * 100

        seq_grouped = self.seq_in_both.groupby(p)
        for amino_acid, seq_group in seq_grouped:
            coord_group = self.coords_in_both.loc[seq_group.index, :]
            kwds["color"] = amino_acid_colors[amino_acid]
            label = "{} {:.1f}%".format(amino_acid, proportions[amino_acid])

            if randomz:
                zorders = np.random.rand(seq_group.shape[0]) + 5
                for z, (strain, row) in zip(zorders, coord_group.iterrows()):
                    ax.scatter(x=row["x"], y=row["y"], zorder=z, **kwds)

                # Plot the final point twice, but add a label for the group
                ax.scatter(x=row["x"], y=row["y"], label=label, **kwds)

            else:
                coord_group.plot.scatter(label=label, **kwds)

        if ellipses:
            add_ellipses(self.map)

        setup_ax(self.map)
        ax.legend()
        ax.text(0.5, 1, p, fontsize=25, va="top", transform=ax.transAxes)

        return ax

    def scatter_variant_positions_colored_by_amino_acid(self, filename,
                                                        **kwds):
        """
        Call scatter_colored_by_amino_acid for all variant positions

        @param filename: Format string with one field to fill in. Each
            position will be substituted in. E.g.:
            "img/melb-h3-2009-coloured-by-pos/{}.png"
        @param kwds: Passed to scatter_colored_by_amino_acid
        """
        for p in self.variant_positions:
            fig, ax = plt.subplots(figsize=(6.4, 7))
            self.scatter_colored_by_amino_acid(p, **kwds)
            plt.tight_layout()
            plt.savefig(filename.format(p), dpi=100)
            plt.close()

    def variant_positions_sorted_by_most_common_variant(self):
        """
        Return a tuple of the variant positions, sorted by the most common
        variant for each, ascending

        @returns Tuple
        """
        proportions_all_variants = map(self.variant_proportions,
                                       self.variant_positions)
        proportions_all_variants_sorted = sorted(proportions_all_variants,
                                                 key=lambda x: x[-1])
        return tuple(p.name for p in proportions_all_variants_sorted)

    def strains_with_combinations(self, combinations):
        """
        Return strains that have combinations of amino acids at particular
        positions.

        @param combinations. Dict. Keys are positions, values are amino
            acids:  E.g. {145: "N", 189: "S"}
        @returns pd.DataFrame. Containing strains with the amino acid
            combinations.
        """
        masks = (self.seq_in_both.loc[:, k] == v
                 for k, v in combinations.iteritems())
        return self.seq_in_both[reduce(and_, masks)]

    def plot_strains_with_combinations(self, combinations, **kwargs):
        """
        Plot a map highlighting strains with combinations of amino acids
        at particular positions.

        @param combinations. Dict. Keys are positions, values are amino
            acids:  E.g. {145: "N", 189: "S"}
        @param **kwargs. Passed to the scatter function of the strains with
            particular combinations.
        """
        df = self.strains_with_combinations(combinations)
        label = "+".join(sorted("{}{}".format(k, v)
                                for k, v in combinations.iteritems()))
        if df.empty:
            print "No strains with {}".format(label)
        else:
            strains = df.index
            print "{} strains with {}".format(len(strains), label)
            ax = self.all_coords.plot.scatter(x="x", y="y", s=5,
                                              color="darkgrey")

            # self.all_coords.loc[strains, :] may be a Series, hence ax.scatter
            ax.scatter(self.all_coords.loc[strains, "x"],
                       self.all_coords.loc[strains, "y"],
                       label=label,
                       **kwargs)
            ax.legend()
            add_ellipses(self.map)
            setup_ax(self.map)
            return ax

    def find_single_substitutions(self, cluster_diff_df, filename=None,
                                  **kwargs):
        """
        Find strains that differ by one substitution out of the combinations
        defined in cluster_combinations

        @param: cluster_diff_df: df containing cluster difference
            substitutions. E.g.:

                            CA04 FU02
                        145    N    K
                        159    F    Y
                        189    N    S
                        226    I    V
                        227    P    S

        @param: filename. If not None, save a plot with filename. Should be
            a format string with room to substitute in a label describing
            the substitutions found.
        @param: kwargs. Passed to plot_strains_with_combinations
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
                    # Label
                    label = "Single_{}-like_{}{}".format(c1, str(p),
                                                         combinations[p])
                    plt.title(label.replace("_", " "))
                    plt.tight_layout()
                    plt.savefig(filename.format(label))
                    plt.close()

    def find_double_substitutions(self, cluster_diff_df, filename=None,
                                  **kwargs):
        """
        Find strains that differ by two substitutions out of the combinations
        defined in cluster_combinations

        @param: cluster_diff_df: df containing cluster difference
            substitutions. E.g.:

                            CA04 FU02
                        145    N    K
                        159    F    Y
                        189    N    S
                        226    I    V

        @param: filename. If not None, save a plot with filename. Should be
            a format string with room to substitute in a label describing
            the substitutions found.
        @param: kwargs. Passed to plot_strains_with_combinations
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
                    subs = "+".join(sorted("{}{}".format(str(p),
                                                         combinations[p])
                                           for p in (p0, p1)))
                    label = "Double_{}-like_{}".format(c1, subs)
                    plt.title(label.replace("_", " "))
                    plt.tight_layout()
                    plt.savefig(filename.format(label))
                    plt.close()
