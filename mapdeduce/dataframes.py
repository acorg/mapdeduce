"""Classes for handling DataFrames containing coordinates and sequences"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.preprocessing import quantile_transform

from .helper import expand_sequences, site_consensus
from .munging import df_from_fasta


def columns_at_positions(columns, positions: list[int]) -> set[str]:
    """
    Return column names that involve any of the given positions.

    Column names may be singles (e.g. 135K), compound by identity
    (e.g. 7D|135K|265E), or compound by collinearity
    (e.g. 145K|155S~189K). Returns the full column name if any
    constituent SNP is at a position in *positions*.

    @param columns: Iterable of column name strings.
    @param positions: List of positions.
    """
    result = set()

    for col in columns:
        found = False
        for collinear_group in col.split("~"):
            for aap in collinear_group.split("|"):
                pos = int(aap.lstrip("-")[:-1])
                if pos in positions:
                    result.add(col)
                    found = True
                    break
            if found:
                break

    return result


class CoordDf:
    """Class for x, y coordinate data."""

    def __init__(self, df: pd.DataFrame) -> None:
        """@param df: pd.DataFrame. Must contain x and y columns."""
        self.df = df

    def __repr__(self) -> str:
        return "CoordDf with {} samples and {} dimensions.".format(
            *self.df.shape
        )

    def rotate(self, a: float, inplace: bool = True) -> Optional["CoordDf"]:
        """
        Rotate points specified by the xy coordinates in the DataFrame
        a degrees around the origin anticlockwise.

        @param a: Number. Arc degrees to rotate the dataframe by
        @param inplace: Bool. Rotate the data inplace, or return a rotated
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

    def pca_rotate(self, inplace: bool = True) -> Optional["CoordDf"]:
        """
        Rotate the coordinates along first and second principal components.

        @param inplace: Bool. Rotate the data inplace, or return a rotated
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

    def quantile_transform(self, inplace: bool = True) -> Optional["CoordDf"]:
        """
        Transform features using quantile information. See
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.
        quantile_transform.html#sklearn.preprocessing.quantile_transform

        @param inplace: Bool.
        """
        arr = quantile_transform(self.df, output_distribution="normal")
        df = pd.DataFrame(arr, index=self.df.index, columns=self.df.columns)

        if inplace:
            self.df = df
        else:
            return CoordDf(df=df)

    def paired_distances(self, other_df: pd.DataFrame) -> np.ndarray:
        """
        Compute euclidean distances between points in self.df and paired
        points in another dataframe. The other dataframe must have the same
        dimensions as self.df

        @param other_df: pd.DataFrame

        @returns ndarray: Euclidean distances.
        """
        if self.df.index.shape != other_df.index.shape:
            raise ValueError("Index lengths mismatched.")

        if self.df.columns.shape != other_df.columns.shape:
            raise ValueError("Column lengths mismatched.")

        n = self.df.shape[0]

        distances = np.empty(n)

        for i in range(n):
            try:
                distances[i] = euclidean(
                    u=self.df.iloc[i, :], v=other_df.iloc[i, :]
                )

            except ValueError:
                # euclidean raises ValueError for NaN coordinates
                # Set distance to NaN for these cases
                distances[i] = np.nan

        return distances


class SeqDf:
    """Class for DataFrames containing amino acid sequences."""

    def __init__(self, df: pd.DataFrame) -> None:
        """
        @param df: pd.DataFrame. Columns are amino acid positions.
        """
        self.df = df

    def __repr__(self) -> str:
        return "SeqDf with {} samples and {} sequence positions.".format(
            *self.df.shape
        )

    @classmethod
    def from_fasta(cls, path: str) -> "SeqDf":
        """
        Make a SeqDf from a fasta file.

        @param path: Str. Path to fasta file.
        """
        return cls(df_from_fasta(path=path, positions="infer"))

    @classmethod
    def from_series(cls, series: pd.Series) -> "SeqDf":
        """
        Make new SeqDf from a series.

        @param series: pd.Series. Each element in series is a string. See
            mapdeduce.helper.expand_sequences for more details.
        """
        return cls(expand_sequences(series))

    def remove_invariant(self, inplace: bool = True) -> Optional["SeqDf"]:
        """
        Remove positions (columns) that contain only one amino acid.

        @param inplace: Bool.
        """
        mask = self.df.apply(lambda x: pd.unique(x).shape[0] > 1)
        n = (~mask).sum()
        print("Removed {} invariant sequence positions".format(n))
        new = self.df.loc[:, self.df.columns[mask]]

        if inplace:
            self.df = new
        else:
            return new

    def get_dummies(self, inplace: bool = True) -> Optional["SeqDf"]:
        """
        Get dummy variable representation of the sequences.

        @param inplace: Bool.
        """
        d = pd.get_dummies(self.df, prefix_sep="").astype(float)

        if inplace:
            self.dummies = d
        else:
            return d

    def shuffle_dummies(self, n_shuffles: int, c: str) -> np.ndarray:
        """
        Return a DataFrame containing n shuffles of the data in column c

        @param n_shuffles: Int. Number of shuffles
        @param c: Must lookup column in self.dummies

        @returns (N, n_shuffles) ndarray
        """
        values = self.dummies.loc[:, c].values
        arr = np.empty((values.shape[0], n_shuffles))
        for i in range(n_shuffles):
            arr[:, i] = sklearn.utils.shuffle(values)
        return arr

    def get_dummies_at_positions(self, positions: list[int]) -> set[str]:
        """
        Return dummies at given HA positions.

        Dummy variable names are either singles (e.g. 135K), compound by
        identity (e.g. 7D|135K|265E), or compound by collinearity
        (e.g. 145K|155S~189K).  Return the full column name if any
        constituent SNP is at a position in *positions*.

        @param positions: List of positions.
        """
        return columns_at_positions(self.dummies.columns, positions)

    def _merge_identical_dummies(self) -> pd.DataFrame:
        """
        Merge SNPs that are identical in all strains.

        @returns pd.DataFrame with merged columns.
        """
        grouped = self.dummies.T.groupby(by=self.dummies.index.tolist())

        return pd.DataFrame(
            data={"|".join(sorted(g.index)): n for n, g in grouped},
            index=self.dummies.index,
        )

    def _merge_dummies_with_complements(self) -> pd.DataFrame:
        """
        Merge SNPs that are identical or complements in all strains.

        Complement SNPs have a "-" prepended to their name. The first AAP
        in the merged name always represents the pattern of 0/1s.

        @returns pd.DataFrame with merged columns.
        """
        # Group by canonical form (handles both duplicates and complements)
        # canonical_tuple -> {'normal': [...], 'complement': [...]}
        canonical_groups = {}

        for col in self.dummies.columns:
            vals = tuple(self.dummies[col].values)
            complement = tuple(1.0 - v for v in vals)

            # Use the smaller tuple as canonical form
            if vals <= complement:
                canonical = vals
                is_complement = False
            else:
                canonical = complement
                is_complement = True

            if canonical not in canonical_groups:
                canonical_groups[canonical] = {
                    "normal": [],
                    "complement": [],
                }

            if is_complement:
                canonical_groups[canonical]["complement"].append(col)
            else:
                canonical_groups[canonical]["normal"].append(col)

        # Build the merged DataFrame
        data = {}
        for canonical, members in canonical_groups.items():
            normal = sorted(members["normal"])
            complements = sorted(members["complement"])

            # Ensure at least one normal member so first AAP defines pattern
            if not normal and complements:
                # Move first complement to normal, flip stored values
                normal = [complements[0]]
                complements = complements[1:]
                stored_values = tuple(1.0 - v for v in canonical)
            else:
                stored_values = canonical

            # Build merged name: normal names first, then -complement names
            names = normal + [f"-{name}" for name in complements]
            merged_name = "|".join(names)

            data[merged_name] = list(stored_values)

        return pd.DataFrame(data, index=self.dummies.index)

    def merge_duplicate_dummies(
        self, inplace: bool = False, merge_complements: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Merge SNPs that are identical (or complements) in all strains.
        Identical SNPs are concatenated with '|' characters.

        @param inplace: Bool.

        @param merge_complements: Bool. If True (default), also merge SNPs
            that are complements of each other (i.e., one profile is 1 minus
            the other). Complement SNPs have a "-" prepended to their name
            in the merged column name. The first AAP in the merged name
            always represents the pattern of 0/1s in the profile.

        @raises ValueError: If any SNP names contain "|" or "-" characters.
        """
        # Validate SNP names don't contain reserved characters that are used
        # to concatenate SNP names
        reserved_chars = {"|", "-"}
        invalid_names = [
            col
            for col in self.dummies.columns
            if any(c in str(col) for c in reserved_chars)
        ]

        if invalid_names:
            raise ValueError(
                f"SNP names cannot contain '|' or '-' characters: "
                f"{invalid_names}"
            )

        if merge_complements:
            df = self._merge_dummies_with_complements()
        else:
            df = self._merge_identical_dummies()

        if inplace:
            self.dummies = df
        else:
            return df

    def consensus(self) -> pd.DataFrame:
        """
        Compute the consensus sequence.

        @returns pd.Series.
        """
        return self.df.apply(site_consensus, axis=0)

    def merge_duplicate_strains(
        self, inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Replace all strains that have the same index with a single
        consensus strain.

        @param inplace: Bool.

        @param returns: mapdeduce.dataframes.SeqDf, if inplace=False.
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

    def prune_collinear_dummies(
        self, threshold: float = 0.95, inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Collapse near-collinear dummy variable clusters into a single
        representative, following the pattern of merge_duplicate_dummies.

        Clusters are joined with ``~`` in the column name.

        @param threshold: float. Maximum r squared between retained SNPs.
        @param inplace: Bool.  If True, update ``self.dummies`` and store
            ``self.collinear_mapping`` and
            ``self.collinear_removed_to_kept``.
        """
        pruned_df, removed_to_kept = prune_collinear_snps(
            self.dummies, threshold=threshold
        )

        # Build reverse mapping: kept -> [removed1, removed2, ...]
        clusters: dict[str, list[str]] = {}
        for removed, kept in removed_to_kept.items():
            clusters.setdefault(kept, []).append(removed)

        # Rename retained columns that have pruned members
        rename_map = {}
        for kept, pruned_list in clusters.items():
            new_name = "~".join(sorted([kept] + pruned_list))
            rename_map[kept] = new_name

        result = pruned_df.rename(columns=rename_map)

        if inplace:
            self.dummies = result
            self.collinear_mapping = {
                k: sorted(v) for k, v in clusters.items()
            }
            self.collinear_removed_to_kept = removed_to_kept
        else:
            return result

    def to_fasta(self, path: str) -> None:
        """
        Write the sequences in fasta format.

        @param path: Path to write file.
        """
        if not path.lower().endswith(".fasta"):
            path += ".fasta"
        with open(path, "w") as handle:
            for row in self.df.iterrows():
                handle.write(">{}\n".format(row.name))
                handle.write("{}\n".format("".join(row)))


def prune_collinear_snps(
    snps: pd.DataFrame, threshold: float = 0.95
) -> pd.DataFrame:
    """
    Prune highly collinear SNPs based on pairwise correlation.

    Iterates through SNPs and removes those with r2 > threshold relative to any
    already-retained SNP.

    @param snps: pd.DataFrame of SNPs (columns) for individuals (rows).
        Column names must be unique.
    @param threshold: maximum allowed r2 between retained SNPs
        (default 0.95)

    @returns: tuple containing:
        pruned_df: DataFrame containing only the retained SNPs
        removed_to_kept: dict mapping each removed SNP name to the name of
            the retained SNP it correlates most highly with
    """
    if not isinstance(snps, pd.DataFrame):
        raise TypeError("snps must be a pandas DataFrame")

    if snps.columns.duplicated().any():
        raise ValueError("SNP column names must be unique")

    if not 0 <= threshold <= 1:
        raise ValueError("r2 threshold must be between 0 and 1 inclusive")

    snp_names = list(snps.columns)
    S = len(snp_names)

    # Standardize SNPs (mean=0, std=1) for correlation calculation
    snps_std = snps.values - snps.values.mean(axis=0)
    norms = np.sqrt((snps_std**2).sum(axis=0))
    # Avoid division issues for uniform/near-uniform SNPs
    norms[norms < 1e-12] = 1
    snps_std = snps_std / norms

    # First pass: determine which SNPs to keep
    keep_indices = [0]  # always keep the first

    for i in range(1, S):
        # Compute r2 with all retained SNPs
        # r = dot product of standardized vectors (already normalized by norms)
        # Use np.dot instead of @ to avoid spurious warnings in NumPy < 2.4
        r2 = np.dot(snps_std[:, i], snps_std[:, keep_indices]) ** 2

        # Keep this SNP if not highly correlated with any retained SNP
        if r2.max() < threshold:
            keep_indices.append(i)

    # Second pass: map removed SNPs to their most correlated kept SNP
    # Done after pruning so we compare against the final set of kept SNPs
    removed_indices = [i for i in range(1, S) if i not in keep_indices]
    removed_to_kept = {}

    kept_snps = snps_std[:, keep_indices]
    for i in removed_indices:
        r2 = np.dot(snps_std[:, i], kept_snps) ** 2
        best_match_idx = keep_indices[np.argmax(r2)]
        removed_to_kept[snp_names[i]] = snp_names[best_match_idx]

    kept_cols = [snp_names[i] for i in keep_indices]

    return snps[kept_cols], removed_to_kept
