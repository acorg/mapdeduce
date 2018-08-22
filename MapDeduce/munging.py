"""Utilities for munging data."""

from Bio import SeqIO
import pandas as pd
import numpy as np
import re


def dict_from_fasta(path):
    """
    Read fasta file given in path.

    @param path: String.

    @returns dict: Keys are record IDs in upper case, values are Biopython
        sequence objects.
    """
    with open(path, 'r') as handle:
        return {r.id.upper(): r.seq for r in SeqIO.parse(handle, 'fasta')}


def df_from_fasta(path, positions=tuple(range(1, 329))):
    """
    Read fasta file specified in path.

    @param path: String.
    @param positions: List-like containing integers. Specifies the numbering
        used in the fasta file. The first item in this list specifies the
        first position in the fasta file. Positions in the fasta file beyond
        the last element in this list are dropped.

    @returns pd.DataFrame: Indexes are record IDs in upper case, columns are
        positions
    """
    with open(path, 'r') as handle:
        seqs = [(r.id.upper(), r.seq)for r in SeqIO.parse(handle, 'fasta')]

    index = [s[0] for s in seqs]

    data = [pd.Series(list(s[1])) for s in seqs]

    df = pd.DataFrame(data, index=index)

    df = df.iloc[:, :len(positions)]  # Drop unwanted columns

    df.columns = positions  # Rename columns

    return df


def read_eu_coordinate_layout(path):
    """
    Read layout files from Eugene.

    @param path: String

    @returns pd.DataFrame: Indexes are strain names, columns are x, y
        coordinates. DataFrame contains only the antigens.
    """
    df = pd.read_csv(filepath_or_buffer=path,
                     sep=' ',
                     index_col=(0, 1),
                     header=None,
                     names=('type', 'strain', 'x', 'y'))
    return df.loc['AG', :]


# Compile these once
strain_regex = re.compile("^A\/[-_A-Z]*\/[-A-Z0-9]*\/[0-9]{4}_")
ah3n2_regex = re.compile("^([A-Z]+_)?A\(H3N2\)\/")
human_regex = re.compile("\/HUMAN\/")


def clean_strain_name(strain_name):
    """
    Replace A(H3N2) with A at the start of a strain name.

    Then, match the first four components of a strain name, delimited by /,
    to remove the passage details, and additional fields that are often
    attached to the end of a string.

    Fields:
        1: A always an A
        2: TASMANIA Always only letters. Can contain _ or -
        3: 57       Any number of numbers.
                    Can contain -. e.g. 16-1252
                    Can contain alphabet, examples: A, B, AUCKLAND
        4: 2015_ four numbers always followed by an underscore.

    @param strain_name. Str.
    """
    strain_name = re.sub(pattern=human_regex, repl="/", string=strain_name)
    strain_name = re.sub(pattern=ah3n2_regex, repl="A/", string=strain_name)
    match = re.match(pattern=strain_regex, string=strain_name)
    try:
        return match.group().strip('_')
    except AttributeError:
        return strain_name


def clean_df_strain_names(df, filename):
    """
    Clean strain names of DataFrame indexes and write a file containing
    rows of the original and altered strain names for inspecting.

    @param df: pd.DataFrame. Indexes are strain names.
    @param filename: Str. Path to write filename containing original and new
        strain names. This only contains strain names that have changed.
    @returns df: pd.Dataframe. With cleaned strain names.
    """
    orig_names = df.index
    new_names = orig_names.map(clean_strain_name)

    # Write file for inspecting name changes
    len_longest_strain_name = max(map(len, new_names))
    col_width = len_longest_strain_name if len_longest_strain_name > 3 else 3
    format_string = '{{:{}}} {{}}\n'.format(col_width)
    with open(filename, 'w') as fobj:
        fobj.write(format_string.format('New', 'Original'))
        for n, o in zip(new_names, orig_names):
            fobj.write(format_string.format(n, o))

    df.index = new_names
    return df


def handle_duplicate_sequences(df):
    """
    (A) Remove rows with identical indexes and sequences.
    (B) Keep rows with duplicate sequences, but different indexes.
    (C) Merge strains with identical indexes, but different sequences.
        (replace ambiguous positions with X).

    @param df. pd.DataFrame. Rows are strains, columns are amino acid
        positions.
    """
    # (A, B) remove strains with repeated names & sequences
    df = df[~(df.duplicated() & df.index.duplicated())]

    # Each set of remaining duplicate indexes have different sequences
    # Merge these groups of sequences
    remaining_dupe_idx = df.index.duplicated(keep=False)
    if remaining_dupe_idx.any():
        merged = {i: df.loc[i, :].apply(merge_amino_acids)
                  for i in df.index[remaining_dupe_idx]}
        merged = pd.DataFrame.from_dict(merged, orient="index")
        merged.columns = df.columns

        # Uniqe indexes
        unique = df[~remaining_dupe_idx]

        return pd.concat((merged, unique))

    else:
        return df


def merge_amino_acids(amino_acids):
    """
    Merge amino acids. If there is only one unique amino acid
    return that. If there is only one unique amino acid, and the
    rest are unknown (np.nan), then return the known amino acid.
    If there are multiple known amino acids, then return unkown
    (np.nan)

    @param amino_acids: pd.Series
    """
    unique = pd.unique(amino_acids)
    if unique.shape[0] == 1:
        return unique[0]

    unique_no_na = amino_acids.dropna().unique()
    if unique_no_na.shape[0] == 1:
        return unique_no_na[0]
    else:
        return np.nan
