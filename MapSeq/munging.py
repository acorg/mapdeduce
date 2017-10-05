"""Utilities for munging data."""

from Bio import SeqIO
import pandas as pd
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
    sequence_dict = dict_from_fasta(path)
    df = pd.DataFrame.from_dict({
        k: tuple(v) for k, v in sequence_dict.iteritems()
    }, orient='index')

    # Drop unwanted columns
    df = df.iloc[:, :len(positions)]

    df.columns = positions

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
