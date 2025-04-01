import warnings
from datetime import datetime
from pathlib import Path

import mapdeduce as md
import pandas as pd
from maps import Chart


def main():
    import argparse

    parser = argparse.ArgumentParser("md-association-test")
    parser.add_argument("--ace", help="Path to .ace file", required=True)
    parser.add_argument("--csv", help="Filename of CSV output.", required=True)
    parser.add_argument(
        "--min-aap-count",
        help="Only test AAPs that appear in at least this many strains.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--sites",
        help=(
            "Sequence sites to include in the analysis. Must pass two values, from and to that "
            "define the first and last sites to include in the analysis. 1-indexed."
        ),
        nargs=2,
        default=(109, 301),
        type=int,
    )
    args = parser.parse_args()

    sites_from, sites_to = args.sites
    if sites_to <= sites_from:
        raise ValueError("Second value of --sites must be greater than first")

    if args.min_aap_count < 0:
        raise ValueError("--min_aap_count must be positive")

    if not Path(args.csv).parent.is_dir():
        raise ValueError(f"directory doesn't exist: {Path(args.csv).parent}")

    chart = Chart.from_ace(args.ace)

    if args.min_aap_count > (len(chart.best_projection.layout) / 2):
        raise ValueError(
            "--min_aap_count must be less than half the number of strains in the map"
        )

    # DataFrame containing sequences and coords
    df_seq_coords = (
        chart.best_projection.layout.join(
            pd.Series(
                {ag.designation: ag.sequence for ag in chart.antigens}, name="sequence"
            ),
            on="Designation",
        )
        .dropna(axis=0)  # Drop strains that don't have coordinates and sequences
        .loc["AG"]  # Only keep antigens
    )

    # Split the sequences so that
    seq_df = (
        df_seq_coords["sequence"]
        .str.split("", expand=True)
        .loc[:, list(range(sites_from, sites_to + 1))]
    )
    coord_df = df_seq_coords[chart.best_projection.layout.columns]

    oms = md.OrderedMapSeq(seq_df=seq_df, coord_df=coord_df)
    oms.filter(remove_invariant=True, get_dummies=True, merge_duplicate_dummies=True)

    aap_counts = oms.seqs.dummies.sum().sort_values()
    common_aaps = aap_counts.index[
        (args.min_aap_count <= aap_counts)
        & (aap_counts <= (len(oms.seqs.dummies) - args.min_aap_count))
    ]

    at = md.AssociationTest(dummies=oms.seqs.dummies, phenotypes=oms.coord.df)

    df_test = at.test_aaps(common_aaps)

    try:
        df_test.to_csv(args.csv)
    except FileNotFoundError:
        now = datetime.now()
        filename = f"{now:%Y%m%d_%H%M%S}_assoc_test.csv"
        df_test.to_csv(filename)
        warnings.warn(f"Couldn't save to {args.csv}. Saved to {filename} instead.")
