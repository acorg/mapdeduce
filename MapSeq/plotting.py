"""Utilities and defaults for plotting"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def setup_ax(map):
    """
    Sets up layout of some ax settings.

    @param map: String. 2017 or Pre2009
    """
    ax = plt.gca()
    if map == 2017:
        ax.set_xticks(range(-6, 7))
        ax.set_yticks(range(-6, 6))
    elif map == 2009:
        ax.set_xlim(-10, 10)
        ax.set_xticks(range(-10, 10))
        ax.set_yticks(range(-13, 9))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect(1)


# Define ellipses to demark clusters on maps
kwds = dict(fc="None", color="black", lw=2)
ellipse_params = {
    2009: {
        "SY97": dict(xy=(-6.5, 3.5), width=3, height=5, angle=-45, **kwds),
        "FU02": dict(xy=(-2.1, 0.5), width=4, height=7, angle=-40, **kwds),
        "CA04": dict(xy=(1.5, -1), width=3.5, height=6.5, angle=-40, **kwds),
        "WI05": dict(xy=(4, -3.5), width=3.5, height=6.5, angle=-40, **kwds),
        "PE09": dict(xy=(6, -7), width=3, height=6, angle=-40, **kwds),
    },
    2017: {
        "PE09": dict(xy=(-4.5, -0.75), width=2, height=3, angle=20, **kwds),
        "SW13": dict(xy=(-2.5, 0.75), width=2.5, height=4, angle=0, **kwds),
        "HK14": dict(xy=(0.5, -0.5), width=3.5, height=5, angle=-25, **kwds),
    }
}


def add_ellipses(map):
    """
    Add ellipses delimiting the clusters to the current ax.

    @param map. Str. Either Pre2009 or 2017
    """
    ax = plt.gca()
    for cluster, params in ellipse_params[map].iteritems():
        ax.add_artist(Ellipse(**params))

        # Label the Ellipse
        if map == 2009:
            x = params["xy"][0] - params["width"] / 2.3
            y = params["xy"][1] - params["height"] / 2.3
            ha = "right"
            va = "top"

        elif map == 2017:
            x = params["xy"][0]
            y = params["xy"][1] + 0.5 + params["height"] / 2.3
            ha = "center"
            va = "bottom"

        ax.text(x, y, cluster, fontsize=12, ha=ha, va=va, zorder=20)


amino_acid_colors = {
    "A": "#F76A05",
    "C": "#dde8cf",
    "D": "#a020f0",
    "E": "#9e806e",
    "F": "#f1b066",
    "G": "#675b2c",
    "H": "#ffc808",
    "I": "#8b8989",
    "K": "#03569b",
    "L": "#9B84AD",
    "M": "#93EDC3",
    "N": "#a2b324",
    "P": "#e9a390",
    "Q": "#742f32",
    "R": "#75ada9",
    "S": "#e72f27",
    "T": "#049457",
    "V": "#00939f",
    "W": "#ED93BD",
    "-": "#777777",  # unknown AA
    "X": "#777777",  # unknown AA
    "Y": "#a5b8c7"
}
