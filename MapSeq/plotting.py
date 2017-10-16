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
    elif map == 2018:
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_xticks(range(-7, 8))
        ax.set_yticks(range(-7, 8))
    elif map == "cdc-melb-2017-merge":
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_xticks(range(-7, 8))
        ax.set_yticks(range(-6, 7))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect(1)


# Define ellipses to demark clusters on maps
kwds = dict(fc="None", color="black", lw=1)
ellipse_params = {
    2009: {
        "SY97": dict(xy=(-6, 3.5), width=3, height=5, angle=-45, **kwds),
        "FU02": dict(xy=(-1.9, 1), width=3.5, height=6, angle=-40, **kwds),
        "CA04": dict(xy=(0.6, -1), width=2.8, height=6, angle=-40, **kwds),
        "WI05": dict(xy=(3.5, -2.5), width=3.5, height=6.5, angle=-40, **kwds),
        "PE09": dict(xy=(6, -7), width=3, height=6, angle=-40, **kwds),
    },
    2017: {
        "PE09": dict(xy=(-4.5, -0.75), width=2, height=3, angle=20, **kwds),
        "SW13": dict(xy=(-2.5, 0.75), width=2.5, height=4, angle=0, **kwds),
        "HK14": dict(xy=(0.5, -0.5), width=3.5, height=5, angle=-25, **kwds),
    },
    # Ugly hack
    # Need to specify different ellipses for CDC vs MELB
    # 2017 key in this dict refers to MELB 2017 data
    # 2018 key is for CDC 2017 data
    2018: {
        "PE09": dict(xy=(-2.7, -2), width=2.5, height=3.5, angle=-10, **kwds),
        "SW13": dict(xy=(0.5, 1.75), width=2, height=4, angle=-70, **kwds),
        "HK14": dict(xy=(1, -0.75), width=2.5, height=5, angle=-70, **kwds),
    },
    "cdc-melb-2017-merge": {
        "PE09": dict(xy=(-2.7, 2), width=2.25, height=4, angle=-15, **kwds),
        "SW13": dict(xy=(0, -2), width=2, height=4.5, angle=85, **kwds),
        "HK14": dict(xy=(1, 1), width=3.75, height=5.25, angle=80, **kwds),
    },
}


def add_ellipses(map):
    """
    Add ellipses delimiting the clusters to the current ax.

    @param map. Str. Either Pre2009 or 2017
    """
    ax = plt.gca()
    for cluster, params in ellipse_params[map].iteritems():
        ax.add_artist(Ellipse(zorder=15, **params))

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

        elif map == 2018:
            if cluster == "PE09":
                x = params["xy"][0] - params["width"] - 0.5
            else:
                x = params["xy"][0] + params["width"] + 1
            y = params["xy"][1] + 1
            ha = "center"
            va = "bottom"

        elif map == "cdc-melb-2017-merge":
            if cluster == "PE09":
                x = params["xy"][0] - params["width"] - 0.5
            elif cluster == "HK14":
                x = params["xy"][0] + params["width"]
            else:
                x = params["xy"][0] + params["width"] + 2
            y = params["xy"][1]
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
    "X": "#777777",  # unknown AA
    "Y": "#a5b8c7"
}


def combination_label(combination):
    """
    Return a label for a combination of substitutions.

    @param combination: Dict. E.g.:
                    {
                    145: "K",
                    193: "G",
                    189: "S",
                    }

    @returns string: E.g.:
                    "145K+189S+193G"
    """
    comb_list = sorted("{}{}".format(k, v) for k, v in combination.iteritems())
    return "+".join(comb_list)


def point_size(n):
    """
    Determine point size as a function of the number of points in the scatter
    plot

    @param n. Int. Number of points.
    """
    if n > 9000:
        return point_size(9000)
    return (-4.26764259e-03 * n) + (2.30663771e-07 * n**2) + 20.21953616
