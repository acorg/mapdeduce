"""Useful data."""

amino_acids = {
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
}

not_109_to_301 = range(1, 109) + range(302, 329)
not_110_to_199 = range(1, 110) + range(200, 329)


# Cluster difference amino acid polymorphisms for post SY97
# Doesn't include positions outside of 109-301

clus_diff_aaps = {
    "SY97-FU02": {
        "131A",
        "131T",
        "144D",
        "144N",
        "155H",
        "155T",
        "156Q",
        "156H",
        "202V",
        "202I",
        "222W",
        "222R",
        "225G",
        "225D",
    },
    "FU02-CA04": {
        "145K",
        "145N",
        "189S",
        "189N",
        "226V",
        "226I",
        "159F",
        "159Y",
        "227S",
        "227P",
    },
    "CA04-WI05": {
        "225D",
        "225N",
        "193F",
        "193S",
    },
    "WI05-PE09": {
        "189N",
        "189K",
        "144N",
        "144K",
        "158K",
        "158N",
    }
}
