"""Useful data."""

# Source:
#   https://www.fludb.org/brc/vaccineRecommend.spg?decorator=influenza

seasonToVaccine = {
    "2000-2001": "A(H3N2)/MOSCOW/10/1999",
    "2001-2002": "A(H3N2)/MOSCOW/10/1999",
    "2002-2003": "A(H3N2)/MOSCOW/10/1999",
    "2003-2004": "A(H3N2)/MOSCOW/10/1999",
    "2004-2005": "A(H3N2)/FUJIAN/411/2002",
    "2005-2006": "A(H3N2)/CALIFORNIA/7/2004",
    "2006-2007": "A(H3N2)/WISCONSIN/67/2005",
    "2007-2008": "A(H3N2)/WISCONSIN/67/2005",
    "2008-2009": "A(H3N2)/BRISBANE/10/2007",
    "2009-2010": "A(H3N2)/BRISBANE/10/2007",
    "2010-2011": "A(H3N2)/PERTH/16/2009",
    "2011-2012": "A(H3N2)/PERTH/16/2009",
    "2012-2013": "A(H3N2)/VICTORIA/361/2011",
    "2013-2014": "A(H3N2)/VICTORIA/361/2011",
    "2014-2015": "A(H3N2)/TEXAS/50/2012",
    "2015-2016": "A(H3N2)/SWITZERLAND/9715293/2013",
    "2016-2017": "A(H3N2)/HONG KONG/4801/2014",
    "2017-2018": "A(H3N2)/HONG KONG/4801/2014",
    "2000": "A(H3N2)/MOSCOW/10/1999",
    "2001": "A(H3N2)/MOSCOW/10/1999",
    "2002": "A(H3N2)/MOSCOW/10/1999",
    "2003": "A(H3N2)/MOSCOW/10/1999",
    "2004": "A(H3N2)/FUJIAN/411/2002",
    "2005": "A(H3N2)/WELLINGTON/1/2004",
    "2006": "A(H3N2)/CALIFORNIA/7/2004",
    "2007": "A(H3N2)/WISCONSIN/67/2005",
    "2008": "A(H3N2)/BRISBANE/10/2007",
    "2009": "A(H3N2)/BRISBANE/10/2007",
    "2010": "A(H3N2)/PERTH/16/2009",
    "2011": "A(H3N2)/PERTH/16/2009",
    "2012": "A(H3N2)/PERTH/16/2009",
    "2013": "A(H3N2)/VICTORIA/361/2011",
    "2014": "A(H3N2)/TEXAS/50/2012",
    "2015": "A(H3N2)/SWITZERLAND/9715293/2013",
    "2016": "A(H3N2)/HONG KONG/4801/2014",
    "2017": "A(H3N2)/HONG KONG/4801/2014",
}

countryToHemisphere = {
    "AUSTRALIA": "S",
    "CANADA": "N",
    "CHINA": "N",
    "GERMANY": "N",
    "NEW ZEALAND": "S",
    "SOUTH AFRICA": "S",
    "SPAIN": "N",
    "TAIWAN": "N",
    "UNITED KINGDOM": "N",
    "UNITED STATES OF AMERICA": "N",
}


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

not_109_to_301 = list(range(1, 109)) + list(range(302, 329))
not_110_to_199 = list(range(1, 110)) + list(range(200, 329))


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
    },
}
