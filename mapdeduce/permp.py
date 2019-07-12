"""Python wrapper for R permp function, in the statmod library."""

import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
rpackages.importr("statmod")


def permp(x, nperm, n1, n2, total_nperm=None, method="auto", twosided=True):
    """
    Calculates exact p-values for permutation tests when permutations
    are randomly drawn with replacement.

    @param x: Int. number of permutations that yielded test statistics at
        least as extreme as the observed data.
    @param nperm: Int. total number of permutations performed.
    @param n1: Int. sample size of group 1. Not required if total.nperm is
        supplied.
    @param n2: Int. sample size of group 2. Not required if total.nperm is
        supplied.
    @param total_nperm: Int. total number of permutations allowable from the
        design  of the experiment.
    @param method: Str. Indicates computation method. Possible values are
        "exact", "approximate" or "auto".
    @param twosided: Bool. Is the test two-sided and symmetric between the
        two groups?

    This function can be used for calculating exact p-values for
    permutation tests where permutations are sampled with replacement,
    using theory and methods developed by Phipson and Smyth (2010).
    The input values are the total number of permutations done
    (nperm) and the number of these that were considered at least as
    extreme as the observed data (x).

    total.nperm is the total number of distinct values of the test
    statistic that are possible. This is generally equal to the number
    of possible permutations, unless a two-sided test is conducted
    with equal sample sizes, in which case total.nperm is half the
    number of permutations, because the test statistic must then be
    symmetric in the two groups. Usually total.nperm is computed
    automatically from n1 and n2, but can also be supplied
    directly by the user.

    When method="exact", the p-values are computed to full machine
    precision by summing a series terms. When method="approximate",
    an approximation is used that is faster and uses less memory. If
    method="auto", the exact calculation is used when total.nperm
    is less than or equal to 10,000 and the approximation is used
    otherwise.

    Reference:
        Phipson B, and Smyth GK (2010). Permutation p-values should never
        be zero: calculating exact p-values when permutations are randomly
        drawn. _Statistical Applications in Genetics and Molecular
        Biology_, Volume 9, Issue 1, Article 39. <URL:
        http://www.statsci.org/smyth/pubs/PermPValuesPreprint.pdf>
    """
    if total_nperm is None:
        total_nperm = robjects.NULL
    else:
        total_nperm = robjects.Vector(total_nperm)

    
    return np.array(robjects.r["permp"](x=robjects.Vector(x),
                                        nperm=robjects.Vector(nperm),
                                        n1=robjects.Vector(n1),
                                        n2=robjects.Vector(n2),
                                        total_nperm=total_nperm,
                                        method=method,
                                        twosided=twosided))
