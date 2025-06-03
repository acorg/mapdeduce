from .dataframes import CoordDf, SeqDf
from .lmm import (
    MvLMM,
    compute_likelihood_ratio_test_statistic,
    compute_p_value,
    cov,
    effective_tests,
)
from .mapseq import MapSeq, OrderedMapSeq
from .plotting import make_ax_a_map

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mapdeduce")
except PackageNotFoundError:
    pass

__all__ = [
    "__version__",
    "MvLMM",
    "compute_likelihood_ratio_test_statistic",
    "compute_p_value",
    "CoordDf",
    "cov",
    "effective_tests",
    "make_ax_a_map",
    "MapSeq",
    "OrderedMapSeq",
    "SeqDf",
]
