"""Tests for mapdeduce.plotting."""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Shadow

from mapdeduce.plotting import plot_arrow

matplotlib.use("Agg")


class TestPlotArrowShadow:
    """Tests for the drop shadow option on plot_arrow."""

    def setup_method(self):
        self.fig, self.ax = plt.subplots()

    def teardown_method(self):
        plt.close(self.fig)

    def test_no_shadow_by_default(self):
        """Shadow should be disabled by default."""
        plot_arrow((0, 0), (1, 1), "red", ax=self.ax)
        shadow_patches = [p for p in self.ax.patches if isinstance(p, Shadow)]
        assert len(shadow_patches) == 0

    def test_shadow_enabled(self):
        """Setting shadow=True should add a Shadow patch to the axes."""
        plot_arrow((0, 0), (1, 1), "red", shadow=True, ax=self.ax)
        shadow_patches = [p for p in self.ax.patches if isinstance(p, Shadow)]
        assert len(shadow_patches) == 1

    def test_shadow_kwds_passed(self):
        """shadow_kwds should be forwarded to Shadow constructor."""
        plot_arrow(
            (0, 0),
            (1, 1),
            "red",
            shadow=True,
            shadow_kwds={"ox": 0.05, "oy": -0.05},
            ax=self.ax,
        )
        shadow_patches = [p for p in self.ax.patches if isinstance(p, Shadow)]
        assert len(shadow_patches) == 1

    def test_shadow_false_explicitly(self):
        """Explicitly passing shadow=False should not add shadow."""
        plot_arrow((0, 0), (1, 1), "red", shadow=False, ax=self.ax)
        shadow_patches = [p for p in self.ax.patches if isinstance(p, Shadow)]
        assert len(shadow_patches) == 0
