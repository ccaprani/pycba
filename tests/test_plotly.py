# -*- coding: utf-8 -*-
"""
Tests for the optional Plotly plotting backend.  Skipped entirely if plotly is
not installed.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pycba as cba

go = pytest.importorskip("plotly.graph_objects")


@pytest.fixture(autouse=True)
def _reset_backend():
    """Keep tests independent of the global backend setting."""
    cba.set_backend("matplotlib")
    yield
    cba.set_backend("matplotlib")
    plt.close("all")


def _beam():
    ba = cba.BeamAnalysis([6.0, 8.0], 30e3, [-1, 0, -1, 0, -1, 0])
    ba.add_udl(1, 10)
    ba.add_udl(2, 10)
    ba.analyze()
    return ba


def _env():
    ba = _beam()
    veh = cba.BridgeAnalysis(ba)
    veh.add_vehicle(np.array([1.5]), np.array([50.0, 50.0]))
    return veh.run_vehicle(0.5)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------
def test_default_backend_is_matplotlib():
    assert cba.get_backend() == "matplotlib"


def test_set_backend_roundtrip():
    cba.set_backend("plotly")
    assert cba.get_backend() == "plotly"
    cba.set_backend("matplotlib")
    assert cba.get_backend() == "matplotlib"


def test_set_backend_invalid():
    with pytest.raises(ValueError, match="Unknown backend"):
        cba.set_backend("ggplot")


# ---------------------------------------------------------------------------
# Result diagrams
# ---------------------------------------------------------------------------
def test_plot_results_plotly():
    fig = _beam().plot_results(backend="plotly")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # M, V, D
    # Moment (row 1) is sagging-positive -> reversed y-axis; others are not.
    assert fig.layout.yaxis.autorange == "reversed"
    assert fig.layout.yaxis2.autorange is None
    assert fig.layout.yaxis3.autorange is None
    assert fig.layout.hovermode == "x unified"


@pytest.mark.parametrize(
    "method,reversed_",
    [("plot_bmd", True), ("plot_sfd", False), ("plot_dsd", False)],
)
def test_single_diagrams_plotly(method, reversed_):
    fig = getattr(_beam(), method)(backend="plotly")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    want = "reversed" if reversed_ else None
    assert fig.layout.yaxis.autorange == want


def test_plotly_hover_carries_units():
    fig = _beam().plot_bmd(backend="plotly", units="N-mm")
    assert "N·mm" in fig.data[0].hovertemplate
    assert fig.layout.yaxis.title.text == "Bending Moment (N·mm)"


def test_envelope_plot_plotly():
    fig = _env().plot(backend="plotly")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # Mmax, Mmin, Vmax, Vmin
    assert fig.layout.yaxis.autorange == "reversed"  # moment panel


# ---------------------------------------------------------------------------
# Global backend switch and matplotlib parity
# ---------------------------------------------------------------------------
def test_global_backend_switch():
    ba = _beam()
    cba.set_backend("plotly")
    assert isinstance(ba.plot_sfd(), go.Figure)


def test_matplotlib_backend_unchanged():
    ba = _beam()
    ax = ba.plot_bmd()  # default backend
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.yaxis_inverted()
    fig, axs = ba.plot_results(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
