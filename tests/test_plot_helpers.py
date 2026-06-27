# -*- coding: utf-8 -*-
"""
Tests for the native plotting helpers: ``plot_vehicle`` (top-level,
``Vehicle.plot`` and ``VehicleLibrary.plot_vehicle``) and
``Envelopes.plot_coincidents``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pycba as cba


# --------------------------------------------------------------------------- #
# plot_vehicle
# --------------------------------------------------------------------------- #
def test_plot_vehicle_access_paths():
    veh = cba.VehicleLibrary.AU.get_m1600(6.25)
    ax1 = cba.plot_vehicle(veh, show=False)
    ax2 = veh.plot(show=False)
    ax3 = cba.VehicleLibrary.plot_vehicle(veh, show=False)
    for ax in (ax1, ax2, ax3):
        assert isinstance(ax, matplotlib.axes.Axes)
        assert ax.texts  # axle-load labels were drawn
    plt.close("all")


def test_plot_vehicle_into_existing_axes():
    fig, ax = plt.subplots()
    out = cba.plot_vehicle(cba.VehicleLibrary.US.get_hl93_truck(), ax=ax, title="HL-93")
    assert out is ax
    assert "HL-93" in ax.get_title()
    plt.close(fig)


def test_plot_vehicle_rejects_non_vehicle():
    for bad in ([1, 2, 3], "truck", 42):
        with pytest.raises(TypeError):
            cba.plot_vehicle(bad)


def test_plot_vehicle_labels_follow_units():
    veh = cba.VehicleLibrary.US.get_hl93_truck()
    ax_si = cba.plot_vehicle(veh, units="SI", show=False)
    assert ax_si.get_xlabel() == "Position along vehicle (m)"
    ax_us = cba.plot_vehicle(veh, units="US-ft", show=False)
    assert ax_us.get_xlabel() == "Position along vehicle (ft)"
    # a force label carries the (relabelled) unit
    labels = [t.get_text() for t in ax_us.texts if t.get_text()]
    assert any("kip" in s for s in labels)
    plt.close("all")


def test_plot_vehicle_single_axle():
    ax = cba.plot_vehicle(cba.VehicleLibrary.AU.get_a160(), show=False)  # 1 axle
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


# --------------------------------------------------------------------------- #
# Envelopes.plot_coincidents
# --------------------------------------------------------------------------- #
def _env():
    ba = cba.BeamAnalysis([20.0, 20.0], 1e8, [-1, 0, -1, 0, -1, 0])
    return cba.BridgeAnalysis(ba, cba.VehicleLibrary.AU.get_m1600(6.25)).run_vehicle(
        2.0
    )


def test_plot_coincidents_structure():
    fig, axs = _env().plot_coincidents(show=False)
    assert len(axs) == 2  # the two primary (left-axis) panels
    assert len(fig.axes) == 4  # plus the two coincident twins
    # moment panel is sagging-positive (inverted)
    assert axs[0].yaxis_inverted()
    # each primary panel carries the envelope max/min + zero line
    assert len(axs[0].lines) >= 3 and len(axs[1].lines) >= 3
    plt.close(fig)


def test_plot_coincidents_twin_has_coincident_curves():
    fig, axs = _env().plot_coincidents(show=False)
    # the twins (right axes) carry the coincident curves
    twins = [a for a in fig.axes if a not in list(axs)]
    assert len(twins) == 2
    assert all(len(t.lines) >= 2 for t in twins)
    plt.close(fig)
