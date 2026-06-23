# -*- coding: utf-8 -*-
"""Tests for the display unit-system layer (labels and deflection scaling)."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import pycba as cba
from pycba import units as U


@pytest.fixture(autouse=True)
def _restore_default_units():
    # Keep the global default from leaking between tests.
    saved = U.get_units()
    yield
    U.set_units(saved)


def _analysed_beam():
    L = [5.0, 6.0]
    EI = 1e8 * np.ones(len(L))
    R = [-1, 0, -1, 0, -1, 0]
    ba = cba.BeamAnalysis(L, EI, R, LM=[[1, 1, 20, 0, 0], [2, 2, 50, 3]])
    ba.analyze()
    return ba


def test_default_is_si_kn_m():
    assert U.get_units() is U.SI
    assert U.SI.moment_axis == "Bending Moment (kNm)"
    assert U.SI.shear_axis == "Shear Force (kN)"
    assert U.SI.deflection_axis == "Deflection (mm)"
    assert U.SI.distance_axis == "Distance along beam (m)"
    assert U.SI.disp_scale == 1000.0


def test_resolve_aliases_case_insensitive():
    assert U.resolve("SI") is U.SI
    assert U.resolve("eu") is U.SI
    assert U.resolve("AUS") is U.SI
    assert U.resolve("US") is U.US_KIP_FT
    assert U.resolve("us-ft") is U.US_KIP_FT
    assert U.resolve("Kip-In") is U.US_KIP_IN
    assert U.resolve("N-mm") is U.SI_N_MM
    assert U.resolve("none") is U.NONE
    assert U.resolve(None) is U.get_units()
    assert U.resolve(U.US_KIP_FT) is U.US_KIP_FT


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        U.resolve("furlongs")


def test_none_drops_unit_suffixes_and_value_units():
    assert U.NONE.moment_axis == "Bending Moment"
    assert U.NONE.deflection_axis == "Deflection"
    assert U.NONE.distance_axis == "Distance along beam"
    assert U.NONE.fmt_force(50) == "50"
    assert U.NONE.fmt_distributed(5, 8) == "5→8"


def test_value_formatting():
    assert U.SI.fmt_force(50) == "50 kN"
    assert U.SI.fmt_moment(30) == "30 kNm"
    assert U.SI.fmt_distributed(20) == "20 kN/m"
    assert U.SI.fmt_distributed(5, 5) == "5 kN/m"
    assert U.SI.fmt_distributed(5, 8) == "5→8 kN/m"
    assert U.US_KIP_FT.fmt_force(3.5) == "3.5 kip"


def test_set_units_global_default():
    cba.set_units("US-ft")
    assert U.get_units() is U.US_KIP_FT
    ba = _analysed_beam()
    fig, axs = ba.plot_results(show=False)
    assert axs[1].get_ylabel() == "Bending Moment (kip·ft)"
    assert axs[3].get_ylabel() == "Deflection (in)"
    plt.close(fig)


def test_per_plot_override_beats_global():
    cba.set_units("SI")
    ba = _analysed_beam()
    fig, axs = ba.plot_results(show=False, units="none")
    assert axs[1].get_ylabel() == "Bending Moment"
    assert axs[3].get_xlabel() == "Distance along beam"
    plt.close(fig)


def test_deflection_scale_applied():
    ba = _analysed_beam()
    fig_si, axs_si = ba.plot_results(show=False, units="SI")
    fig_nm, axs_nm = ba.plot_results(show=False, units="N-mm")
    # deflection trace is the 2nd line on the deflection panel (after the beam line)
    d_si = axs_si[3].lines[1].get_ydata()
    d_nm = axs_nm[3].lines[1].get_ydata()
    nz = np.abs(d_nm) > 1e-12
    # SI shows native×1000 (mm), N-mm shows native×1 -> ratio 1000
    assert np.allclose(d_si[nz] / d_nm[nz], 1000.0)
    plt.close(fig_si)
    plt.close(fig_nm)


def test_render_load_labels_follow_units():
    beam = cba.BeamAnalysis([6.0], 1e8, [-1, 0, -1, 0], LM=[[1, 1, 20]]).beam
    assert "kN/m" in beam.to_tikz(units="SI")
    us = beam.to_tikz(units="US-ft")
    assert "kip/ft" in us and "kN/m" not in us


def test_bridge_plot_static_labels_follow_units():
    L = [20.0, 25.0]
    EI = 30e11 * 1e-6 * np.ones(len(L))
    R = [-1, 0, -1, 0, -1, 0]
    ba = cba.BridgeAnalysis(
        cba.BeamAnalysis(L, EI, R), cba.Vehicle([1.5, 1.5], [60, 60, 60])
    )
    fig, axs = ba.plot_static(20.0, units="US-ft")
    assert axs[1].get_ylabel() == "Bending Moment (kip·ft)"
    # vehicle caption uses the force unit
    caption = [t.get_text() for t in axs[0].texts if "Vehicle:" in t.get_text()][0]
    assert "kip" in caption and "kN" not in caption
    plt.close(fig)
