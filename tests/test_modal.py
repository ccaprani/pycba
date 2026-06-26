# -*- coding: utf-8 -*-
"""
Tests for free-vibration (modal) analysis, validated against the classical
analytic natural frequencies of prismatic beams.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pycba as cba


L, EI, m = 10.0, 1.0, 1.0
C = np.sqrt(EI / m)


def test_simply_supported_frequencies():
    res = cba.BeamAnalysis([L], EI, [-1, 0, -1, 0]).modal(mass=m, n_modes=4)
    exact = np.array([(n * np.pi / L) ** 2 * C for n in (1, 2, 3, 4)])
    assert np.allclose(res.omega, exact, rtol=2e-3)
    assert np.allclose(res.f, res.omega / (2 * np.pi))


def test_cantilever_frequencies():
    res = cba.BeamAnalysis([L], EI, [-1, -1, 0, 0]).modal(mass=m, n_modes=3)
    bl = np.array([1.8751, 4.69409, 7.85476])  # beta_n * L
    assert np.allclose(res.omega, (bl / L) ** 2 * C, rtol=2e-3)


def test_fixed_fixed_frequencies():
    res = cba.BeamAnalysis([L], EI, [-1, -1, -1, -1]).modal(mass=m, n_modes=3)
    bl = np.array([4.73004, 7.85320, 10.9956])
    assert np.allclose(res.omega, (bl / L) ** 2 * C, rtol=2e-3)


def test_two_span_continuous():
    # Two equal SS spans: the lowest mode is the antisymmetric one whose half is
    # an SS span, so omega_1 == (pi/Lspan)^2 sqrt(EI/m).
    res = cba.BeamAnalysis([L, L], EI, [-1, 0, -1, 0, -1, 0]).modal(mass=m, n_modes=2)
    assert res.omega[0] == pytest.approx((np.pi / L) ** 2 * C, rel=2e-3)


def test_frequencies_ascending_and_periods():
    res = cba.BeamAnalysis([L], EI, [-1, 0, -1, 0]).modal(mass=m, n_modes=5)
    assert np.all(np.diff(res.omega) >= 0)
    assert np.allclose(res.periods, 1.0 / res.f)
    assert res.n_modes == 5


def test_per_span_mass():
    res = cba.BeamAnalysis([L, L], EI, [-1, 0, -1, 0, -1, 0]).modal(
        mass=[m, m], n_modes=2
    )
    assert res.omega[0] == pytest.approx((np.pi / L) ** 2 * C, rel=2e-3)


def test_mode_shape_and_plot():
    res = cba.BeamAnalysis([L], EI, [-1, 0, -1, 0]).modal(mass=m, n_modes=2)
    x, v = res.mode_shape(0)
    assert x[0] == 0.0 and x[-1] == pytest.approx(L)
    assert np.max(np.abs(v)) == pytest.approx(1.0)  # normalised
    # first SS mode is a single half-sine, peaking mid-span
    assert abs(v[np.argmin(np.abs(x - L / 2))]) == pytest.approx(1.0, abs=1e-3)
    ax = res.plot([0, 1])
    assert ax.has_data()
    plt.close("all")


def test_spring_support_modal_runs():
    ba = cba.BeamAnalysis([L], EI, [1e3, 0, 1e3, 0])  # vertical springs both ends
    res = ba.modal(mass=m, n_modes=3)
    assert np.all(np.isfinite(res.omega)) and res.omega[0] > 0


# ---------------------------------------------------------------------------
# Unsupported combinations
# ---------------------------------------------------------------------------
def test_nonprismatic_raises():
    from pycba.section import SectionEI

    sec = SectionEI([("linear", [0.0, L], [2.0, 1.0])])
    with pytest.raises(NotImplementedError):
        cba.BeamAnalysis([L], sec, [-1, 0, -1, 0]).modal(mass=m)


def test_timoshenko_raises():
    with pytest.raises(NotImplementedError):
        cba.BeamAnalysis([L], EI, [-1, 0, -1, 0], GAv=1e3).modal(mass=m)


def test_released_raises():
    with pytest.raises(NotImplementedError):
        cba.BeamAnalysis([L], EI, [-1, 0, -1, 0], eletype=[2]).modal(mass=m)


def test_mass_length_mismatch_raises():
    with pytest.raises(ValueError):
        cba.BeamAnalysis([L, L], EI, [-1, 0, -1, 0, -1, 0]).modal(mass=[m, m, m])
