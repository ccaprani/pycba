# -*- coding: utf-8 -*-
"""
Tests for the beam-on-elastic-(Winkler)-foundation feature, validated against
Hetényi's analytic infinite-beam solutions.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import pycba as cba
from pycba.foundation import (
    FoundationElement,
    characteristic_length,
    auto_subdivisions,
)
from pycba.section import SectionEI


def _beta(EI, kf):
    return (kf / (4.0 * EI)) ** 0.25


# ---------------------------------------------------------------------------
# Validation against Hetényi (long beam ~ infinite beam)
# ---------------------------------------------------------------------------
def test_point_load_matches_hetenyi():
    L, EI, kf, P = 20.0, 1.0, 4.0, 1.0  # lambda = 1, so L/lambda = 20 (long)
    beta = _beta(EI, kf)
    ba = cba.BeamAnalysis([L], EI, [0, 0, 0, 0], kf=kf)  # free-free on foundation
    ba.add_pl(1, P, L / 2)
    ba.analyze()
    res = ba.beam_results.results

    i = np.argmin(np.abs(res.x - L / 2))
    assert res.D[i] == pytest.approx(-P * beta / (2 * kf), rel=2e-3)  # downward
    assert np.max(np.abs(res.M)) == pytest.approx(P / (4 * beta), rel=5e-3)


def test_udl_interior_deflection():
    """Far from the ends, a UDL on a long beam compresses the foundation with
    no bending: v -> -w/k."""
    L, EI, kf, w = 20.0, 1.0, 4.0, 2.0
    ba = cba.BeamAnalysis([L], EI, [0, 0, 0, 0], kf=kf)
    ba.add_udl(1, w)
    ba.analyze()
    res = ba.beam_results.results
    i = np.argmin(np.abs(res.x - L / 2))
    assert res.D[i] == pytest.approx(-w / kf, rel=1e-4)


def test_foundation_carries_load_equilibrium():
    """The foundation reaction integrates to the applied load (free-free beam)."""
    L, EI, kf, P = 20.0, 1.0, 4.0, 3.0
    ba = cba.BeamAnalysis([L], EI, [0, 0, 0, 0], kf=kf)
    ba.add_pl(1, P, L / 2)
    ba.analyze()
    res = ba.beam_results.results
    from scipy.integrate import trapezoid

    F_found = trapezoid(kf * (-res.D), res.x)  # sum of distributed soil reaction
    assert F_found == pytest.approx(P, rel=1e-2)


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def test_characteristic_length_and_mesh():
    assert characteristic_length(1.0, 4.0) == pytest.approx(1.0)
    # ~8 sub-elements per characteristic length, clipped to [8, 400]
    assert auto_subdivisions(20.0, 1.0, 4.0) == 160
    assert auto_subdivisions(0.5, 1.0, 4.0) == 8  # n_min
    assert auto_subdivisions(1000.0, 1.0, 4.0) == 400  # n_max


def test_finer_mesh_closer_to_hetenyi():
    L, EI, kf, P = 20.0, 1.0, 4.0, 1.0
    beta = _beta(EI, kf)
    exact = P / (4 * beta)
    err = []
    for n in (20, 80, 320):
        fe = FoundationElement(L, EI, kf, n=n)
        loads = [cba.load.LoadPL(0, P, L / 2)]
        d = np.linalg.solve(fe.k, -fe.ref(loads))
        res = fe.recover(d, loads, npts=800)
        err.append(abs(np.max(np.abs(res.M)) - exact))
    assert err[0] > err[1] > err[2]  # monotone convergence


# ---------------------------------------------------------------------------
# Broadcasting and the default (no-foundation) path
# ---------------------------------------------------------------------------
def test_kf_broadcast_and_per_span():
    b = cba.Beam([5.0, 5.0], 1.0, [-1, 0, -1, 0, -1, 0], eletype=[1, 1], kf=100.0)
    assert b.mbr_kf == [100.0, 100.0]
    b2 = cba.Beam(
        [5.0, 5.0], 1.0, [-1, 0, -1, 0, -1, 0], eletype=[1, 1], kf=[100.0, None]
    )
    assert b2.mbr_kf == [100.0, None]


def test_default_path_unchanged():
    a = cba.BeamAnalysis([6.0, 8.0], 30e3, [-1, 0, -1, 0, -1, 0])
    a.add_udl(1, 10)
    a.add_udl(2, 10)
    a.analyze()
    b = cba.BeamAnalysis([6.0, 8.0], 30e3, [-1, 0, -1, 0, -1, 0], kf=None)
    b.add_udl(1, 10)
    b.add_udl(2, 10)
    b.analyze()
    assert np.array_equal(a.beam_results.results.M, b.beam_results.results.M)
    assert np.array_equal(a.beam_results.results.D, b.beam_results.results.D)


def test_mixed_foundation_and_normal_span():
    """One foundation span and one ordinary span in the same beam."""
    ba = cba.BeamAnalysis([8.0, 8.0], 1e4, [-1, 0, -1, 0, -1, 0], kf=[2000.0, None])
    ba.add_udl(1, 20)
    ba.add_udl(2, 20)
    ba.analyze()
    assert np.all(np.isfinite(ba.beam_results.results.M))


# ---------------------------------------------------------------------------
# Clear errors for unsupported combinations
# ---------------------------------------------------------------------------
def test_nonprismatic_foundation_raises():
    sec = SectionEI([("linear", [0.0, 6.0], [2.0, 1.0])])
    ba = cba.BeamAnalysis([6.0], sec, [0, 0, 0, 0], kf=100.0)
    ba.add_pl(1, 1.0, 3.0)
    with pytest.raises(NotImplementedError):
        ba.analyze()


def test_timoshenko_foundation_raises():
    ba = cba.BeamAnalysis([6.0], 1.0, [0, 0, 0, 0], GAv=1e3, kf=100.0)
    ba.add_pl(1, 1.0, 3.0)
    with pytest.raises(NotImplementedError):
        ba.analyze()


def test_released_foundation_raises():
    ba = cba.BeamAnalysis([6.0], 1.0, [0, 0, 0, 0], eletype=[2], kf=100.0)
    ba.add_pl(1, 1.0, 3.0)
    with pytest.raises(NotImplementedError):
        ba.analyze()


def test_unsupported_load_on_foundation_raises():
    ba = cba.BeamAnalysis([6.0], 1.0, [-1, 0, -1, 0], kf=100.0)
    ba.add_ml(1, 5.0, 3.0)  # moment load not supported on a foundation span
    with pytest.raises(NotImplementedError):
        ba.analyze()
