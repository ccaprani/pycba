# -*- coding: utf-8 -*-
"""
Tests for shear-point stations and the corrected critical-shear extraction
for moving-load bridge assessment (roadmap item: "Add shear points for moving
load and correct extraction of critical shears for bridge assessment").
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import pycba as cba
from pycba.section import SectionEI


EI = 30e11 * 1e-6  # kNm^2, a typical bridge value


def _simple_span(L=20.0):
    return cba.BeamAnalysis([L], EI, [-1, 0, -1, 0])


# ---------------------------------------------------------------------------
# Default path is untouched
# ---------------------------------------------------------------------------
def test_default_grid_bit_for_bit():
    """Requesting no shear points must leave the evaluation grid identical."""
    bridge = _simple_span()
    ba = cba.BridgeAnalysis(bridge)
    ba.add_vehicle(np.array([]), np.array([100.0]))

    env_base = ba.run_vehicle(0.5)
    env_none = ba.run_vehicle(0.5, shear_points=None)

    assert np.array_equal(env_base.x, env_none.x)
    assert np.array_equal(env_base.Vmax, env_none.Vmax)
    assert np.array_equal(env_base.Mmax, env_none.Mmax)
    # No leakage onto the underlying analysis between runs.
    assert ba.ba.shear_points is None


# ---------------------------------------------------------------------------
# Both-sided capture of the shear discontinuity
# ---------------------------------------------------------------------------
def test_shear_point_captures_both_sides_static():
    """A station pair at a coincident point load recovers both shear limits."""
    L, P, a = 20.0, 100.0, 8.0
    ba = _simple_span(L)
    ba.add_pl(1, P, a)
    ba.shear_points = {0: np.array([a])}
    ba.analyze()

    res = ba.beam_results.results
    near = np.abs(res.x - a) < 1e-6
    V = res.V[near]

    b = L - a
    assert V.max() == pytest.approx(P * b / L)  # left limit = +60
    assert V.min() == pytest.approx(P * b / L - P)  # right limit = -40
    assert V.max() - V.min() == pytest.approx(P)  # the jump equals the axle load


def test_shear_point_exceeds_coarse_grid():
    """A shear point off the coarse grid captures more than the nearest node."""
    L, P, a = 20.0, 100.0, 7.37  # a not on the dx grid
    # Coarse grid only (no shear point)
    ba0 = _simple_span(L)
    ba0.add_pl(1, P, a)
    ba0.analyze()
    r0 = ba0.beam_results.results
    coarse_peak = np.abs(r0.V).max()

    # With a shear point exactly at the load
    ba1 = _simple_span(L)
    ba1.add_pl(1, P, a)
    ba1.shear_points = {0: np.array([a])}
    ba1.analyze()
    r1 = ba1.beam_results.results
    exact_left = P * (L - a) / L
    near = np.abs(r1.x - a) < 1e-6
    assert r1.V[near].max() == pytest.approx(exact_left)
    # The exact left limit is at least as large as anything the coarse grid saw.
    assert exact_left >= coarse_peak - 1e-9


# ---------------------------------------------------------------------------
# Moving-load critical shear matches closed form
# ---------------------------------------------------------------------------
def test_moving_shear_point_analytical():
    """Enveloped shear at a section, single axle on a simple span, vs statics."""
    L, P, d = 20.0, 100.0, 5.0
    bridge = _simple_span(L)
    ba = cba.BridgeAnalysis(bridge)
    ba.add_vehicle(np.array([]), np.array([P]))

    env = ba.run_vehicle(0.05, shear_points=d)
    cv = ba.critical_values(env)
    sp = cv["shear_points"][d]

    # Max shear at section d: axle just to the right -> V = +P (L - d)/L
    # Min shear at section d: axle just to the left  -> V = -P d / L
    assert sp["Vmax"] == pytest.approx(P * (L - d) / L, abs=0.5)  # +75
    assert sp["Vmin"] == pytest.approx(-P * d / L, abs=0.5)  # -25
    assert sp["Vmax"] >= sp["Vmax_right"]  # left-limit governs the max here


# ---------------------------------------------------------------------------
# resolve_shear_points helper
# ---------------------------------------------------------------------------
def test_resolve_points_to_local_coords():
    bridge = cba.BeamAnalysis([10.0, 10.0], EI, [-1, 0, -1, 0, -1, 0])
    sp, xg = cba.resolve_shear_points(bridge.beam, points=[5.0, 12.0])

    assert set(xg) == {5.0, 12.0}
    assert sp[0] == pytest.approx([5.0])  # global 5  -> span 0, local 5
    assert sp[1] == pytest.approx([2.0])  # global 12 -> span 1, local 2


def test_resolve_off_deck_dropped():
    bridge = cba.BeamAnalysis([10.0, 10.0], EI, [-1, 0, -1, 0, -1, 0])
    sp, xg = cba.resolve_shear_points(bridge.beam, points=[-1.0, 25.0, 8.0])
    assert set(xg) == {8.0}


def test_resolve_d_from_supports():
    bridge = cba.BeamAnalysis([10.0, 10.0], EI, [-1, 0, -1, 0, -1, 0])
    sp, xg = cba.resolve_shear_points(bridge.beam, d_from_supports=1.0)
    # Supports at x = 0, 10, 20 -> sections at 1, 9, 11, 19 (off-deck dropped).
    assert set(np.round(xg, 6)) == {1.0, 9.0, 11.0, 19.0}


# ---------------------------------------------------------------------------
# The augmented grid stays compatible with the rest of the Envelopes API
# ---------------------------------------------------------------------------
def test_augmented_grid_envelope_api():
    bridge = _simple_span(20.0)
    ba = cba.BridgeAnalysis(bridge)
    ba.add_vehicle(np.array([]), np.array([100.0]))

    env = ba.run_vehicle(0.5, shear_points=[7.0, 13.0])
    # at() de-duplicates stations, so it tolerates the inserted pairs.
    assert "Mmax" in env.at(10.0, attrs=("Mmax",))
    # per_span chunks by actual member station counts.
    assert len(env.per_span("Mmax")) == bridge.beam.no_spans
    # augment with a compatible (same shear-point) envelope must not raise.
    env2 = ba.run_vehicle(0.5, shear_points=[7.0, 13.0])
    env.augment(env2)


def test_shear_points_with_timoshenko():
    """Shear points work on a shear-deformable member; the jump is still P."""
    L, P, a = 20.0, 100.0, 8.0
    ba = cba.BeamAnalysis([L], EI, [-1, 0, -1, 0], GAv=1.0e6)
    ba.add_pl(1, P, a)
    ba.shear_points = {0: np.array([a])}
    ba.analyze()
    res = ba.beam_results.results
    near = np.abs(res.x - a) < 1e-6
    assert near.sum() >= 2  # the pair was inserted
    assert res.V[near].max() - res.V[near].min() == pytest.approx(P, rel=1e-6)


def test_shear_points_with_nonprismatic():
    """Shear points work on a non-prismatic (SectionEI) member."""
    L, P, a = 20.0, 100.0, 8.0
    sec = SectionEI([("linear", [0.0, L], [2 * EI, EI])])
    ba = cba.BeamAnalysis([L], sec, [-1, 0, -1, 0])
    ba.add_pl(1, P, a)
    ba.shear_points = {0: np.array([a])}
    ba.analyze()
    res = ba.beam_results.results
    near = np.abs(res.x - a) < 1e-6
    # Shear is a statics quantity, independent of the EI variation.
    assert res.V[near].max() == pytest.approx(P * (L - a) / L)
    assert res.V[near].max() - res.V[near].min() == pytest.approx(P)
