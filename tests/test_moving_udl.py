# -*- coding: utf-8 -*-
"""
Moving partial UDL and lane-UDL-with-exclusion ("PCBeaman") tests
(roadmap item: "Add moving partial UDL (required for certain code load
models)").  These formalise the exploratory prototype scripts that were
previously at the repo root.
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import pycba as cba
from pycba.analysis import BeamAnalysis
from pycba.bridge import BridgeAnalysis

EI = 1e8  # kNm^2


def _two_span(Ls=(10.0, 10.0)):
    return BeamAnalysis(list(Ls), EI, [-1, 0, -1, 0, -1, 0])


# ---------------------------------------------------------------------------
# Partial-UDL plumbing: an interval split at span boundaries == a full UDL
# ---------------------------------------------------------------------------
def test_interval_udl_split_equals_full():
    """A global UDL interval split into per-span partial UDLs superimposes
    exactly to the same result as native full-span UDLs."""
    W = 10.0
    ba = BridgeAnalysis(_two_span())
    ba.add_vehicle(np.array([]), np.array([0.0]))  # no axle effect

    # The deck-wide UDL [0, 20], split internally at an arbitrary point.
    rows = ba._interval_udl_LM(0.0, 7.3, W) + ba._interval_udl_LM(7.3, 20.0, W)
    split = _two_span()
    split.set_loads(rows)
    split.analyze()

    full = _two_span()
    full.add_udl(1, W)
    full.add_udl(2, W)
    full.analyze()

    assert np.allclose(
        split.beam_results.results.M, full.beam_results.results.M, atol=1e-9
    )


# ---------------------------------------------------------------------------
# run_load_model: vehicle + co-travelling lane UDL
# ---------------------------------------------------------------------------
def test_run_load_model_no_udl_matches_run_vehicle():
    """A gap wider than the deck removes the lane UDL, recovering run_vehicle."""
    Ls = [10.0, 10.0]
    ba = BridgeAnalysis(_two_span(Ls))
    ba.add_vehicle(np.array([]), np.array([200.0]))
    env_model = ba.run_load_model(step=0.25, w_lane=10.0, gap=1e3)

    ba2 = BridgeAnalysis(_two_span(Ls))
    ba2.add_vehicle(np.array([]), np.array([200.0]))
    env_veh = ba2.run_vehicle(step=0.25)

    assert np.allclose(env_model.Mmax, env_veh.Mmax)
    assert np.allclose(env_model.Vmax, env_veh.Vmax)


def test_run_load_model_adds_lane_load():
    """Adding a lane UDL increases the governing sagging moment."""
    ba = BridgeAnalysis(_two_span())
    ba.add_vehicle(np.array([]), np.array([200.0]))
    env_veh = ba.run_vehicle(step=0.25)

    ba2 = BridgeAnalysis(_two_span())
    ba2.add_vehicle(np.array([]), np.array([200.0]))
    env_model = ba2.run_load_model(step=0.25, w_lane=10.0, gap=0.0)

    assert env_model.Mmax.max() > env_veh.Mmax.max()


def test_run_load_model_full_deck_superposition():
    """With the UDL present at every position (single axle, gap=0), each
    position's result equals axle-only + full-deck UDL by linearity."""
    Ls = [10.0, 10.0]
    W, P = 10.0, 200.0
    ba = BridgeAnalysis(_two_span(Ls))
    ba.add_vehicle(np.array([]), np.array([P]))
    ba.run_load_model(step=0.5, w_lane=W, gap=0.0)
    combined = ba.vResults

    ba2 = BridgeAnalysis(_two_span(Ls))
    ba2.add_vehicle(np.array([]), np.array([P]))
    ba2.run_vehicle(step=0.5)
    veh_only = ba2.vResults

    udl = _two_span(Ls)
    udl.add_udl(1, W)
    udl.add_udl(2, W)
    udl.analyze()
    udl_M = udl.beam_results.results.M

    # Check a handful of positions across the traverse.
    for k in range(0, len(combined), 7):
        assert np.allclose(
            combined[k].results.M, veh_only[k].results.M + udl_M, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Lane-UDL exclusion zone (the PCBeaman method) reduces the loaded-span effect
# ---------------------------------------------------------------------------
def test_lane_udl_exclusion_reduces_loaded_span_moment():
    """Excluding the lane UDL under the vehicle reduces the sagging moment on
    the span carrying the vehicle (vs. loading every span)."""
    Ls = [10.0, 10.0]
    W, P = 10.0, 200.0

    no_excl = _two_span(Ls)
    no_excl.add_udl(1, W)
    no_excl.add_udl(2, W)
    no_excl.add_pl(1, P, 5.0)  # vehicle at midspan of span 1
    no_excl.analyze()
    M_no = no_excl.beam_results.results.M

    with_excl = _two_span(Ls)
    with_excl.add_udl(2, W)  # lane UDL excluded from span 1 (under the vehicle)
    with_excl.add_pl(1, P, 5.0)
    with_excl.analyze()
    M_ex = with_excl.beam_results.results.M

    n_span1 = no_excl.beam_results.vRes[0].x.size
    assert M_ex[:n_span1].max() < M_no[:n_span1].max()


# ---------------------------------------------------------------------------
# A36-style UDL patterning: simply-supported spans vs continuous analysis
# ---------------------------------------------------------------------------
def test_simply_supported_span_patterning():
    W, L = 10.0, 10.0

    ss = BeamAnalysis([L], EI, [-1, 0, -1, 0])
    ss.add_udl(1, W)
    ss.analyze()
    assert ss.beam_results.results.M.max() == pytest.approx(W * L**2 / 8)  # 125

    cont = _two_span([L, L])
    cont.add_udl(1, W)
    cont.add_udl(2, W)
    cont.analyze()
    Mc = cont.beam_results.results.M
    # Continuity: interior-support hogging of wL^2/8, reduced midspan sagging.
    assert Mc.min() == pytest.approx(-W * L**2 / 8)
    assert Mc.max() < W * L**2 / 8
