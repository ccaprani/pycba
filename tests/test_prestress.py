# -*- coding: utf-8 -*-
"""
Tests for the post-tensioning equivalent-load preprocessor (pycba.prestress).

The strongest check is the *determinate* identity: on a statically determinate
member the prestress secondary moment is zero, so the analysed balanced moment
must equal the primary moment ``-F·e(x)`` everywhere.  This validates the whole
emitted load set (curvature UDLs, kink point loads, anchorage moments and the
free-tip anchorage force) for every profile in one shot.
"""

import numpy as np
import pytest
import pycba as cba
from pycba.prestress import (
    Parabola,
    CompoundParabola,
    Harp,
    DoubleHarp,
    equivalent_loads,
)

EI = 1e8
F = 2000.0


def _e_of_x(segs, xs):
    out = np.empty_like(xs, dtype=float)
    for j, x in enumerate(xs):
        for s in segs:
            if s.x1 - 1e-9 <= x <= s.x2 + 1e-9:
                out[j] = s.e1 + s.slope_start() * (x - s.x1) + 0.5 * s.epp * (x - s.x1) ** 2
                break
    return out


def _determinate_maxerr(R, prof, L=8.0, cant=None):
    """max |M_bal(x) - (-F e(x))| over the interior of a determinate span."""
    LM = equivalent_loads(cba.Beam([L], EI, R, eletype=[1]), F, [prof])
    ba = cba.BeamAnalysis([L], EI, R, LM)
    ba.analyze()
    res = ba.beam_results.results
    Mexp = -F * _e_of_x(prof.segments(L, cant), res.x)
    m = (res.x > 0.02 * L) & (res.x < 0.98 * L)
    return np.max(np.abs(res.M[m] - Mexp[m]))


SS = [-1, 0, -1, 0]  # simply supported (determinate span)
CANT_R = [-1, -1, 0, 0]  # fixed-free: right cantilever
CANT_L = [0, 0, -1, -1]  # free-fixed: left cantilever

SPAN_PROFILES = [
    ("Type1 centreline parabola", Parabola(0.10, 0.55, 0.10)),
    ("Type1 asymmetric parabola", Parabola(-0.05, 0.55, 0.15)),
    ("Type2 face-to-face parabola", Parabola(0.10, 0.55, 0.10, c_left=1.0, c_right=1.0)),
    ("Type3 compound parabola", CompoundParabola(0.10, 0.55, 0.10, a=2.0, b=2.0, c=4.0)),
    ("Type4 single harp", Harp(0.10, 0.55, 0.10, a=4.0)),
    ("Type5 face-to-face harp", Harp(0.10, 0.55, 0.10, a=4.0, c_left=1.0, c_right=1.0)),
    ("Type6 double harp", DoubleHarp(0.10, 0.55, 0.55, 0.10, a=2.0, b=6.0)),
    ("Type7 ff double harp", DoubleHarp(0.1, 0.55, 0.55, 0.1, a=2, b=6, c_left=1, c_right=1)),
]

CANT_PROFILES = [
    ("Type8 cantilever parabola", CANT_R, Parabola(0.10, 0.0, 0.45), "right"),
    ("Type9 ff cantilever parabola", CANT_R, Parabola(0.10, 0.0, 0.45, c_left=1.0), "right"),
    ("Type10 cantilever harp", CANT_R, Harp(0.10, 0.30, 0.45, a=4.0), "right"),
    ("Type11 ff cantilever harp", CANT_R, Harp(0.10, 0.30, 0.45, a=4.0, c_left=1.0), "right"),
    ("Type8 left cantilever", CANT_L, Parabola(0.45, 0.0, 0.10), "left"),
    ("Type10 left cant harp", CANT_L, Harp(0.45, 0.30, 0.10, a=4.0), "left"),
]


@pytest.mark.parametrize("label,prof", SPAN_PROFILES, ids=[p[0] for p in SPAN_PROFILES])
def test_span_profiles_reproduce_primary_moment(label, prof):
    assert _determinate_maxerr(SS, prof) < 1.0


@pytest.mark.parametrize(
    "label,R,prof,cant", CANT_PROFILES, ids=[p[0] for p in CANT_PROFILES]
)
def test_cantilever_profiles_reproduce_primary_moment(label, R, prof, cant):
    assert _determinate_maxerr(R, prof, cant=cant) < 1.0


def test_parabola_balanced_udl_and_midspan_moment():
    """Centreline parabola: w = -8 F a / L^2 (upward); SS midspan moment = -F a."""
    L, a = 10.0, 0.4
    LM = equivalent_loads(cba.Beam([L], EI, SS, eletype=[1]), F, [Parabola(0.0, a, 0.0)])
    udl = [r for r in LM if r[1] == 1][0][2]
    assert udl == pytest.approx(-8 * F * a / L**2)
    ba = cba.BeamAnalysis([L], EI, SS, LM)
    ba.analyze()
    res = ba.beam_results.results
    assert res.M[np.argmin(abs(res.x - L / 2))] == pytest.approx(-F * a)


def test_constant_eccentricity_uniform_moment_no_reactions():
    """A straight tendon at constant eccentricity gives uniform -F e, no reactions."""
    L, e = 10.0, 0.3
    LM = equivalent_loads(cba.Beam([L], EI, SS, eletype=[1]), F, [Parabola(e, e, e)])
    ba = cba.BeamAnalysis([L], EI, SS, LM)
    ba.analyze()
    res = ba.beam_results.results
    m = (res.x > 0.05 * L) & (res.x < 0.95 * L)
    assert np.allclose(res.M[m], -F * e, atol=1e-2)
    assert np.allclose(ba.beam_results.R, 0.0, atol=1e-6)


def test_continuous_matches_direct_balanced_udl():
    """On a 2-span continuous beam the PT loads must match the equivalent UDL."""
    L2 = [10.0, 10.0]
    R2 = [-1, 0, -1, 0, -1, 0]
    a = 0.4
    LM = equivalent_loads(
        cba.Beam(L2, EI, R2, eletype=[1, 1]), F, [Parabola(0, a, 0), Parabola(0, a, 0)]
    )
    ba = cba.BeamAnalysis(L2, EI, R2, LM)
    ba.analyze()
    w = -8 * F * a / 100.0
    bau = cba.BeamAnalysis(L2, EI, R2, [[1, 1, w], [2, 1, w]])
    bau.analyze()
    assert np.allclose(
        ba.beam_results.results.M, bau.beam_results.results.M, atol=1e-6
    )


def test_per_span_force_and_unstressed_span():
    L2 = [8.0, 8.0]
    R2 = [-1, 0, -1, 0, -1, 0]
    beam = cba.Beam(L2, EI, R2, eletype=[1, 1])
    LM = equivalent_loads(beam, [1000.0, 1500.0], [Parabola(0, 0.3, 0), None])
    spans = {r[0] for r in LM}
    assert spans == {1}  # only the first (stressed) span produces loads


def test_compound_cantilever_raises():
    with pytest.raises(NotImplementedError):
        equivalent_loads(
            cba.Beam([8.0], EI, CANT_R, eletype=[1]),
            F,
            [CompoundParabola(0.1, 0.5, 0.1, a=2, b=2, c=4)],
        )


def test_profile_count_must_match_spans():
    beam = cba.Beam([8.0, 8.0], EI, [-1, 0, -1, 0, -1, 0], eletype=[1, 1])
    with pytest.raises(ValueError, match="one per span"):
        equivalent_loads(beam, F, [Parabola(0, 0.3, 0)])
