"""
Nonlinear module validation against closed-form plastic collapse solutions.

All theoretical collapse load factors are derived from the virtual work
method (upper-bound plastic analysis).  These are exact for elastic-
perfectly-plastic material (q = 0) and independent of EI.

Beam configuration for most tests:
    Two equal spans of length L, three pin supports (no moment fixity).
    Mp = 432 kNm, My = 376 kNm, EI = 67,035 kNm^2, q = 0.

Validation metrics checked per test case:
    1. Collapse detected (mechanism formed)
    2. Collapse load factor vs closed-form (% error)
    3. Hinge locations match theoretical mechanism
    4. Moments at hinge locations ~ Mp at collapse (equilibrium check)
"""

import pytest
import numpy as np
from pycba.nonlinear import NonlinearBeamAnalysis


L_SPAN = 12.0
MP = 432.0
MY = 376.0
EI = 67035.0

BEAM_KWARGS = dict(
    L=[L_SPAN, L_SPAN],
    EI=EI,
    R=[-1, 0, -1, 0, -1, 0],
    Mp=MP,
    My=MY,
    q=0.0,
)


# ---- Helpers ----

def hinge_locations(result):
    """Return sorted x-coordinates of plastic hinges."""
    return sorted(
        h.location for h in result.hinge_events if h.event_type == "plastic_hinge"
    )


def moments_at_hinges(result):
    """Return |M| at each plastic-hinge node (using final moment distribution)."""
    hinge_nodes = [
        h.node_index for h in result.hinge_events if h.event_type == "plastic_hinge"
    ]
    return [abs(result.final_moments[n]) for n in hinge_nodes]


# ---- Test 1: Point load at midspan ----
#
# Virtual work:
#   Mechanism with hinges at midspan (x = L/2) and interior support (x = L).
#   Left part rotates theta, right part rotates phi.
#   Deflection at midspan = (L/2)*theta, and (L/2)*theta = (L/2)*phi => phi = theta.
#   At midspan hinge: rotation = theta + phi = 2*theta.
#   At support hinge: rotation = phi = theta.
#   Internal work = Mp*2*theta + Mp*theta = 3*Mp*theta
#   External work = lambda*P*(L/2)*theta
#   lambda = 3*Mp / (P*L/2)

def test_point_load_midspan():
    """Point load P=100 kN at midspan of span 1.

    lambda_theory = 3*Mp / (P*L/2) = 3*432 / (100*6) = 2.16
    """
    P = 100.0
    lambda_theory = 3 * MP / (P * L_SPAN / 2)  # = 2.16

    nba = NonlinearBeamAnalysis(**BEAM_KWARGS, mesh_size=0.5)
    result = nba.analyze(LM=[[1, 2, P, L_SPAN / 2]], lambda_max=5.0)

    # 1. Collapse detected
    assert result.collapsed

    # 2. Load factor within 1% of closed-form
    error_pct = abs(result.collapse_lambda - lambda_theory) / lambda_theory * 100
    assert error_pct < 1.0, f"error {error_pct:.2f}% exceeds 1% (got {result.collapse_lambda:.4f}, expected {lambda_theory:.4f})"

    # 3. Hinges at midspan (x=6) and interior support (x=12)
    locs = hinge_locations(result)
    assert any(abs(x - 6.0) <= 1.0 for x in locs), f"No hinge near midspan: {locs}"
    assert any(abs(x - 12.0) <= 1.0 for x in locs), f"No hinge near support B: {locs}"

    # 4. Moments at hinge locations ~ Mp
    for m in moments_at_hinges(result):
        assert m == pytest.approx(MP, rel=0.02), f"|M| = {m:.1f} at hinge, expected ~ {MP}"


# ---- Test 2: UDL on one span ----
#
# Virtual work:
#   UDL w on span 1.  Mechanism: hinges at x = a and x = L (support B).
#   Left part rotates theta, deflection at hinge = a*theta.
#   Right part: (L-a)*phi = a*theta => phi = a*theta/(L-a).
#   At x=a: hinge rotation = theta + phi.
#   At x=L: hinge rotation = phi.
#   Internal = Mp*(theta + phi) + Mp*phi = Mp*(theta + 2*phi)
#            = Mp*theta*(1 + 2a/(L-a)) = Mp*theta*(L+a)/(L-a)
#   External = lambda*w * [a*L*theta/2]  (area under triangular deflection)
#   lambda = 2*Mp*(L+a) / (w*a*L*(L-a))
#
#   Minimise over a: optimal a = L*(sqrt(2)-1).
#   At a_opt: lambda_theory ~ 1.749 for L=12, w=20.
#
#   NOTE: The concentrated-plasticity model requires a hinge to form at a
#   mesh node.  For UDL the theoretical hinge at x* = 4.97 m generally
#   doesn't coincide with a mesh node, so the NFEA result depends on
#   which nearby node the hinge snaps to.  This is inherent to the method
#   and convergence is not monotonic with mesh refinement.

def test_udl_one_span():
    """UDL w=20 kN/m on span 1 only.

    lambda_theory ~ 1.749.  We accept up to 3% error due to hinge-snap
    discretisation inherent in concentrated plasticity models.
    """
    w = 20.0
    a_opt = L_SPAN * (np.sqrt(2) - 1)
    lambda_theory = 2 * MP * (L_SPAN + a_opt) / (w * a_opt * L_SPAN * (L_SPAN - a_opt))

    nba = NonlinearBeamAnalysis(**BEAM_KWARGS, mesh_size=1.0)
    result = nba.analyze(LM=[[1, 1, w]], lambda_max=5.0)

    assert result.collapsed

    error_pct = abs(result.collapse_lambda - lambda_theory) / lambda_theory * 100
    assert error_pct < 3.0, f"error {error_pct:.2f}% exceeds 3% (got {result.collapse_lambda:.4f}, expected {lambda_theory:.4f})"

    # Support hinge at x=12
    locs = hinge_locations(result)
    assert any(abs(x - 12.0) <= 1.5 for x in locs), f"No hinge near support B: {locs}"

    # Sagging hinge should be somewhere in the span (between 3 and 7 m)
    assert any(3.0 <= x <= 7.0 for x in locs), f"No sagging hinge in span: {locs}"


# ---- Test 3: Two point loads ----
#
# P1 = 100 kN at x=4 m, P2 = 75 kN at x=8 m (both in span 1).
#
# Candidate mechanisms (hinges at a load point and at support B):
#
# (a) Hinge at x=4 and x=12:
#   theta at A, deflection at x=4 is 4*theta.
#   Right part (4 to 12, length 8): 8*phi = 4*theta => phi = theta/2.
#   Deflection at x=8 = 4*phi = 2*theta.
#   Internal = Mp*(theta + phi) + Mp*phi = Mp*(3/2*theta + theta/2) = 2*Mp*theta
#   External = lambda*(P1*4*theta + P2*2*theta) = lambda*theta*(400+150) = 550*lambda*theta
#   lambda = 2*Mp/550 = 864/550 = 1.571  <-- CRITICAL (lowest)
#
# (b) Hinge at x=8 and x=12:
#   theta at A, deflection at x=8 = 8*theta.
#   phi = 2*theta, deflection at x=4 = 4*theta.
#   Internal = Mp*3*theta + Mp*2*theta = 5*Mp*theta
#   External = lambda*(100*4*theta + 75*8*theta) = 1000*lambda*theta
#   lambda = 5*432/1000 = 2.16
#
# Mechanism (a) governs: lambda = 1.571

def test_two_point_loads():
    """Two point loads: P1=100 kN at x=4 m, P2=75 kN at x=8 m.

    Critical mechanism: hinges at x=4 m and x=12 m.
    lambda_theory = 2*Mp / (4*P1 + 2*P2) = 864/550 = 1.571
    """
    P1, x1 = 100.0, 4.0
    P2, x2 = 75.0, 8.0

    lambda_theory = 2 * MP / (4 * P1 + 2 * P2)  # = 1.5709

    nba = NonlinearBeamAnalysis(**BEAM_KWARGS, mesh_size=0.5)
    result = nba.analyze(
        LM=[[1, 2, P1, x1], [1, 2, P2, x2]],
        lambda_max=5.0,
    )

    assert result.collapsed

    error_pct = abs(result.collapse_lambda - lambda_theory) / lambda_theory * 100
    assert error_pct < 1.0, f"error {error_pct:.2f}% exceeds 1%"

    # Hinge near x=4 and x=12
    locs = hinge_locations(result)
    assert any(abs(x - 4.0) <= 1.0 for x in locs), f"No hinge near x=4: {locs}"
    assert any(abs(x - 12.0) <= 1.0 for x in locs), f"No hinge near support B: {locs}"


# ---- Test 4: Mesh refinement for point load ----
#
# For point loads coinciding with mesh nodes, the only error source is the
# incremental load stepping.  Verify that all mesh sizes produce small error.

def test_mesh_sizes_point_load():
    """Point load at midspan: all mesh sizes should give < 0.5% error
    (the load point always falls on a node for these mesh sizes).
    """
    P = 100.0
    lambda_theory = 3 * MP / (P * L_SPAN / 2)

    for ms in [2.0, 1.0, 0.5, 0.25]:
        nba = NonlinearBeamAnalysis(**BEAM_KWARGS, mesh_size=ms)
        r = nba.analyze(LM=[[1, 2, P, L_SPAN / 2]], lambda_max=5.0)
        assert r.collapsed, f"No collapse at mesh_size={ms}"
        err = abs(r.collapse_lambda - lambda_theory) / lambda_theory * 100
        assert err < 0.5, f"mesh_size={ms}: error {err:.3f}% exceeds 0.5%"


# ---- Test 5: Elastic regime — no collapse below yield ----

def test_no_collapse_below_yield():
    """At a load factor well below first yield, no hinges should form."""
    P = 100.0
    nba = NonlinearBeamAnalysis(**BEAM_KWARGS, mesh_size=0.5)
    result = nba.analyze(LM=[[1, 2, P, L_SPAN / 2]], lambda_max=0.5)

    assert not result.collapsed
    plastic = [h for h in result.hinge_events if h.event_type == "plastic_hinge"]
    assert len(plastic) == 0, f"Unexpected plastic hinges at low load: {plastic}"


# ---- Test 6: Three-span beam, UDL on all spans ----
#
# Three equal spans of length L_s, pin supports, UDL w on every span.
#
# The critical mechanism is in an OUTER span (propped cantilever: pinned
# at one end, continuous at the other).  This is identical to test 2 (UDL
# on one span of a 2-span beam) because the collapse mechanism is local
# to a single span.
#
# lambda_outer = 2*Mp*(L_s + a_opt) / (w*a_opt*L_s*(L_s - a_opt))
#   where a_opt = L_s*(sqrt(2) - 1)
#
# The centre span (effectively fixed-fixed) has:
# lambda_centre = 16*Mp / (w*L_s^2)
#
# For L_s=10, w=20, Mp=432:
#   lambda_outer ~ 2.518
#   lambda_centre = 3.456
# Outer span governs.

def test_three_span_symmetric_udl():
    """Three equal 10 m spans, UDL on all spans.

    The outer span collapses first (propped cantilever mechanism):
    lambda_theory ~ 2.52.  Centre span (fixed-fixed) would need lambda=3.46.
    """
    L_s = 10.0
    w = 20.0

    # Outer-span mechanism governs
    a_opt = L_s * (np.sqrt(2) - 1)
    lambda_outer = 2 * MP * (L_s + a_opt) / (w * a_opt * L_s * (L_s - a_opt))
    lambda_centre = 16 * MP / (w * L_s ** 2)
    assert lambda_outer < lambda_centre  # confirm outer governs

    nba = NonlinearBeamAnalysis(
        L=[L_s, L_s, L_s],
        EI=EI,
        R=[-1, 0, -1, 0, -1, 0, -1, 0],
        Mp=MP,
        My=MY,
        q=0.0,
        mesh_size=0.5,
    )
    result = nba.analyze(LM=[[1, 1, w], [2, 1, w], [3, 1, w]], lambda_max=10.0)

    assert result.collapsed

    error_pct = abs(result.collapse_lambda - lambda_outer) / lambda_outer * 100
    assert error_pct < 5.0, f"error {error_pct:.2f}% (got {result.collapse_lambda:.4f}, expected {lambda_outer:.4f})"

    # At least one hinge near an interior support
    locs = hinge_locations(result)
    assert any(abs(x - 10.0) <= 1.5 or abs(x - 20.0) <= 1.5 for x in locs), (
        f"No hinge near interior supports: {locs}"
    )
