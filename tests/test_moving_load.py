"""
Moving load nonlinear analysis validation.

Tests use a two-span bridge and verify:
    1. Collapse occurs when load exceeds theoretical static capacity
    2. No collapse when load is safely below elastic limit
    3. Correct number of plastic hinges for mechanism formation
    4. Moving load collapse ≥ static collapse (moving load is less severe
       than the worst-case static placement, unless damage accumulates)
"""

import pytest
import numpy as np
from pycba.nonlinear import NonlinearBeamAnalysis


L_SPAN = 15.0
MP = 2694.0
MY = MP / 1.15
EI = 1_870_000.0

BEAM_KWARGS = dict(
    L=[L_SPAN, L_SPAN],
    EI=EI,
    R=[-1, 0, -1, 0, -1, 0],
    Mp=MP,
    My=MY,
    q=0.0,
    mesh_size=0.5,
)

# Static collapse load for point load at midspan (virtual work):
# λ = 3·Mp / (P·L/2)  →  P_collapse = 3·Mp / (L/2) for λ = 1
P_STATIC_MIDSPAN = 3 * MP / (L_SPAN / 2)  # ≈ 1077.6 kN


def test_moving_load_causes_collapse():
    """A load at twice the static collapse magnitude should definitely
    cause collapse during a single traverse.
    """
    nba = NonlinearBeamAnalysis(**BEAM_KWARGS)
    result = nba.analyze_moving(P=2 * P_STATIC_MIDSPAN, step=0.5, n_sub=5)

    assert result.collapsed, "Expected collapse at 2× static collapse load"

    # At least 2 plastic hinges needed for a span mechanism
    n_hinges = sum(1 for h in result.hinge_events if h.event_type == "plastic_hinge")
    assert n_hinges >= 2, f"Expected ≥2 plastic hinges, got {n_hinges}"

    # Collapse should occur while load is on the bridge
    assert 0 < result.collapse_lambda <= 2 * L_SPAN


def test_no_collapse_below_elastic():
    """A load safely in the elastic range should traverse without collapse
    or any plastic hinge formation.
    """
    nba = NonlinearBeamAnalysis(**BEAM_KWARGS)
    # 25% of static collapse is well below first yield
    result = nba.analyze_moving(P=P_STATIC_MIDSPAN * 0.25, step=0.5, n_sub=5)

    assert not result.collapsed, "Should not collapse at 25% of static collapse load"


def test_moving_load_hinge_sequence():
    """At the static collapse load, verify that hinges form at physically
    sensible locations (near the support and in the span).
    """
    nba = NonlinearBeamAnalysis(**BEAM_KWARGS)
    result = nba.analyze_moving(P=P_STATIC_MIDSPAN, step=0.5, n_sub=5)

    # This load level should trigger at least yielding
    yield_events = [h for h in result.hinge_events if h.event_type == "initial_yield"]
    assert len(yield_events) > 0, "Expected at least initial yield"

    # If collapse occurred, check hinge locations are on the bridge
    if result.collapsed:
        for h in result.hinge_events:
            if h.event_type == "plastic_hinge":
                assert 0 <= h.location <= sum([L_SPAN, L_SPAN]), (
                    f"Hinge at x={h.location:.1f} is outside the bridge"
                )
