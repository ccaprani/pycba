import pytest
import numpy as np
import pycba as cba


def test_sum_compatible_envelopes():
    """Test summing two compatible envelopes from load patterning and vehicle analysis"""
    L = [30, 30, 30]
    EI = 30 * 10e9 * 1e-6
    R = [-1, 0, -1, 0, -1, 0, -1, 0]

    bridge = cba.BeamAnalysis(L, EI, R)

    # Dead + permanent loads (factored)
    LMd = [
        [1, 1, 12.5 * 1.35 + 50 * 1.5, 0, 0],
        [2, 1, 12.5 * 1.35 + 50 * 1.5, 0, 0],
        [3, 1, 12.5 * 1.35 + 50 * 1.5, 0, 0],
    ]

    # Variable UDL loads
    LMq = [
        [1, 1, 13.5, 0, 0],
        [2, 1, 13.5, 0, 0],
        [3, 1, 13.5, 0, 0],
    ]

    # Create load pattern envelope
    loadpattern = cba.LoadPattern(bridge)
    loadpattern.set_dead_loads(LMd, 1.0, 1.0)
    loadpattern.set_live_loads(LMq, 1.50, 0)
    env_udl = loadpattern.analyze()

    # Create vehicle envelope
    vehicle = cba.Vehicle([1.20], [600, 600])
    bridge_analysis = cba.BridgeAnalysis(bridge, vehicle)
    env_veh = bridge_analysis.run_vehicle(0.5)

    # Sum the envelopes
    env_sum = cba.Envelopes.zero_like(env_udl)
    env_sum.sum(env_udl)
    env_sum.sum(env_veh)

    # Verify: the summed envelope equals the element-wise sum of the individual ones
    assert env_sum.Mmax == pytest.approx(env_udl.Mmax + env_veh.Mmax)
    assert env_sum.Mmin == pytest.approx(env_udl.Mmin + env_veh.Mmin)
    assert env_sum.Vmax == pytest.approx(env_udl.Vmax + env_veh.Vmax)
    assert env_sum.Vmin == pytest.approx(env_udl.Vmin + env_veh.Vmin)
    assert env_sum.Rmaxval == pytest.approx(env_udl.Rmaxval + env_veh.Rmaxval)
    assert env_sum.Rminval == pytest.approx(env_udl.Rminval + env_veh.Rminval)


def test_sum_incompatible_envelopes():
    """Test that summing envelopes from different beam geometries raises ValueError"""
    L1 = [30, 30, 30]
    EI1 = 30 * 10e9 * 1e-6
    R1 = [-1, 0, -1, 0, -1, 0, -1, 0]
    bridge1 = cba.BeamAnalysis(L1, EI1, R1)

    L2 = [20, 20]
    EI2 = 30 * 10e9 * 1e-6
    R2 = [-1, 0, -1, 0, -1, 0]
    bridge2 = cba.BeamAnalysis(L2, EI2, R2)

    LMd1 = [[1, 1, 12.5, 0, 0], [2, 1, 12.5, 0, 0], [3, 1, 12.5, 0, 0]]
    LMq1 = [[1, 1, 13.5, 0, 0], [2, 1, 13.5, 0, 0], [3, 1, 13.5, 0, 0]]

    LMd2 = [[1, 1, 12.5, 0, 0], [2, 1, 12.5, 0, 0]]
    LMq2 = [[1, 1, 13.5, 0, 0], [2, 1, 13.5, 0, 0]]

    lp1 = cba.LoadPattern(bridge1)
    lp1.set_dead_loads(LMd1, 1.35, 1.0)
    lp1.set_live_loads(LMq1, 1.50, 0)
    env1 = lp1.analyze()

    lp2 = cba.LoadPattern(bridge2)
    lp2.set_dead_loads(LMd2, 1.35, 1.0)
    lp2.set_live_loads(LMq2, 1.50, 0)
    env2 = lp2.analyze()

    env_sum = cba.Envelopes.zero_like(env1)
    with pytest.raises(ValueError, match="Cannot sum with an inconsistent envelope"):
        env_sum.sum(env2)
