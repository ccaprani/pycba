# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pycba as cba


def test_maxvals():
    """
    Single span bridge with M1600 load
    """
    L = [37]
    EI = 30 * 1e11 * np.ones(len(L)) * 1e-6
    R = [-1, 0, -1, 0]

    bridge = cba.BeamAnalysis(L, EI, R)
    bridge.npts = 500  # Use more points along the beam members
    vehicle = cba.VehicleLibrary.get_m1600(6.25)
    bridge_analysis = cba.BridgeAnalysis(bridge, vehicle)
    env = bridge_analysis.run_vehicle(0.1)
    cvals = bridge_analysis.critical_values(env)
    pos = cvals["Mmax"]["pos"][0]
    at = cvals["Mmax"]["at"]
    val = cvals["Mmax"]["val"]
    assert pos == pytest.approx(29.1)
    assert at == pytest.approx(20.35)
    assert val == pytest.approx(7809.3)


def test_la():
    """
    Tests 300LA against Appendix C of AS5100.2
    """
    spans = [5, 10, 20, 30, 40, 50, 100]
    bm = []
    sf = []

    for s in spans:
        L = [s]
        EI = 30 * 1e11 * np.ones(len(L)) * 1e-6
        R = [-1, 0, -1, 0]

        bridge = cba.BeamAnalysis(L, EI, R)
        bridge.npts = 500  # Use more points along the beam members
        vehicle = cba.VehicleLibrary.get_la_rail()
        bridge_analysis = cba.BridgeAnalysis(bridge, vehicle)
        env = bridge_analysis.run_vehicle(0.1)
        cvals = bridge_analysis.critical_values(env)
        m = cvals["Mmax"]["val"]
        v = max(cvals["Rmax0"]["val"], cvals["Rmax1"]["val"])
        # now round to nearest 5 as code does
        bm.append(round(m / 5) * 5)
        sf.append(round(v / 5) * 5)
    assert bm == pytest.approx([705, 2400, 6300, 12515, 21555, 32560, 126300])
    # assert v == pytest.approx([665,1050,1515,2005,2505,3015,5525])


def test_make_train():
    """
    Train of prime movers and platform
    """
    L = [25, 25]
    EI = 30 * 1e11 * np.ones(len(L)) * 1e-6
    R = [-1, 0, -1, 0, -1, 0]
    bridge = cba.BeamAnalysis(L, EI, R)

    prime_mover = cba.Vehicle(
        axle_spacings=np.array([3.2, 1.2]),
        axle_weights=np.array([6.5, 9.25, 9.25]) * 9.81,
    )
    platform_trailer = cba.Vehicle(
        axle_spacings=np.array(
            [
                1.8,
            ]
            * 9
        ),
        axle_weights=np.array([12] * 10) * 9.81,
    )

    inter_spaces = [
        np.array([5.0, 6.3, 8.0, 6.0, 4.8]),
        np.array([4.8, 6.0, 7.5, 6.0, 5.0]),
        np.array([5.0, 6.3, 8.0, 6.3, 5.0]),
    ]

    envs = []
    for s in inter_spaces:
        vehicle = cba.make_train(
            [prime_mover] * 2 + [platform_trailer] * 2 + [prime_mover] * 2, s
        )
        bridge_analysis = cba.BridgeAnalysis(bridge, vehicle)
        envs.append(bridge_analysis.run_vehicle(0.1))

    envenv = cba.Envelopes.zero_like(envs[0])
    for e in envs:
        envenv.augment(e)

    assert envenv.Rmaxval == pytest.approx(
        [708.09969787, 1606.84763766, 694.89625861], abs=1e-6
    )
    assert envenv.Rminval == pytest.approx([-41.9197831, 0.0, -47.23971016], abs=1e-6)


def test_coincident_simple_span():
    """
    Verify coincident V at Mmax matches a static analysis at the critical
    truck position.
    """
    L = [20]
    EI = 30 * 1e11 * np.ones(len(L)) * 1e-6
    R = [-1, 0, -1, 0]

    bridge = cba.BeamAnalysis(L, EI, R)
    vehicle = cba.Vehicle(np.array([4, 4]), np.array([50, 40, 30]))
    bridge_analysis = cba.BridgeAnalysis(bridge, vehicle)
    env = bridge_analysis.run_vehicle(0.5)

    # Find location of Mmax
    i_mmax = env.Mmax.argmax()

    # Get critical truck position from critical_values
    cvals = bridge_analysis.critical_values(env)
    pos_mmax = cvals["Mmax"]["pos"][0]

    # Re-run static analysis at critical position
    static_res = bridge_analysis.static_vehicle(pos_mmax)
    V_at_mmax_point = static_res.results.V[i_mmax]

    assert env.Vco_Mmax[i_mmax] == pytest.approx(V_at_mmax_point, abs=1e-6)


def test_coincident_at_every_point():
    """
    Brute-force: for each point with a significant Mmax, verify Vco_Mmax
    matches V from the analysis that produced Mmax at that point. Uses
    argmax on stacked results to identify the governing analysis.
    """
    L = [10, 10]
    EI = 30 * 1e11 * np.ones(len(L)) * 1e-6
    R = [-1, 0, -1, 0, -1, 0]

    bridge = cba.BeamAnalysis(L, EI, R)
    vehicle = cba.Vehicle(np.array([3]), np.array([100, 100]))
    bridge_analysis = cba.BridgeAnalysis(bridge, vehicle)
    env = bridge_analysis.run_vehicle(0.5)

    # Stack all results into 2D arrays (nres x npts)
    M_all = np.array([res.results.M for res in bridge_analysis.vResults])
    V_all = np.array([res.results.V for res in bridge_analysis.vResults])

    npts = len(env.x)
    # Use argmax to find which result governs at each point
    idx_mmax = np.argmax(M_all, axis=0)

    checked = 0
    for j in range(npts):
        # Only check points with significant Mmax (skip boundary/zero points)
        if env.Mmax[j] > 1.0:
            assert env.Vco_Mmax[j] == pytest.approx(
                V_all[idx_mmax[j], j], abs=1e-6
            )
            checked += 1

    # Ensure we checked a meaningful number of points
    assert checked > npts // 4


def test_coincident_augment():
    """
    Two vehicles produce envelopes which are augmented. Coincident values
    must track across the augmentation.
    """
    L = [25, 25]
    EI = 30 * 1e11 * np.ones(len(L)) * 1e-6
    R = [-1, 0, -1, 0, -1, 0]
    bridge = cba.BeamAnalysis(L, EI, R)

    veh1 = cba.Vehicle(np.array([3.0, 1.2]), np.array([60, 90, 90]))
    ba1 = cba.BridgeAnalysis(bridge, veh1)
    env1 = ba1.run_vehicle(0.5)

    veh2 = cba.Vehicle(np.array([5.0, 1.2, 1.2]), np.array([50, 80, 80, 80]))
    ba2 = cba.BridgeAnalysis(bridge, veh2)
    env2 = ba2.run_vehicle(0.5)

    env_combined = cba.Envelopes.zero_like(env1)
    env_combined.augment(env1)
    env_combined.augment(env2)

    # At the point of global Mmax, the coincident V should come from
    # whichever vehicle produced that Mmax
    i_mmax = env_combined.Mmax.argmax()
    if env1.Mmax[i_mmax] >= env2.Mmax[i_mmax]:
        assert env_combined.Vco_Mmax[i_mmax] == pytest.approx(
            env1.Vco_Mmax[i_mmax], abs=1e-6
        )
    else:
        assert env_combined.Vco_Mmax[i_mmax] == pytest.approx(
            env2.Vco_Mmax[i_mmax], abs=1e-6
        )


def test_coincident_zero_like():
    """
    Verify zero_like creates zeroed coincident arrays.
    """
    L = [20]
    EI = 30 * 1e11 * np.ones(len(L)) * 1e-6
    R = [-1, 0, -1, 0]
    bridge = cba.BeamAnalysis(L, EI, R)
    vehicle = cba.Vehicle(np.array([4, 4]), np.array([50, 40, 30]))
    bridge_analysis = cba.BridgeAnalysis(bridge, vehicle)
    env = bridge_analysis.run_vehicle(0.5)

    zero_env = cba.Envelopes.zero_like(env)
    assert np.all(zero_env.Vco_Mmax == 0)
    assert np.all(zero_env.Vco_Mmin == 0)
    assert np.all(zero_env.Mco_Vmax == 0)
    assert np.all(zero_env.Mco_Vmin == 0)


def test_coincident_critical_values():
    """
    Verify critical_values includes coincident load effects.
    """
    L = [37]
    EI = 30 * 1e11 * np.ones(len(L)) * 1e-6
    R = [-1, 0, -1, 0]

    bridge = cba.BeamAnalysis(L, EI, R)
    bridge.npts = 500
    vehicle = cba.VehicleLibrary.get_m1600(6.25)
    bridge_analysis = cba.BridgeAnalysis(bridge, vehicle)
    env = bridge_analysis.run_vehicle(0.1)
    cvals = bridge_analysis.critical_values(env)

    # New keys exist
    assert "Vco" in cvals["Mmax"]
    assert "Vco" in cvals["Mmin"]
    assert "Mco" in cvals["Vmax"]
    assert "Mco" in cvals["Vmin"]

    # Coincident V at Mmax matches envelope attribute
    i_mmax = env.Mmax.argmax()
    assert cvals["Mmax"]["Vco"] == pytest.approx(env.Vco_Mmax[i_mmax])
