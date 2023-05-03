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


def test_from_convoy():
    """
    Convoy of prime movers and platform
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
        vehicle = cba.Vehicle.from_convoy(
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
