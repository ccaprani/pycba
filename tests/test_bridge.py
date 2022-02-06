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
