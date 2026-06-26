# -*- coding: utf-8 -*-
"""
Tests for the international code load models added to ``VehicleLibrary``
(extracted from AASHTO, Eurocode, BS5400/CS454, CSA, JTG and AS5100).
Each is checked against the standard published axle weights and spacings.
"""

import numpy as np
import pytest
import pycba as cba

VL = cba.VehicleLibrary


def _check(veh, weights, spacings):
    assert np.allclose(veh.axw, weights)
    assert np.allclose(veh.axs, spacings)
    assert veh.NoAxles == len(weights)
    assert veh.W == pytest.approx(sum(weights))


def test_hl93_truck():
    _check(VL.get_hl93_truck(), [35, 145, 145], [4.3, 4.3])
    _check(VL.get_hl93_truck(rear_spacing=9.0), [35, 145, 145], [4.3, 9.0])
    assert VL.get_hl93_truck().W == pytest.approx(325.0)
    for bad in (4.0, 9.5):
        with pytest.raises(ValueError):
            VL.get_hl93_truck(rear_spacing=bad)


def test_hl93_tandem():
    _check(VL.get_hl93_tandem(), [110, 110], [1.2])


def test_eurocode_lm1():
    _check(VL.get_lm1(), [300, 300], [1.2])
    _check(VL.get_lm1(alpha_Q=0.9), [270, 270], [1.2])


def test_eurocode_lm71():
    _check(VL.get_lm71(), [250, 250, 250, 250], [1.6, 1.6, 1.6])
    assert VL.get_lm71().W == pytest.approx(1000.0)
    _check(VL.get_lm71(alpha=1.21), [302.5] * 4, [1.6, 1.6, 1.6])


def test_csa_cl625():
    _check(VL.get_cl625(), [50, 125, 125, 175, 150], [3.6, 1.2, 6.6, 6.6])
    assert VL.get_cl625().W == pytest.approx(625.0)


def test_bs5400_hb():
    _check(VL.get_hb(), [450, 450, 450, 450], [1.8, 6.0, 1.8])  # 45 units
    _check(VL.get_hb(units=30, inner_spacing=21.0), [300] * 4, [1.8, 21.0, 1.8])


def test_china_jtg_vehicle():
    _check(VL.get_jtg_vehicle(), [30, 120, 120, 140, 140], [3.0, 1.4, 7.0, 1.4])
    assert VL.get_jtg_vehicle().W == pytest.approx(550.0)


def test_as5100_single_axle_wheel():
    a = VL.get_a160()
    assert a.NoAxles == 1 and a.axw[0] == pytest.approx(160.0) and len(a.axs) == 0
    w = VL.get_w80()
    assert w.NoAxles == 1 and w.axw[0] == pytest.approx(80.0)


def test_models_run_in_a_bridge_analysis():
    """Each new model traverses a simple bridge and produces finite effects."""
    ba = cba.BeamAnalysis([20.0, 20.0], 1e8, [-1, 0, -1, 0, -1, 0])
    for getter in (
        VL.get_hl93_truck,
        VL.get_hl93_tandem,
        VL.get_lm1,
        VL.get_lm71,
        VL.get_cl625,
        VL.get_jtg_vehicle,
        VL.get_a160,
    ):
        veh = getter()
        env = cba.BridgeAnalysis(ba, veh).run_vehicle(2.0)
        assert np.all(np.isfinite(env.Mmax)) and env.Mmax.max() > 0
