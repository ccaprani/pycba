# -*- coding: utf-8 -*-
"""
Tests for the region-grouped code load models in ``VehicleLibrary``
(AASHTO, Eurocode, BS5400/CS454, CSA, JTG, AS5100 and the historical NAASRA /
AREA models).  Each is checked against the standard published axle weights and
spacings.
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


# --------------------------------------------------------------------------- #
# United States - AASHTO LRFD + AREA Cooper
# --------------------------------------------------------------------------- #
def test_hl93_truck():
    _check(VL.US.get_hl93_truck(), [35, 145, 145], [4.3, 4.3])
    _check(VL.US.get_hl93_truck(rear_spacing=9.0), [35, 145, 145], [4.3, 9.0])
    assert VL.US.get_hl93_truck().W == pytest.approx(325.0)
    for bad in (4.0, 9.5):
        with pytest.raises(ValueError):
            VL.US.get_hl93_truck(rear_spacing=bad)


def test_hl93_tandem():
    _check(VL.US.get_hl93_tandem(), [110, 110], [1.2])


def test_cooper_e_series():
    e80 = VL.US.get_cooper(80)
    assert e80.NoAxles == 18  # two locomotives, 9 axles each
    # driving axle = 80 kip = 355.86 kN; lead = 40 kip; tender = 52 kip
    assert e80.axw.max() == pytest.approx(80 * 4.4482216, rel=1e-4)
    assert e80.axw.min() == pytest.approx(40 * 4.4482216, rel=1e-4)
    # linear scaling with the E-number; spacings do not scale
    e40 = VL.US.get_cooper(40)
    assert np.allclose(e80.axw, 2.0 * e40.axw)
    assert np.allclose(e80.axs, e40.axs)


# --------------------------------------------------------------------------- #
# Europe - Eurocode EN 1991-2
# --------------------------------------------------------------------------- #
def test_eurocode_lm1():
    _check(VL.EU.get_lm1(), [300, 300], [1.2])
    _check(VL.EU.get_lm1(alpha_Q=0.9), [270, 270], [1.2])


def test_eurocode_lm71():
    _check(VL.EU.get_lm71(), [250, 250, 250, 250], [1.6, 1.6, 1.6])
    assert VL.EU.get_lm71().W == pytest.approx(1000.0)
    _check(VL.EU.get_lm71(alpha=1.21), [302.5] * 4, [1.6, 1.6, 1.6])


# --------------------------------------------------------------------------- #
# Canada / UK / China
# --------------------------------------------------------------------------- #
def test_csa_cl625():
    _check(VL.CA.get_cl625(), [50, 125, 125, 175, 150], [3.6, 1.2, 6.6, 6.6])
    assert VL.CA.get_cl625().W == pytest.approx(625.0)


def test_bs5400_hb():
    _check(VL.UK.get_hb(), [450, 450, 450, 450], [1.8, 6.0, 1.8])  # 45 units
    _check(VL.UK.get_hb(units=30, inner_spacing=21.0), [300] * 4, [1.8, 21.0, 1.8])


def test_china_jtg_vehicle():
    _check(VL.CN.get_jtg_vehicle(), [30, 120, 120, 140, 140], [3.0, 1.4, 7.0, 1.4])
    assert VL.CN.get_jtg_vehicle().W == pytest.approx(550.0)


# --------------------------------------------------------------------------- #
# Australia - AS5100 + historical NAASRA
# --------------------------------------------------------------------------- #
def test_as5100_single_axle_wheel():
    a = VL.Aus.get_a160()
    assert a.NoAxles == 1 and a.axw[0] == pytest.approx(160.0) and len(a.axs) == 0
    w = VL.Aus.get_w80()
    assert w.NoAxles == 1 and w.axw[0] == pytest.approx(80.0)


def test_naasra_t44():
    _check(VL.Aus.get_t44(), [48, 96, 96, 96, 96], [3.7, 1.2, 3.0, 1.2])
    _check(
        VL.Aus.get_t44(variable_spacing=8.0), [48, 96, 96, 96, 96], [3.7, 1.2, 8.0, 1.2]
    )
    assert VL.Aus.get_t44().W == pytest.approx(432.0)
    for bad in (2.9, 8.1):
        with pytest.raises(ValueError):
            VL.Aus.get_t44(variable_spacing=bad)


def test_naasra_ms18():
    _check(VL.Aus.get_ms18(), [35.6, 142.3, 142.3], [4.27, 4.27])
    _check(VL.Aus.get_ms18(rear_spacing=9.14), [35.6, 142.3, 142.3], [4.27, 9.14])
    with pytest.raises(ValueError):
        VL.Aus.get_ms18(rear_spacing=10.0)


def test_as5100_300la_matches_la_rail():
    a = VL.Aus.get_300la(axle_group_count=3)
    b = VL.Aus.get_la_rail(axle_group_count=3, axle_weight=300)
    assert np.allclose(a.axw, b.axw) and np.allclose(a.axs, b.axs)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def test_models_run_in_a_bridge_analysis():
    """Each axle model traverses a simple bridge and produces finite effects."""
    ba = cba.BeamAnalysis([20.0, 20.0], 1e8, [-1, 0, -1, 0, -1, 0])
    for veh in (
        VL.US.get_hl93_truck(),
        VL.US.get_hl93_tandem(),
        VL.US.get_cooper(80),
        VL.EU.get_lm1(),
        VL.EU.get_lm71(),
        VL.CA.get_cl625(),
        VL.CN.get_jtg_vehicle(),
        VL.Aus.get_t44(),
        VL.Aus.get_ms18(),
        VL.Aus.get_a160(),
    ):
        env = cba.BridgeAnalysis(ba, veh).run_vehicle(2.0)
        assert np.all(np.isfinite(env.Mmax)) and env.Mmax.max() > 0
