# -*- coding: utf-8 -*-
"""
Tests for the results query/export ergonomics: ``at()`` point queries and
``to_dataframe()`` / ``to_csv()`` export on ``BeamResults`` and ``Envelopes``.
"""

import numpy as np
import pytest
import pycba as cba


def _ss_udl():
    # simple span, L=10, UDL w=10 -> M_mid = wL^2/8 = 125
    ba = cba.BeamAnalysis([10.0], 1e7, [-1, 0, -1, 0], LM=[[1, 1, 10, 0, 0]])
    ba.analyze()
    return ba


def test_at_point_query_values():
    ba = _ss_udl()
    mid = ba.at(5.0)
    assert set(mid) == {"M", "V", "R", "D"}
    assert mid["M"] == pytest.approx(125.0, rel=1e-3)  # wL^2/8
    assert abs(mid["V"]) == pytest.approx(0.0, abs=1e-6)  # zero shear at mid-span
    # quarter point: V = w*(L/2 - x) = 10*(5-2.5) = 25
    assert abs(ba.at(2.5)["V"]) == pytest.approx(25.0, rel=1e-3)
    # subset of attrs
    assert set(ba.at(5.0, attrs=("M",))) == {"M"}


def test_at_before_analysis_returns_none():
    ba = cba.BeamAnalysis([10.0], 1e7, [-1, 0, -1, 0])
    assert ba.at(5.0) is None
    assert ba.to_dataframe() is None
    assert ba.to_csv("/tmp/never.csv") is None


def test_to_dataframe_columns_and_length():
    ba = _ss_udl()
    df = ba.to_dataframe()
    assert list(df.columns) == ["x", "M", "V", "R", "D"]
    assert len(df) == len(ba.beam_results.results.x)
    # the frame's peak moment matches the analytic mid-span value
    assert df["M"].max() == pytest.approx(125.0, rel=1e-3)
    # same frame via the BeamResults method
    df2 = ba.beam_results.to_dataframe()
    assert df2.equals(df)


def test_to_csv_roundtrip(tmp_path):
    ba = _ss_udl()
    p = tmp_path / "results.csv"
    out = ba.to_csv(p)
    assert out == p and p.exists()
    # header + numeric round-trip (no pandas needed to read it back)
    with open(p) as f:
        header = f.readline().strip()
    assert header == "x,M,V,R,D"
    data = np.loadtxt(p, delimiter=",", skiprows=1)
    assert data.shape[1] == 5
    assert data[:, 1].max() == pytest.approx(125.0, rel=1e-3)


def test_envelopes_export(tmp_path):
    ba = cba.BeamAnalysis([20.0, 20.0], 1e8, [-1, 0, -1, 0, -1, 0])
    veh = cba.VehicleLibrary.US.get_hl93_truck()
    env = cba.BridgeAnalysis(ba, veh).run_vehicle(2.0)

    df = env.to_dataframe()
    assert list(df.columns) == ["x", "Mmax", "Mmin", "Vmax", "Vmin"]
    assert len(df) == len(env.x)
    assert np.allclose(df["x"].to_numpy(), env.x)

    p = tmp_path / "env.csv"
    env.to_csv(p, attrs=("Mmax", "Mmin"))
    with open(p) as f:
        assert f.readline().strip() == "x,Mmax,Mmin"
    arr = np.loadtxt(p, delimiter=",", skiprows=1)
    assert arr.shape[1] == 3


def test_envelopes_at_still_works():
    # the pre-existing Envelopes.at point query is unaffected
    ba = cba.BeamAnalysis([20.0, 20.0], 1e8, [-1, 0, -1, 0, -1, 0])
    env = cba.BridgeAnalysis(ba, cba.VehicleLibrary.US.get_hl93_tandem()).run_vehicle(
        2.0
    )
    vals = env.at(20.0, attrs=("Mmax", "Mmin"))
    assert set(vals) == {"Mmax", "Mmin"}
