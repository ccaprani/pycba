"""
Basic tests for PyCBA operation
"""

import pytest
import numpy as np
import pycba as cba


def test_2span_udl():
    """
    Execute a two-span beam analysis and check the reaction results
    """

    L = [7.5, 7.0]
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, 0, -1, 0, -1, 0]
    LM = [[1, 1, 20, 0, 0], [2, 1, 20, 0, 0]]

    beam_analysis = cba.BeamAnalysis(L, EI, R, LM)
    out = beam_analysis.analyze()
    assert out == 0

    r = beam_analysis.beam_results.R
    assert r == pytest.approx([57.41666667, 181.42261905, 51.16071429])

    dmax = max(beam_analysis.beam_results.results.D)
    dmin = min(beam_analysis.beam_results.results.D)
    assert [dmax, dmin] == pytest.approx(
        [1.0118938958333364e-05, -0.0020629648925781247], abs=1e-6
    )


def test_2span_pl_ml_fixed():
    """
    Execute a two-span beam analysis and check the reaction results
    """

    L = [5.0, 5.0]
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, 0, -1, 0, -1, -1]
    LM = [[1, 2, 50, 3, 0], [2, 4, 50, 2, 0]]

    beam_analysis = cba.BeamAnalysis(L, EI, R, LM)
    out = beam_analysis.analyze()
    assert out == 0

    r = beam_analysis.beam_results.R
    assert r == pytest.approx([14.0, 57.6, -21.6, 28.0])

    dmax = max(beam_analysis.beam_results.results.D)
    dmin = min(beam_analysis.beam_results.results.D)
    assert [dmax, dmin] == pytest.approx(
        [0.00017414416666666662, -0.00042251493055555565], abs=1e-6
    )


def test_3span_diff_settlement():
    L = [15, 15, 15]
    EI = 30 * np.array([500e8, 500e8, 500e8]) * 1e-6  # kNm2
    R = [-1, 0, 1e8, 0, 1e8, 0, -1, 0]
    LM = [[1, 1, 20, 0, 0], [2, 1, 20, 0, 0], [3, 1, 20, 0, 0]]

    beam_analysis = cba.BeamAnalysis(L, EI, R, LM)
    out = beam_analysis.analyze()
    assert out == 0

    r = beam_analysis.beam_results.R
    assert r == pytest.approx([120.00175999, 120.00175999])

    dmax = max(beam_analysis.beam_results.results.D)
    dmin = min(beam_analysis.beam_results.results.D)
    assert [dmax, dmin] == pytest.approx(
        [0.0002778734578837245, -0.004648274177216889], abs=1e-6
    )


def test_3span_subframe():
    L = [6, 8, 6]
    EI = 30 * np.array([50e8, 50e8, 50e8]) * 1e-6
    R = [-1, 486e9, -1, 486e9, -1, 486e9, -1, 486e9]
    LM = [[1, 1, 10, 0, 0], [2, 1, 20, 0, 0], [3, 1, 10, 0, 0]]

    beam_analysis = cba.BeamAnalysis(L, EI, R, LM)
    out = beam_analysis.analyze()
    assert out == 0

    r = beam_analysis.beam_results.R
    assert r == pytest.approx([29.99999451, 110.00000549, 110.00000549, 29.99999451])

    dmax = max(beam_analysis.beam_results.results.D)
    dmin = min(beam_analysis.beam_results.results.D)
    assert [dmax, dmin] == pytest.approx([0.0, -0.0014222225377228067], abs=1e-6)


def test_4span_posttensioned():
    L = [6, 8, 6, 8]
    EI = 30 * np.array([100e8, 100e8, 100e8, 100e8]) * 1e-6  # kNm2
    R = [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0]
    LM = [
        [1, 3, 20, 0, 1],
        [1, 3, -10, 1, 4],
        [1, 3, 20, 5, 1],
        [1, 4, 50, 0, 0],
        [2, 3, 20, 0, 1.5],
        [2, 3, -10, 1.5, 5],
        [2, 3, 20, 6.5, 1.5],
        [3, 3, 20, 0, 1],
        [3, 3, -10, 1, 4],
        [3, 3, 20, 5, 1],
        [4, 3, 20, 0, 1.5],
        [4, 3, -10, 1.5, 5],
        [4, 3, 20, 6.5, 1.5],
        [4, 4, -75, 8, 0],
    ]
    beam_analysis = cba.BeamAnalysis(L, EI, R, LM)
    out = beam_analysis.analyze()
    assert out == 0

    r = beam_analysis.beam_results.R
    assert r == pytest.approx(
        [14.87359893, -13.64621406, 15.69634808, -17.62411672, 20.70038377]
    )

    dmax = max(beam_analysis.beam_results.results.D)
    dmin = min(beam_analysis.beam_results.results.D)
    assert [dmax, dmin] == pytest.approx(
        [0.0012715273282923978, -0.00013897273612280755], abs=1e-6
    )


def test_2span_pinned():
    w = 20
    L = [10, 10]
    EI = 30 * np.array([600e7, 600e7]) * 1e-6
    eType = [2, 1]
    R = [-1, 0, -1, 0, -1, 0]
    LM = [[1, 1, w, 0, 0], [2, 1, w, 0, 0]]
    beam_analysis = cba.BeamAnalysis(L, EI, R, LM, eType)
    out = beam_analysis.analyze()
    assert out == 0

    r = beam_analysis.beam_results.R
    assert r == pytest.approx([w * 5, w * 10, w * 5])

    dmax = max(beam_analysis.beam_results.results.D)
    dmin = min(beam_analysis.beam_results.results.D)
    assert [dmax, dmin] == pytest.approx([0, -5 * w * L[0] ** 4 / (384 * EI[0])])


def test_3span_hinge():
    L = [5, 5, 10]
    EI = 30 * 600e7 * np.ones(len(L)) * 1e-6
    eType = [2, 1, 1]
    R = [-1, -1, 0, 0, -1, 0, -1, 0]
    LM = [[3, 2, 20, 5, 0]]
    beam_analysis = cba.BeamAnalysis(L, EI, R, LM, eType)
    out = beam_analysis.analyze()
    assert out == 0

    r = beam_analysis.beam_results.R
    assert r == pytest.approx([-3.75, -18.75, 15.625, 8.125])

    dmax = max(beam_analysis.beam_results.results.D)
    dmin = min(beam_analysis.beam_results.results.D)
    assert [dmax, dmin] == pytest.approx(
        [0.0008680555555555557, -0.0016677326388888887], abs=1e-6
    )


def test_3span_hinge_il():
    L = [5, 5, 10]
    EI = 30 * 600e7 * np.ones(len(L)) * 1e-6
    eType = [2, 1, 1]
    R = [-1, -1, 0, 0, -1, 0, -1, 0]

    ils = cba.InfluenceLines(L, EI, R, eType)
    ils.create_ils(step=0.05)
    (x, y) = ils.get_il(15.0, "V")

    assert [min(y), max(y)] == pytest.approx([-0.4009469062500001, 0.59375], abs=1e-6)


def test_moment_load():
    L = [10.0]
    EI = 30 * 600e7 * np.ones(len(L)) * 1e-6
    eType = [1]
    R = [-1, 0, -1, 0]

    for a in [0, 5, 10]:
        LM = [[1, 4, 10, a, 0]]
        beam_analysis = cba.BeamAnalysis(L, EI, R, LM, eType)
        out = beam_analysis.analyze()
        assert out == 0

        # Check deflection closes
        d = beam_analysis.beam_results.D[[0, 2]]
        assert d == pytest.approx([0.0, 0.0])
