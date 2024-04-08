"""
Basic tests for PyCBA operation
"""

import pytest
import numpy as np
import pycba as cba


def test_1span_ee():
    """
    Test fixed-fixed beam with point load in the middle
    """

    P = 10  # kN
    L = 10  # m
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, -1, -1, -1]
    LM = [[1, 2, P, 0.5 * L, 0]]

    beam_analysis = cba.BeamAnalysis([L], EI, R, LM)
    out = beam_analysis.analyze()
    assert out == 0

    Ma = beam_analysis.beam_results.results.M[1]
    Mb = beam_analysis.beam_results.results.M[-2]
    Mc = beam_analysis.beam_results.results.M[51]

    assert Ma == pytest.approx(-P * L / 8)
    assert Mb == pytest.approx(-P * L / 8)
    assert Mc == pytest.approx(P * L / 8)


def test_1span_ep():
    """
    Test fixed-pinned beam with point load in the middle
    """

    P = 10  # kN
    L = 10  # m
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, -1, -1, 0]
    LM = [[1, 2, P, 0.5 * L, 0]]

    beam_analysis = cba.BeamAnalysis([L], EI, R, LM)
    out = beam_analysis.analyze()
    assert out == 0

    Ma = beam_analysis.beam_results.results.M[1]
    Mb = beam_analysis.beam_results.results.M[-2]
    Mc = beam_analysis.beam_results.results.M[51]

    assert Ma == pytest.approx(-3 * P * L / 16)
    assert Mb == pytest.approx(0)
    assert Mc == pytest.approx(5 * P * L / 32)


def test_1span_ep_eletype2():
    """
    Test fixed-pinned beam with point load in the middle, using eleType 2
    """

    P = 10  # kN
    L = 10  # m
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, -1, -1, -1]  # Notice, fixed-fixed supports
    LM = [[1, 2, P, 0.5 * L, 0]]

    beam_analysis = cba.BeamAnalysis([L], EI, R, LM, eletype=[2])
    out = beam_analysis.analyze()
    assert out == 0

    Ma = beam_analysis.beam_results.results.M[1]
    Mb = beam_analysis.beam_results.results.M[-2]
    Mc = beam_analysis.beam_results.results.M[51]

    assert Ma == pytest.approx(-3 * P * L / 16)
    assert Mb == pytest.approx(0)
    assert Mc == pytest.approx(5 * P * L / 32)


def test_1span_pe():
    """
    Test pinned-fixed beam with point load in the middle
    """

    P = 10  # kN
    L = 10  # m
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, 0, -1, -1]
    LM = [[1, 2, P, 0.5 * L, 0]]

    beam_analysis = cba.BeamAnalysis([L], EI, R, LM)
    out = beam_analysis.analyze()
    assert out == 0

    Ma = beam_analysis.beam_results.results.M[1]
    Mb = beam_analysis.beam_results.results.M[-2]
    Mc = beam_analysis.beam_results.results.M[51]

    assert Ma == pytest.approx(0)
    assert Mb == pytest.approx(-3 * P * L / 16)
    assert Mc == pytest.approx(5 * P * L / 32)


def test_1span_pe_eletype3():
    """
    Test pinned-fixed beam with point load in the middle, using eleType 3
    """

    P = 10  # kN
    L = 10  # m
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, -1, -1, -1]  # Notice, fixed-fixed supports
    LM = [[1, 2, P, 0.5 * L, 0]]

    beam_analysis = cba.BeamAnalysis([L], EI, R, LM, eletype=[3])
    out = beam_analysis.analyze()
    assert out == 0

    Ma = beam_analysis.beam_results.results.M[1]
    Mb = beam_analysis.beam_results.results.M[-2]
    Mc = beam_analysis.beam_results.results.M[51]

    assert Ma == pytest.approx(0)
    assert Mb == pytest.approx(-3 * P * L / 16)
    assert Mc == pytest.approx(5 * P * L / 32)


def test_1span_pp():
    """
    Test pinned-pinned beam with point load in the middle
    """

    P = 10  # kN
    L = 10  # m
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, 0, -1, 0]
    LM = [[1, 2, P, 0.5 * L, 0]]

    beam_analysis = cba.BeamAnalysis([L], EI, R, LM)
    out = beam_analysis.analyze()
    assert out == 0

    Ma = beam_analysis.beam_results.results.M[1]
    Mb = beam_analysis.beam_results.results.M[-2]
    Mc = beam_analysis.beam_results.results.M[51]

    assert Ma == pytest.approx(0)
    assert Mb == pytest.approx(0)
    assert Mc == pytest.approx(P * L / 4)


def test_1span_pp_eletype4():
    """
    Test pinned-pinned beam with point load in the middle, using eleType 4
    """

    P = 10  # kN
    L = 10  # m
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, -1, -1, -1]  # Notice, fixed-fixed supports
    LM = [[1, 2, P, 0.5 * L, 0]]

    beam_analysis = cba.BeamAnalysis([L], EI, R, LM, eletype=[4])
    out = beam_analysis.analyze()
    assert out == 0

    Ma = beam_analysis.beam_results.results.M[1]
    Mb = beam_analysis.beam_results.results.M[-2]
    Mc = beam_analysis.beam_results.results.M[51]

    assert Ma == pytest.approx(0)
    assert Mb == pytest.approx(0)
    assert Mc == pytest.approx(P * L / 4)


def get_1span_beam_def(etype):
    P = 10  # kN
    L = 10  # m
    a = 0.25 * L
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, -1, -1, -1]  # Notice, fixed-fixed supports
    LM = [[1, 2, P, a, 0]]

    beam_analysis = cba.BeamAnalysis([L], EI, R, LM, eletype=[etype])
    beam_analysis.analyze()

    d = beam_analysis.beam_results.results.D
    dmax = min(d)

    return P, L, EI, a, dmax


def test_1span_def_ff():
    """
    Test fixed-fixed beam deflection for off-centre point load
    """

    P, L, EI, aa, dmax = get_1span_beam_def(etype=1)

    # a>b
    b = aa
    a = L - b
    ymax = -(2 * P * a**3 * b**2) / (3 * (3 * a + b) ** 2 * EI)

    assert dmax == pytest.approx(ymax, abs=1e-6)


def test_1span_def_fp():
    """
    Test fixed-pinned beam deflection for off-centre point load
    """
    P, L, EI, aa, dmax = get_1span_beam_def(etype=2)

    a = aa
    b = L - a
    ymax = -(P * a**2 * b) / (6 * EI) * (b / (3 * L - a)) ** 0.5

    assert dmax == pytest.approx(ymax, abs=1e-6)


def test_1span_def_pf():
    """
    Test pinned-fixed beam deflection for off-centre point load
    """
    P, L, EI, aa, dmax = get_1span_beam_def(etype=3)

    b = aa
    ymax = -(P * b) / (3 * EI) * (L**2 - b**2) ** 3 / (3 * L**2 - b**2) ** 2

    assert dmax == pytest.approx(ymax, abs=1e-6)


def test_1span_def_pp():
    """
    Test pinned-pinned beam deflection for off-centre point load
    """
    P, L, EI, aa, dmax = get_1span_beam_def(etype=4)

    a = aa
    ymax = -(3**0.5) * P * a * (L**2 - a**2) ** 1.5 / (27 * EI * L)

    assert dmax == pytest.approx(ymax, abs=1e-6)


def test_2span_udl():
    """
    Execute a two-span beam analysis and check the reaction results.
    Uses a direct definition of the LM
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


def test_2span_load_wrappers():
    """
    Execute a two-span beam analysis and check the reaction results.
    Uses the wrappers for defining loads.
    """

    L = [7.5, 7.0]
    EI = 30 * 600e7 * 1e-6  # kNm2
    R = [-1, 0, -1, 0, -1, 0]

    beam_analysis = cba.BeamAnalysis(L, EI, R)
    beam_analysis.add_pl(1, 40, 3.5)
    beam_analysis.add_udl(1, 10)
    beam_analysis.add_pudl(2, 20, 2.0, 3.0)
    beam_analysis.add_ml(2, 50, 3)

    out = beam_analysis.analyze()
    assert out == 0

    r = beam_analysis.beam_results.R
    assert r == pytest.approx([45.41648878, 121.10155896, 8.48195226])

    dmax = max(beam_analysis.beam_results.results.D)
    dmin = min(beam_analysis.beam_results.results.D)
    assert [dmax, dmin] == pytest.approx(
        [0.00011753212898873195, -0.0023044064201367506]
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
        [0.0002778734578837245, -0.004648274177216889], abs=1e-5
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


def test_flipped_hinge():
    # 3-span with etype 2
    L = [5, 5, 10]
    EI = 30 * 600e7 * np.ones(len(L)) * 1e-6
    eType = [2, 1, 1]
    R = [-1, -1, 0, 0, -1, 0, -1, -1]
    LM = [[3, 2, 20, 5, 0]]
    beam_analysis = cba.BeamAnalysis(L, EI, R, LM, eType)
    out = beam_analysis.analyze()
    d1 = beam_analysis.beam_results.results.D

    # Same beam flipped, etype 3
    L = [10, 5, 5]
    EI = 30 * 600e7 * np.ones(len(L)) * 1e-6
    eType = [1, 1, 3]
    R = [-1, -1, -1, 0, 0, 0, -1, -1]
    LM = [[1, 2, 20, 5, 0]]
    beam_analysis = cba.BeamAnalysis(L, EI, R, LM, eType)
    beam_analysis.analyze()
    d2 = beam_analysis.beam_results.results.D

    # Confirm flipped deflected shapes are close
    assert d1 == pytest.approx(d2[::-1], abs=1e-7)


def test_hinges():
    # Based on example in Logan's First Course in FE, Ex. 4.10
    a = 4
    b = 2
    P = 20
    L = [a, b]
    EI = 30 * 600e7 * 1e-6
    EIvec = EI * np.ones(len(L))
    eType = [1, 3]
    R = [-1, -1, 0, 0, -1, -1]
    LM = [[1, 2, P, a, 0]]
    beam_analysis = cba.BeamAnalysis(L, EIvec, R, LM, eType)
    beam_analysis.analyze()

    phi2_1 = beam_analysis.beam_results.vRes[0].R[-2]
    phi2_1_theory = -(a**2 * b**3 * P) / (2 * (b**3 + a**3) * EI)

    assert phi2_1_theory == pytest.approx(phi2_1)

    phi2_2 = beam_analysis.beam_results.vRes[1].R[1]
    phi2_2_theory = (a**3 * b**2 * P) / (2 * (b**3 + a**3) * EI)

    assert phi2_2_theory == pytest.approx(phi2_2)


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


def test_envelopes():
    L = [6, 4, 6]
    EI = 30 * 10e9 * 1e-6
    R = [-1, 0, -1, 0, -1, 0, -1, 0]
    beam_analysis = cba.BeamAnalysis(L, EI, R)

    LMg = [[1, 1, 25, 0, 0], [2, 1, 25, 0, 0], [3, 1, 25, 0, 0]]
    γg_max = 1.4
    γg_min = 1.0
    LMq = [[1, 1, 10, 0, 0], [2, 1, 10, 0, 0], [3, 1, 10, 0, 0]]
    γq_max = 1.6
    γq_min = 0

    lp = cba.LoadPattern(beam_analysis)
    lp.set_dead_loads(LMg, γg_max, γg_min)
    lp.set_live_loads(LMq, γq_max, γq_min)
    env = lp.analyze()

    m_locs = np.array([3, 6, 8, 10, 13])
    idx = [(np.abs(env.x - x)).argmin() for x in m_locs]
    assert np.allclose(
        env.Mmax[idx], np.array([163.79, 0, 11.75, 0, 163.79]), atol=1e-2
    )
    assert np.allclose(
        env.Mmin[idx], np.array([0, -163.38, -81.42, -163.38, 0]), atol=1e-2
    )

    n = beam_analysis.beam_results.npts
    nspans = beam_analysis.beam.no_spans
    Vmax = np.array(
        [np.max(env.Vmax[i * (n + 3) : (i + 1) * (n + 3)]) for i in range(nspans)]
    )
    assert np.allclose(Vmax, np.array([131.1, 123.94, 180.23]), atol=1e-2)
    Vmin = np.array(
        [np.min(env.Vmin[i * (n + 3) : (i + 1) * (n + 3)]) for i in range(nspans)]
    )
    assert np.allclose(Vmin, np.array([-180.23, -123.94, -131.10]), atol=1e-2)
