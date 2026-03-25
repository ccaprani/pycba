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


def test_load_superposition_rotations():
    """
    Test that MemberResults superposition correctly adds rotations (R).
    This is a regression test for a bug where R was incorrectly added as V.
    """
    from pycba.load import LoadUDL, LoadPL

    x = np.linspace(0, 10, 100)
    load1 = LoadUDL(0, 10)
    load2 = LoadPL(0, 20, 5)

    res1 = load1.get_mbr_results(x, 10)
    res2 = load2.get_mbr_results(x, 10)

    combined = res1 + res2

    mid = 50
    expected_R = res1.R[mid] + res2.R[mid]
    assert combined.R[mid] == pytest.approx(expected_R)


def test_prescribed_settlement_no_load():
    """
    Test fixed-pinned beam with prescribed settlement at pin, no external load.

    Analytical solution for imposed displacement delta at pin B (DOF 2):
      theta_B  = 3*delta / (2*L)
      R_vA     = -3*EI*delta / L^3
      R_thetaA = -3*EI*delta / L^2
      R_vB     =  3*EI*delta / L^3
    """
    L = 10.0
    EI = 30 * 600e7 * 1e-6  # 180 000 kNm^2
    delta = -0.01  # 10 mm downward settlement
    R = [-1, -1, -1, 0]
    D = [None, None, delta, None]

    ba = cba.BeamAnalysis([L], EI, R, D=D)
    out = ba.analyze()
    assert out == 0

    r = ba.beam_results.R
    assert r == pytest.approx(
        [
            -3 * EI * delta / L**3,   # vertical reaction at A
            -3 * EI * delta / L**2,   # moment reaction at A
             3 * EI * delta / L**3,   # vertical reaction at B
        ],
        abs=1e-6,
    )

    # Deformed shape: only the settlement node should be non-zero
    d_nodes = ba.beam_results.D[[0, 2]]  # v_A, v_B
    assert d_nodes == pytest.approx([0.0, delta], abs=1e-9)


def test_prescribed_settlement_superposition():
    """
    For a linear system, reactions with both UDL and prescribed settlement must
    equal the sum of the load-only and settlement-only reactions.
    """
    L = 10.0
    EI = 30 * 600e7 * 1e-6
    w = 20.0
    delta = -0.01
    R = [-1, -1, -1, 0]
    LM = [[1, 1, w, 0, 0]]
    D = [None, None, delta, None]

    # Combined
    ba_both = cba.BeamAnalysis([L], EI, R, LM, D=D)
    ba_both.analyze()

    # Load only
    ba_load = cba.BeamAnalysis([L], EI, R, LM)
    ba_load.analyze()

    # Settlement only
    ba_sett = cba.BeamAnalysis([L], EI, R, D=D)
    ba_sett.analyze()

    assert ba_both.beam_results.R == pytest.approx(
        ba_load.beam_results.R + ba_sett.beam_results.R, abs=1e-6
    )


def test_spring_forces_reported():
    """
    Test that BeamResults.Rs equals -k_s * u_i (upward positive) for every spring-supported DOF.
    """
    L = [6, 8, 6]
    EI = 30 * np.array([50e8, 50e8, 50e8]) * 1e-6
    R = [-1, 486e9, -1, 486e9, -1, 486e9, -1, 486e9]
    LM = [[1, 1, 10, 0, 0], [2, 1, 20, 0, 0], [3, 1, 10, 0, 0]]

    ba = cba.BeamAnalysis(L, EI, R, LM)
    ba.analyze()

    r_vec = ba.beam.restraints
    d_vec = ba.beam_results.D
    rs = ba.beam_results.Rs

    spring_dofs = [i for i, rv in enumerate(r_vec) if rv > 0]
    expected_rs = np.array([-r_vec[i] * d_vec[i] for i in spring_dofs])

    assert len(rs) == len(spring_dofs)
    assert rs == pytest.approx(expected_rs, abs=1e-6)


def test_spring_prescribed_displacement_error():
    """
    A ValueError must be raised when a spring DOF simultaneously has a
    prescribed displacement and a non-zero consistent nodal load, because the
    elimination method would silently discard the load — an invalid model.
    """
    L = [5.0, 5.0]
    EI = 30 * 600e7 * 1e-6
    ks = 1e5
    w = 10.0
    R = [-1, 0, ks, 0, -1, 0]
    LM = [[1, 1, w, 0, 0]]           # UDL on span 1 → non-zero CNL at DOF 2
    D = [None, None, -0.005, None, None, None]

    ba = cba.BeamAnalysis(L, EI, R, LM, D=D)
    with pytest.raises(ValueError, match="Invalid model at DOF"):
        ba.analyze()


def test_trapezoidal_reduces_to_udl():
    """
    Trapezoidal load with w1 == w2 must give identical results to a UDL.
    """
    L = [10]
    EI = 30 * 600e7 * 1e-6
    R = [-1, 0, -1, 0]
    w = 5

    ba_udl = cba.BeamAnalysis(L, EI, R, [[1, 1, w, 0, 0]])
    ba_udl.analyze()

    ba_trap = cba.BeamAnalysis(L, EI, R, [[1, 5, w, w]])
    ba_trap.analyze()

    assert ba_udl.beam_results.results.M == pytest.approx(
        ba_trap.beam_results.results.M, abs=1e-10
    )
    assert ba_udl.beam_results.results.V == pytest.approx(
        ba_trap.beam_results.results.V, abs=1e-10
    )
    assert ba_udl.beam_results.R == pytest.approx(ba_trap.beam_results.R, abs=1e-10)


def test_trapezoidal_equilibrium():
    """
    Reactions for a trapezoidal load must sum to the total applied load.
    """
    L = [8]
    EI = 30 * 600e7 * 1e-6
    R = [-1, 0, -1, 0]
    w1, w2 = 3, 12

    ba = cba.BeamAnalysis(L, EI, R, [[1, 5, w1, w2]])
    ba.analyze()

    total_load = (w1 + w2) * L[0] / 2
    assert sum(ba.beam_results.R) == pytest.approx(total_load, abs=1e-6)


def test_trapezoidal_fixed_fixed():
    """
    Trapezoidal load on a fixed-fixed beam: verify end moments against
    known formulas. For w(x) = w1 + (w2-w1)*x/L on a fixed-fixed beam:
      Ma = w1*L^2/12 + (w2-w1)*L^2/30
      Mb = -(w1*L^2/12 + (w2-w1)*L^2/20)
    """
    L_val = 10
    EI = 30 * 600e7 * 1e-6
    R = [-1, -1, -1, -1]
    w1, w2 = 2, 8

    ba = cba.BeamAnalysis([L_val], EI, R, [[1, 5, w1, w2]])
    ba.analyze()

    dw = w2 - w1
    Ma_exact = w1 * L_val**2 / 12 + dw * L_val**2 / 30
    Mb_exact = -(w1 * L_val**2 / 12 + dw * L_val**2 / 20)

    Ma = ba.beam_results.results.M[1]
    Mb = ba.beam_results.results.M[-2]

    # In the M diagram: M[1] = -Ma_cnl (hogging), M[-2] = Mb_cnl
    assert Ma == pytest.approx(-Ma_exact, abs=1e-6)
    assert Mb == pytest.approx(Mb_exact, abs=1e-6)


def test_trapezoidal_add_trap():
    """
    The add_trap convenience method must produce the same result as a load matrix.
    """
    L = [10]
    EI = 30 * 600e7 * 1e-6
    R = [-1, 0, -1, 0]
    w1, w2 = 5, 15

    ba1 = cba.BeamAnalysis(L, EI, R, [[1, 5, w1, w2]])
    ba1.analyze()

    ba2 = cba.BeamAnalysis(L, EI, R)
    ba2.add_trap(1, w1, w2)
    ba2.analyze()

    assert ba1.beam_results.results.M == pytest.approx(
        ba2.beam_results.results.M, abs=1e-10
    )


def test_trapezoidal_triangular():
    """
    Pure triangular load (w1=0, w2=w) on a SS beam: verify reactions
    and max moment against textbook formulas.
    Va = wL/6, Vb = wL/3, Mmax = wL^2*sqrt(3)/27 at x = L/sqrt(3).
    """
    L_val = 12
    EI = 30 * 600e7 * 1e-6
    R = [-1, 0, -1, 0]
    w = 10

    ba = cba.BeamAnalysis([L_val], EI, R, [[1, 5, 0, w]])
    ba.analyze()

    r = ba.beam_results.R
    assert r[0] == pytest.approx(w * L_val / 6, abs=1e-6)
    assert r[1] == pytest.approx(w * L_val / 3, abs=1e-6)

    # Max moment at x = L/sqrt(3)
    Mmax_exact = w * L_val**2 * 3**0.5 / 27
    M = ba.beam_results.results.M
    assert max(M) == pytest.approx(Mmax_exact, rel=1e-3)


def test_trapezoidal_partial_reduces_to_pudl():
    """
    Partial trapezoidal with w1 == w2 must match a partial UDL.
    """
    L = [10]
    EI = 30 * 600e7 * 1e-6
    R = [-1, 0, -1, 0]
    w = 8
    a, c = 2, 5

    ba_pudl = cba.BeamAnalysis(L, EI, R, [[1, 3, w, a, c]])
    ba_pudl.analyze()

    ba_trap = cba.BeamAnalysis(L, EI, R, [[1, 5, w, w, a, c]])
    ba_trap.analyze()

    assert ba_pudl.beam_results.results.M == pytest.approx(
        ba_trap.beam_results.results.M, abs=1e-10
    )
    assert ba_pudl.beam_results.results.V == pytest.approx(
        ba_trap.beam_results.results.V, abs=1e-10
    )
    assert ba_pudl.beam_results.R == pytest.approx(ba_trap.beam_results.R, abs=1e-10)


def test_trapezoidal_partial_equilibrium():
    """
    Reactions for a partial trapezoidal load must sum to the total load.
    """
    L = [10]
    EI = 30 * 600e7 * 1e-6
    R = [-1, 0, -1, 0]
    w1, w2 = 3, 12
    a, c = 2, 5

    ba = cba.BeamAnalysis(L, EI, R, [[1, 5, w1, w2, a, c]])
    ba.analyze()

    total_load = (w1 + w2) * c / 2
    assert sum(ba.beam_results.R) == pytest.approx(total_load, abs=1e-6)


def test_trapezoidal_partial_full_span_equiv():
    """
    Explicit [span, 5, w1, w2, 0, L] must equal [span, 5, w1, w2].
    """
    L_val = 8
    EI = 30 * 600e7 * 1e-6
    R = [-1, -1, -1, -1]
    w1, w2 = 4, 14

    ba1 = cba.BeamAnalysis([L_val], EI, R, [[1, 5, w1, w2]])
    ba1.analyze()

    ba2 = cba.BeamAnalysis([L_val], EI, R, [[1, 5, w1, w2, 0, L_val]])
    ba2.analyze()

    assert ba1.beam_results.results.M == pytest.approx(
        ba2.beam_results.results.M, abs=1e-10
    )
    assert ba1.beam_results.results.V == pytest.approx(
        ba2.beam_results.results.V, abs=1e-10
    )
    assert ba1.beam_results.R == pytest.approx(ba2.beam_results.R, abs=1e-10)


def test_trapezoidal_multispan():
    """
    Trapezoidal loads on a 2-span beam: check equilibrium and deflections
    close at supports.
    """
    L = [8, 10]
    EI = 30 * 600e7 * 1e-6
    R = [-1, 0, -1, 0, -1, 0]
    w1a, w2a = 5, 15
    w1b, w2b = 10, 3

    ba = cba.BeamAnalysis(L, EI, R, [[1, 5, w1a, w2a], [2, 5, w1b, w2b]])
    ba.analyze()

    total = (w1a + w2a) * L[0] / 2 + (w1b + w2b) * L[1] / 2
    assert sum(ba.beam_results.R) == pytest.approx(total, abs=1e-6)

    # Deflections at supports must be zero
    d = ba.beam_results.D
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    assert d[2] == pytest.approx(0.0, abs=1e-9)
    assert d[4] == pytest.approx(0.0, abs=1e-9)


def test_trapezoidal_reversed_symmetry():
    """
    On a SS beam, trapez(w1→w2) reversed is equivalent to trapez(w2→w1).
    Reactions should swap; moments should be mirror images.
    """
    L_val = 10
    EI = 30 * 600e7 * 1e-6
    R = [-1, 0, -1, 0]
    w1, w2 = 4, 12

    ba1 = cba.BeamAnalysis([L_val], EI, R, [[1, 5, w1, w2]])
    ba1.analyze()

    ba2 = cba.BeamAnalysis([L_val], EI, R, [[1, 5, w2, w1]])
    ba2.analyze()

    # Reactions should be swapped
    R1 = ba1.beam_results.R
    R2 = ba2.beam_results.R
    assert R1[0] == pytest.approx(R2[1], abs=1e-10)
    assert R1[1] == pytest.approx(R2[0], abs=1e-10)

    # Moment diagrams should be mirror images (skip padded boundary indices)
    M1 = ba1.beam_results.results.M[2:-2]
    M2 = ba2.beam_results.results.M[2:-2]
    assert M1 == pytest.approx(M2[::-1], abs=1e-6)


def test_trapezoidal_partial_add_trap():
    """
    add_trap with a, c parameters must match load matrix with 6 elements.
    """
    L = [10]
    EI = 30 * 600e7 * 1e-6
    R = [-1, 0, -1, 0]
    w1, w2, a, c = 5, 15, 2, 6

    ba1 = cba.BeamAnalysis(L, EI, R, [[1, 5, w1, w2, a, c]])
    ba1.analyze()

    ba2 = cba.BeamAnalysis(L, EI, R)
    ba2.add_trap(1, w1, w2, a=a, c=c)
    ba2.analyze()

    assert ba1.beam_results.results.M == pytest.approx(
        ba2.beam_results.results.M, abs=1e-10
    )


def test_trapezoidal_partial_fixed_fixed():
    """
    Partial trapezoidal on a fixed-fixed beam: verify against numerical
    integration using many small point loads.
    """
    L_val = 10
    EI = 30 * 600e7 * 1e-6
    R = [-1, -1, -1, -1]
    w1, w2 = 5, 20
    a, c = 3, 4  # load from x=3 to x=7

    # Analytical trapezoidal
    ba_trap = cba.BeamAnalysis([L_val], EI, R, [[1, 5, w1, w2, a, c]])
    ba_trap.analyze()

    # Numerical: many point loads
    n_pts = 200
    dx = c / n_pts
    LM_pts = []
    for i in range(n_pts):
        xi = a + (i + 0.5) * dx
        wi = w1 + (w2 - w1) * (i + 0.5) * dx / c
        LM_pts.append([1, 2, wi * dx, xi, 0])

    ba_num = cba.BeamAnalysis([L_val], EI, R, LM_pts)
    ba_num.analyze()

    assert ba_trap.beam_results.R == pytest.approx(ba_num.beam_results.R, rel=1e-3)
    assert ba_trap.beam_results.results.M == pytest.approx(
        ba_num.beam_results.results.M, rel=1e-3, abs=1e-3
    )


def test_trapezoidal_factor_LM():
    """
    factor_LM must correctly scale w1 and w2 but preserve a and c.
    """
    from pycba.load import factor_LM

    LM = [[1, 5, 4, 10, 2, 5]]
    gamma = 1.5
    LM_f = factor_LM(LM, gamma)

    assert LM_f[0][2] == pytest.approx(6.0)   # w1 * 1.5
    assert LM_f[0][3] == pytest.approx(15.0)  # w2 * 1.5
    assert LM_f[0][4] == pytest.approx(2.0)   # a unchanged
    assert LM_f[0][5] == pytest.approx(5.0)   # c unchanged


def test_unstable_structure_error():
    """
    A geometrically unstable beam (no supports) must raise a ValueError
    with a clear stability message rather than a raw NumPy LinAlgError.
    """
    L = [5.0]
    EI = 30 * 600e7 * 1e-6
    R = [0, 0, 0, 0]               # no supports → singular stiffness matrix
    LM = [[1, 1, 10, 0, 0]]

    ba = cba.BeamAnalysis(L, EI, R, LM)
    with pytest.raises(ValueError, match="geometrically unstable"):
        ba.analyze()

