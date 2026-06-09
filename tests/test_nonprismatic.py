"""
Tests for the non-prismatic (variable-EI) element and the imposed-curvature
(initial-strain) load.

Coverage
--------
* Prismatic-limit regression: for a constant ``EI`` the flexibility-integrated
  non-prismatic element must reproduce the closed-form prismatic stiffness
  ``k_FF`` / ``k_FP`` / ``k_PF`` / ``k_PP`` and the released end forces of every
  load type to machine precision (primary correctness gate).
* Variable-EI: a cantilever and a propped cantilever with linearly-varying
  ``EI`` checked against analytical / fine-discretisation references.
* Imposed curvature: simply-supported (zero reactions, midspan ``κL²/8``),
  fixed-fixed (``M = EIκ``), 2-span continuous (pier moment ``1.5 EIκ``),
  quadrature/interpolation convergence, and imposed curvature on a
  non-prismatic member.
"""

import numpy as np
import pytest
import pycba as cba
from pycba.section import SectionEI
from pycba.load import (
    LoadUDL,
    LoadPL,
    LoadPUDL,
    LoadML,
    LoadTrapez,
    LoadIC,
)


# ---------------------------------------------------------------------------
#  Prismatic-limit regression (primary correctness gate)
# ---------------------------------------------------------------------------
EI_REF = 30 * 600e7 * 1e-6  # 180 000 kNm^2
L_REF = 10.0


@pytest.mark.parametrize("eType", [1, 2, 3, 4])
def test_prismatic_limit_stiffness(eType):
    """Constant EI ⇒ non-prismatic element == closed-form prismatic k."""
    beam = cba.Beam()
    sec = SectionEI([0.0, L_REF], [EI_REF, EI_REF])
    k_np = beam.k_nonprismatic(sec, L_REF, eType)
    k_prism = {
        1: beam.k_FF,
        2: beam.k_FP,
        3: beam.k_PF,
        4: beam.k_PP,
    }[eType](EI_REF, L_REF)
    assert np.allclose(k_np, k_prism)


@pytest.mark.parametrize("eType", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "load",
    [
        LoadUDL(0, 20.0),
        LoadPL(0, 50.0, 3.0),
        LoadPUDL(0, 15.0, 2.0, 5.0),
        LoadML(0, 40.0, 4.0),
        LoadTrapez(0, 10.0, 30.0, 1.0, 6.0),
    ],
)
def test_prismatic_limit_released_end_forces(load, eType):
    """Constant EI ⇒ non-prismatic released end forces == prismatic get_ref."""
    beam = cba.Beam()
    sec = SectionEI([0.0, L_REF], [EI_REF, EI_REF])
    ref_prism = np.array(load.get_ref(L_REF, eType))
    ref_np = beam._ref_nonprismatic(load, sec, L_REF, eType)
    assert np.allclose(ref_prism, ref_np, atol=1e-6)


def test_constant_sectionei_analysis_matches_scalar():
    """A full analysis with a constant SectionEI equals the scalar-EI run."""
    L = [7.5, 7.0]
    R = [-1, 0, -1, 0, -1, 0]
    LM = [[1, 1, 20, 0, 0], [2, 2, 40, 3.5]]

    ba_scalar = cba.BeamAnalysis(L, EI_REF, R, LM)
    ba_scalar.analyze()

    sec = SectionEI([0.0, 10.0], [EI_REF, EI_REF])
    ba_sec = cba.BeamAnalysis(L, [sec, sec], R, LM)
    ba_sec.analyze()

    # Simpson quadrature is exact for the polynomial integrands, so a constant
    # SectionEI reproduces the closed-form scalar-EI result to machine precision.
    assert ba_sec.beam_results.R == pytest.approx(ba_scalar.beam_results.R, abs=1e-9)
    assert ba_sec.beam_results.results.D == pytest.approx(
        ba_scalar.beam_results.results.D, abs=1e-12
    )


# ---------------------------------------------------------------------------
#  SectionEI behaviour
# ---------------------------------------------------------------------------
def test_sectionei_interpolation():
    sec = SectionEI([0.0, 10.0], [100.0, 300.0])  # linear
    assert sec(0.0) == pytest.approx(100.0)
    assert sec(10.0) == pytest.approx(300.0)
    assert sec(5.0) == pytest.approx(200.0)
    assert not sec.is_constant


def test_sectionei_constant_flag():
    sec = SectionEI([0.0, 5.0, 10.0], [200.0, 200.0, 200.0])
    assert sec.is_constant
    assert sec(3.3) == pytest.approx(200.0)


def test_sectionei_validation():
    with pytest.raises(ValueError):
        SectionEI([0.0, 1.0], [1.0])  # length mismatch
    with pytest.raises(ValueError):
        SectionEI([1.0, 0.0], [1.0, 2.0])  # not increasing
    with pytest.raises(ValueError):
        SectionEI([0.0, 1.0], [1.0, -2.0])  # non-positive EI


# ---------------------------------------------------------------------------
#  Variable-EI element validation
# ---------------------------------------------------------------------------
def test_variable_ei_cantilever_tip_deflection():
    """Cantilever with linear EI under a tip load vs fine numeric reference."""
    L = 8.0
    EI0, EI1 = 100000.0, 300000.0
    P = 50.0
    sec = SectionEI([0.0, L], [EI0, EI1])

    ba = cba.BeamAnalysis([L], sec, [-1, -1, 0, 0], [[1, 2, P, L, 0]])
    ba.analyze(npts=2000)
    tip = ba.beam_results.results.D[-2]

    # Fine reference: cantilever moment M(x) = -P(L-x), double-integrate M/EI(x)
    xx = np.linspace(0.0, L, 400001)
    EIx = EI0 + (EI1 - EI0) * xx / L
    curv = -P * (L - xx) / EIx
    rot = np.concatenate([[0.0], np.cumsum((curv[1:] + curv[:-1]) / 2 * np.diff(xx))])
    defl = np.concatenate([[0.0], np.cumsum((rot[1:] + rot[:-1]) / 2 * np.diff(xx))])

    assert tip == pytest.approx(defl[-1], rel=1e-4)
    # Reactions are statically determinate: V = P, M = P*L (pyCBA sign convention)
    assert ba.beam_results.R == pytest.approx([P, P * L], abs=1e-6)


def test_variable_ei_propped_cantilever_reactions():
    """Propped cantilever, linear EI, UDL: single NP element vs discretisation."""
    L = 10.0
    EI0, EI1 = 100000.0, 250000.0
    w = 20.0
    sec = SectionEI([0.0, L], [EI0, EI1])

    ba = cba.BeamAnalysis([L], sec, [-1, -1, -1, 0], [[1, 1, w]])
    ba.analyze(npts=1000)
    R_np = ba.beam_results.R

    # Reference: many constant-EI segments approximating EI(x)
    N = 400
    Ls = [L / N] * N
    xm = (np.arange(N) + 0.5) * L / N
    EIs = list(EI0 + (EI1 - EI0) * xm / L)
    R = [-1, -1] + [0, 0] * (N - 1) + [-1, 0]
    LM = [[k + 1, 1, w] for k in range(N)]
    ba_ref = cba.BeamAnalysis(Ls, EIs, R, LM, eletype=[1] * N)
    ba_ref.analyze()

    assert R_np == pytest.approx(ba_ref.beam_results.R, rel=1e-3)


def test_variable_ei_convergence():
    """Tip deflection converges with the post-processing point count."""
    L = 8.0
    EI0, EI1 = 100000.0, 300000.0
    P = 50.0
    sec = SectionEI([0.0, L], [EI0, EI1])

    xx = np.linspace(0.0, L, 800001)
    EIx = EI0 + (EI1 - EI0) * xx / L
    curv = -P * (L - xx) / EIx
    rot = np.concatenate([[0.0], np.cumsum((curv[1:] + curv[:-1]) / 2 * np.diff(xx))])
    defl = np.concatenate([[0.0], np.cumsum((rot[1:] + rot[:-1]) / 2 * np.diff(xx))])
    ref = defl[-1]

    errs = []
    for n in [100, 400, 1600]:
        ba = cba.BeamAnalysis([L], sec, [-1, -1, 0, 0], [[1, 2, P, L, 0]])
        ba.analyze(npts=n)
        errs.append(abs(ba.beam_results.results.D[-2] - ref))
    # Trapezoidal integration: error roughly quarters as npts quadruples.
    assert errs[1] < errs[0]
    assert errs[2] < errs[1]


# ---------------------------------------------------------------------------
#  Imposed-curvature load validation
# ---------------------------------------------------------------------------
def test_imposed_curvature_simply_supported():
    """Uniform κ on a simply-supported span: zero reactions/moment, δ=κL²/8."""
    L = 10.0
    EI = 180000.0
    kappa = 1e-4

    ba = cba.BeamAnalysis([L], EI, [-1, 0, -1, 0], [[1, 6, kappa]])
    ba.analyze(npts=500)

    assert ba.beam_results.R == pytest.approx([0.0, 0.0], abs=1e-9)
    assert np.max(np.abs(ba.beam_results.results.M)) == pytest.approx(0.0, abs=1e-9)

    x = ba.beam_results.results.x
    D = ba.beam_results.results.D
    dmid = D[np.argmin(np.abs(x - L / 2))]
    assert dmid == pytest.approx(-kappa * L**2 / 8, rel=1e-4)


def test_imposed_curvature_fixed_fixed():
    """Uniform κ on a fixed-fixed prismatic span: M = -EIκ (constant)."""
    L = 10.0
    EI = 180000.0
    kappa = 1e-4

    ba = cba.BeamAnalysis([L], EI, [-1, -1, -1, -1], [[1, 6, kappa]])
    ba.analyze(npts=200)
    M = ba.beam_results.results.M

    assert M[1] == pytest.approx(-EI * kappa, rel=1e-9)
    assert M[-2] == pytest.approx(-EI * kappa, rel=1e-9)
    assert M[100] == pytest.approx(-EI * kappa, rel=1e-9)


def test_imposed_curvature_two_span_pier_moment():
    """Uniform κ on a 2-equal-span continuous beam: pier moment = -1.5 EIκ."""
    L = 10.0
    EI = 180000.0
    kappa = 1e-4

    LM = [[1, 6, kappa], [2, 6, kappa]]
    ba = cba.BeamAnalysis([L, L], EI, [-1, 0, -1, 0, -1, 0], LM)
    ba.analyze(npts=200)

    n = ba.beam_results.npts
    M_pier = ba.beam_results.results.M[n + 1]  # interior support
    assert M_pier == pytest.approx(-1.5 * EI * kappa, rel=1e-3)
    # Self-equilibrating: reactions sum to zero.
    assert np.sum(ba.beam_results.R) == pytest.approx(0.0, abs=1e-6)


def test_imposed_curvature_linear_field():
    """Linear κ(x) = k0 + k1·x on a simply-supported span: no internal force."""
    L = 10.0
    EI = 180000.0
    k0, k1 = 1e-4, 2e-5

    ba = cba.BeamAnalysis([L], EI, [-1, 0, -1, 0], [[1, 6, k0, k1]])
    ba.analyze(npts=1000)

    assert ba.beam_results.R == pytest.approx([0.0, 0.0], abs=1e-9)
    assert np.max(np.abs(ba.beam_results.results.M)) == pytest.approx(0.0, abs=1e-9)

    # Compare midspan deflection to a fine double-integration reference.
    xx = np.linspace(0.0, L, 200001)
    k = k0 + k1 * xx
    rot = np.concatenate([[0.0], np.cumsum((k[1:] + k[:-1]) / 2 * np.diff(xx))])
    defl = np.concatenate([[0.0], np.cumsum((rot[1:] + rot[:-1]) / 2 * np.diff(xx))])
    defl = defl - defl[-1] / L * xx  # enforce D(L) = 0
    ref_mid = defl[np.argmin(np.abs(xx - L / 2))]

    x = ba.beam_results.results.x
    D = ba.beam_results.results.D
    dmid = D[np.argmin(np.abs(x - L / 2))]
    assert dmid == pytest.approx(ref_mid, rel=1e-4)


def test_imposed_curvature_nonprismatic_member():
    """Uniform κ on a non-prismatic fixed-fixed span vs fine discretisation."""
    L = 10.0
    EI0, EI1 = 100000.0, 300000.0
    kappa = 1e-4
    sec = SectionEI([0.0, L], [EI0, EI1])

    ba = cba.BeamAnalysis([L], sec, [-1, -1, -1, -1], [[1, 6, kappa]])
    ba.analyze(npts=1000)
    M = ba.beam_results.results.M

    N = 200
    Ls = [L / N] * N
    xm = (np.arange(N) + 0.5) * L / N
    EIs = list(EI0 + (EI1 - EI0) * xm / L)
    R = [-1, -1] + [0, 0] * (N - 1) + [-1, -1]
    LM = [[k + 1, 6, kappa] for k in range(N)]
    ba_ref = cba.BeamAnalysis(Ls, EIs, R, LM, eletype=[1] * N)
    ba_ref.analyze()
    Mref = ba_ref.beam_results.results.M

    assert M[1] == pytest.approx(Mref[1], rel=1e-3)
    assert M[-2] == pytest.approx(Mref[-2], rel=1e-3)


def test_loadic_fixed_end_moment_value():
    """Direct check of the closed-form prismatic fixed-end moment M = EIκ."""
    L = 10.0
    EI = 180000.0
    kappa = 1e-4
    ld = LoadIC(0, kappa, EI=EI)
    cnl = ld.get_cnl(L, 1)
    assert cnl.Ma == pytest.approx(EI * kappa, rel=1e-6)
    assert cnl.Mb == pytest.approx(-EI * kappa, rel=1e-6)
    # No net transverse load ⇒ end shears form an equilibrating couple.
    assert cnl.Va == pytest.approx((cnl.Ma + cnl.Mb) / L)


def test_add_ic_wrapper():
    """The add_ic convenience wrapper matches the explicit load-matrix form."""
    L = 10.0
    EI = 180000.0
    kappa = 1e-4

    ba1 = cba.BeamAnalysis([L], EI, [-1, -1, -1, -1], [[1, 6, kappa]])
    ba1.analyze()

    ba2 = cba.BeamAnalysis([L], EI, [-1, -1, -1, -1])
    ba2.add_ic(1, kappa)
    ba2.analyze()

    assert ba2.beam_results.results.M == pytest.approx(ba1.beam_results.results.M)
