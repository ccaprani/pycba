"""
Tests for the non-prismatic (variable-EI) element and the imposed-curvature
(initial-strain) load.

Coverage
--------
* Prismatic-limit regression: for a single constant ``SectionEI`` segment the
  breakpoint-aware flexibility-integrated element must reproduce the closed-form
  prismatic stiffness ``k_FF`` / ``k_FP`` / ``k_PF`` / ``k_PP`` and the released
  end forces of every load type to machine precision (primary correctness gate).
* Segment builder: ``const`` / ``linear`` / ``pwl`` / ``poly`` segments,
  contiguity / coverage validation, breakpoints, and a discontinuous (step)
  rigidity.
* Variable-EI element validation against analytical / fine-mesh references:
  a linear taper (wedge), a straight-haunch + flat-soffit ``pwl`` member (exact
  at low order via breakpoint splitting), and a parabolic-soffit ``poly``
  member.
* Imposed curvature: simply-supported (zero reactions, midspan ``-κL²/8``),
  fixed-fixed (``M = -EIκ``), 2-span continuous (pier moment ``-1.5 EIκ``), and
  imposed curvature on a non-prismatic member.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
#  Helpers
# ---------------------------------------------------------------------------
def _fine_mesh_ktheta(ei_fn, L, breaks=None):
    """Independent reference ``K_theta`` by adaptive flexibility integration.

    Recovers the 2x2 end moment-rotation stiffness ``F^-1`` of the released
    (simply-supported) element by adaptive Gauss-Kronrod quadrature
    (``scipy.integrate.quad``) of the unit-moment flexibility integrals,
    completely independently of :class:`SectionEI`.  Integration is split at the
    supplied ``breaks`` (EI kinks/steps) so the reference is itself essentially
    exact -- the truth for the haunch / parabolic verification.
    """
    from scipy import integrate

    pts = sorted({0.0, L, *([] if breaks is None else list(breaks))})

    def quad_split(integrand):
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            total += integrate.quad(integrand, a, b, limit=200)[0]
        return total

    def mi(x):
        return 1.0 - x / L

    def mj(x):
        return -x / L

    F = np.array(
        [
            [
                quad_split(lambda x: mi(x) * mi(x) / ei_fn(x)),
                quad_split(lambda x: mi(x) * mj(x) / ei_fn(x)),
            ],
            [
                quad_split(lambda x: mi(x) * mj(x) / ei_fn(x)),
                quad_split(lambda x: mj(x) * mj(x) / ei_fn(x)),
            ],
        ]
    )
    return np.linalg.inv(F)


# ---------------------------------------------------------------------------
#  Prismatic-limit regression (primary correctness gate)
# ---------------------------------------------------------------------------
EI_REF = 30 * 600e7 * 1e-6  # 180 000 kNm^2
L_REF = 10.0


def _const_section(L=L_REF, ei=EI_REF):
    return SectionEI().add_segment("const", [0.0, L], ei)


@pytest.mark.parametrize("eType", [1, 2, 3, 4])
def test_prismatic_limit_stiffness(eType):
    """Constant EI ⇒ non-prismatic element == closed-form prismatic k (machine)."""
    beam = cba.Beam()
    sec = _const_section()
    k_np = beam.k_nonprismatic(sec, L_REF, eType)
    k_prism = {
        1: beam.k_FF,
        2: beam.k_FP,
        3: beam.k_PF,
        4: beam.k_PP,
    }[
        eType
    ](EI_REF, L_REF)
    # Machine-precision gate: a single const segment is integrated exactly.
    assert np.allclose(k_np, k_prism, rtol=1e-12, atol=1e-6)


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
    sec = _const_section()
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

    sec = [_const_section(7.5), _const_section(7.0)]
    ba_sec = cba.BeamAnalysis(L, sec, R, LM)
    ba_sec.analyze()

    # The integrands are exact, so a constant SectionEI reproduces the
    # closed-form scalar-EI result to machine precision.
    assert ba_sec.beam_results.R == pytest.approx(ba_scalar.beam_results.R, abs=1e-9)
    assert ba_sec.beam_results.results.D == pytest.approx(
        ba_scalar.beam_results.results.D, abs=1e-12
    )


# ---------------------------------------------------------------------------
#  Segment-builder behaviour
# ---------------------------------------------------------------------------
def test_builder_linear_segment():
    sec = SectionEI().add_segment("linear", [0.0, 10.0], [100.0, 300.0])
    assert sec(0.0) == pytest.approx(100.0)
    assert sec(10.0) == pytest.approx(300.0)
    assert sec(5.0) == pytest.approx(200.0)
    assert not sec.is_constant
    assert sec.length == pytest.approx(10.0)
    assert list(sec.breakpoints) == pytest.approx([0.0, 10.0])


def test_builder_oneliner_equivalent_to_chained():
    chained = (
        SectionEI()
        .add_segment("linear", [0.0, 3.0], [3.0e5, 1.2e5])
        .add_segment("const", [3.0, 9.0], 1.2e5)
        .add_segment("linear", [9.0, 12.0], [1.2e5, 3.0e5])
    )
    oneliner = SectionEI(
        [
            ("linear", [0.0, 3.0], [3.0e5, 1.2e5]),
            {"seg_type": "const", "x": [3.0, 9.0], "ei": 1.2e5},
            ("linear", [9.0, 12.0], [1.2e5, 3.0e5]),
        ]
    )
    xx = np.linspace(0.0, 12.0, 101)
    assert np.allclose(chained(xx), oneliner(xx))
    assert list(chained.breakpoints) == pytest.approx(list(oneliner.breakpoints))


def test_builder_pwl_multipoint_one_call():
    """A multi-point pwl in a single call has the interior kinks as breakpoints."""
    sec = SectionEI().add_segment(
        "pwl", [0.0, 3.0, 6.0, 9.0, 12.0], [3e5, 1.6e5, 1.2e5, 1.6e5, 3e5]
    )
    assert list(sec.breakpoints) == pytest.approx([0.0, 3.0, 6.0, 9.0, 12.0])
    assert sec(1.5) == pytest.approx(0.5 * (3e5 + 1.6e5))
    assert sec(6.0) == pytest.approx(1.2e5)


def test_builder_constant_flag():
    sec = (
        SectionEI()
        .add_segment("const", [0.0, 5.0], 200.0)
        .add_segment("const", [5.0, 10.0], 200.0)
    )
    assert sec.is_constant
    assert sec(3.3) == pytest.approx(200.0)


def test_builder_step_discontinuity():
    """A coincident x with differing EI is an allowed step (left-continuous)."""
    sec = (
        SectionEI()
        .add_segment("const", [0.0, 5.0], 1.0e5)
        .add_segment("const", [5.0, 10.0], 2.0e5)
    )
    assert sec(4.999) == pytest.approx(1.0e5)
    assert sec(5.0) == pytest.approx(1.0e5)  # left piece value at the step
    assert sec(5.001) == pytest.approx(2.0e5)
    assert list(sec.breakpoints) == pytest.approx([0.0, 5.0, 10.0])


def test_builder_poly_callable():
    """A poly segment from a callable evaluated in the span-local coordinate."""
    sec = SectionEI().add_segment(
        "poly", [0.0, 4.0], lambda x: 1.0e5 * (1.0 + 0.1 * x) ** 3, degree=6
    )
    assert sec(0.0) == pytest.approx(1.0e5, rel=1e-9)
    assert sec(4.0) == pytest.approx(1.0e5 * 1.4**3, rel=1e-9)
    assert sec(2.0) == pytest.approx(1.0e5 * 1.2**3, rel=1e-9)


def test_builder_validation():
    # unknown type
    with pytest.raises(ValueError):
        SectionEI().add_segment("cubic", [0.0, 1.0], [1.0, 2.0])
    # not increasing within a call
    with pytest.raises(ValueError):
        SectionEI().add_segment("linear", [1.0, 0.0], [1.0, 2.0])
    # length mismatch
    with pytest.raises(ValueError):
        SectionEI().add_segment("pwl", [0.0, 1.0, 2.0], [1.0, 2.0])
    # non-positive EI
    with pytest.raises(ValueError):
        SectionEI().add_segment("linear", [0.0, 1.0], [1.0, -2.0])
    # first segment must start at 0
    with pytest.raises(ValueError):
        SectionEI().add_segment("const", [1.0, 5.0], 1.0e5)
    # gap between segments
    with pytest.raises(ValueError):
        (
            SectionEI()
            .add_segment("const", [0.0, 5.0], 1.0e5)
            .add_segment("const", [6.0, 10.0], 1.0e5)
        )
    # overlap between segments
    with pytest.raises(ValueError):
        (
            SectionEI()
            .add_segment("const", [0.0, 5.0], 1.0e5)
            .add_segment("const", [4.0, 10.0], 1.0e5)
        )


def test_coverage_must_match_span_length():
    """Attaching a section that does not span the full length raises."""
    sec = SectionEI().add_segment("const", [0.0, 8.0], 1.0e5)
    with pytest.raises(ValueError):
        cba.Beam().add_span(10.0, sec, 1)
    # exact coverage is fine
    cba.Beam().add_span(8.0, sec, 1)


# ---------------------------------------------------------------------------
#  Variable-EI element validation
# ---------------------------------------------------------------------------
def test_variable_ei_cantilever_tip_deflection():
    """Cantilever with a linear-taper segment under a tip load vs fine reference."""
    L = 8.0
    EI0, EI1 = 100000.0, 300000.0
    P = 50.0
    sec = SectionEI().add_segment("linear", [0.0, L], [EI0, EI1])

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


def test_wedge_frame_constants_machine_precision():
    """Linear-taper (wedge) element factors == analytical flexibility reference."""
    EI0, r, L = 100000.0, 2.0, 10.0
    sec = SectionEI().add_segment("linear", [0.0, L], [EI0, r * EI0])

    # High-accuracy adaptive reference (independent of SectionEI).
    Kref = _fine_mesh_ktheta(lambda x: EI0 * (1.0 + (r - 1.0) * x / L), L)
    Kth = cba.Beam().k_theta(sec, L)
    assert np.allclose(Kth, Kref, rtol=1e-9)


def test_straight_haunch_flat_soffit_exact_low_order():
    """Straight haunch + flat soffit (pwl) is exact at low order vs fine mesh.

    A symmetric member: straight (linear) haunch over [0, a] and [L-a, L] with a
    flat (const) soffit between.  Because the breakpoint-aware integrator splits
    at the kinks, the piecewise-linear EI is captured *exactly* with only the
    natural low Gauss order per piece -- no global high-degree fit is needed.
    """
    L, a = 12.0, 3.0
    EI_end, EI_mid = 3.0e5, 1.2e5
    sec = SectionEI().add_segment(
        "pwl", [0.0, a, L - a, L], [EI_end, EI_mid, EI_mid, EI_end]
    )

    def ei_fn(x):
        if x < a:
            return EI_end + (EI_mid - EI_end) * (x / a)
        if x > L - a:
            return EI_mid + (EI_end - EI_mid) * ((x - (L - a)) / a)
        return EI_mid

    Kref = _fine_mesh_ktheta(ei_fn, L, breaks=[a, L - a])
    Kth = cba.Beam().k_theta(sec, L)
    # Exact: piecewise-linear EI integrated piece-by-piece between breakpoints.
    err = np.max(np.abs(Kth - Kref) / np.abs(Kref))
    assert err < 1e-9, f"haunch+flat rel err = {err:.2e}"


def test_parabolic_soffit_poly_segment():
    """Parabolic-soffit (EI ~ depth^3) single poly segment vs fine reference."""
    L = 10.0
    # Parabolic depth d(x) deepening to the right; EI = E b d(x)^3 / 12.
    E, b, d0, d1 = 30.0e6, 1.0, 0.6, 1.2

    def depth(x):
        return d0 + (d1 - d0) * (np.asarray(x, dtype=float) / L) ** 2

    def ei_fn(x):
        return E * b * depth(x) ** 3 / 12.0

    # Parabolic soffit -> EI ~ degree-6 in x: a callable poly segment captures it.
    sec = SectionEI().add_segment("poly", [0.0, L], ei_fn, degree=6)
    assert sec(0.0) == pytest.approx(ei_fn(0.0), rel=1e-9)
    assert sec(L) == pytest.approx(ei_fn(L), rel=1e-9)

    Kref = _fine_mesh_ktheta(ei_fn, L)
    Kth = cba.Beam().k_theta(sec, L)
    err = np.max(np.abs(Kth - Kref) / np.abs(Kref))
    assert err < 1e-6, f"parabolic-soffit rel err = {err:.2e}"


def test_variable_ei_propped_cantilever_reactions():
    """Propped cantilever, linear EI, UDL: single NP element vs discretisation."""
    L = 10.0
    EI0, EI1 = 100000.0, 250000.0
    w = 20.0
    sec = SectionEI().add_segment("linear", [0.0, L], [EI0, EI1])

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
    sec = SectionEI().add_segment("linear", [0.0, L], [EI0, EI1])

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
    sec = SectionEI().add_segment("linear", [0.0, L], [EI0, EI1])

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
    """Direct check of the closed-form prismatic fixed-end moment M = EIκ.

    ``LoadIC`` is no longer a public export; reach it via load-type code 6 (the
    parse_LM dispatch) -- the public path -- and via add_ic below.
    """
    L = 10.0
    EI = 180000.0
    kappa = 1e-4
    ld = LoadIC(0, kappa, EI=EI)
    cnl = ld.get_cnl(L, 1)
    assert cnl.Ma == pytest.approx(EI * kappa, rel=1e-6)
    assert cnl.Mb == pytest.approx(-EI * kappa, rel=1e-6)
    # No net transverse load ⇒ end shears form an equilibrating couple.
    assert cnl.Va == pytest.approx((cnl.Ma + cnl.Mb) / L)


def test_loadic_not_public_but_reachable_as_type_6():
    """LoadIC is internal (not exported) yet reachable via a type-6 LoadMatrix."""
    assert not hasattr(cba, "LoadIC")
    from pycba.load import parse_LM, LoadIC as _LoadIC

    loads = parse_LM([[1, 6, 1e-4, 2e-5]])
    assert len(loads) == 1 and isinstance(loads[0], _LoadIC)


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


# ---------------------------------------------------------------------------
#  SectionEI.plot input-verification figure
# ---------------------------------------------------------------------------
def test_sectionei_plot_haunch_flat(monkeypatch):
    """``SectionEI.plot`` returns a Figure/Axes spanning [0, L], marks the
    breakpoints, and never calls ``plt.show()``."""

    # plt.show() must not be called by a library plotting method.
    def _boom(*args, **kwargs):  # pragma: no cover - only fires on failure
        raise AssertionError("SectionEI.plot must not call plt.show()")

    monkeypatch.setattr(plt, "show", _boom)

    # Symmetric straight-haunch + flat-soffit member (kinks at the joins).
    L = 12.0
    sec = SectionEI(
        [
            ("linear", [0.0, 3.0], [3.0e5, 1.2e5]),
            ("const", [3.0, 9.0], 1.2e5),
            ("linear", [9.0, 12.0], [1.2e5, 3.0e5]),
        ]
    )

    fig, ax = sec.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)

    # The drawn EI curve spans the full local span [0, L].
    assert ax.get_xlim() == pytest.approx((0.0, L))
    data_lines = [ln for ln in ax.lines if ln.get_linestyle() == "-"]
    assert data_lines, "expected solid EI data lines"
    xmin = min(ln.get_xdata().min() for ln in data_lines)
    xmax = max(ln.get_xdata().max() for ln in data_lines)
    assert xmin == pytest.approx(0.0)
    assert xmax == pytest.approx(L)

    # The breakpoints are marked with vertical lines.
    vlines = [ln.get_xdata()[0] for ln in ax.lines if ln.get_linestyle() == ":"]
    for bp in sec.breakpoints:
        assert any(np.isclose(bp, vx) for vx in vlines), f"breakpoint {bp} not marked"

    plt.close(fig)


def test_sectionei_plot_prismatic_and_ax_reuse():
    """A single-const (prismatic) section plots without error; a step renders a
    solid vertical riser; and a supplied ``ax`` is reused."""
    sec = SectionEI().add_segment("const", [0.0, 10.0], 2.0e5)
    fig, ax = sec.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(fig)

    # A genuine step discontinuity draws exactly one solid vertical riser at the
    # step (part of the region's black outline), spanning the two EI values.
    step = (
        SectionEI()
        .add_segment("const", [0.0, 5.0], 1.0e5)
        .add_segment("const", [5.0, 10.0], 2.0e5)
    )
    fig2, ax2 = step.plot()
    risers = [
        ln
        for ln in ax2.lines
        if len(np.asarray(ln.get_xdata())) == 2
        and np.allclose(np.asarray(ln.get_xdata(), dtype=float), 5.0)
        and np.max(np.asarray(ln.get_ydata(), dtype=float)) > 10.0
    ]
    assert len(risers) == 1
    plt.close(fig2)

    # A supplied axes (and its figure) is drawn into, not replaced.
    my_fig, my_ax = plt.subplots()
    rfig, rax = sec.plot(ax=my_ax)
    assert rax is my_ax and rfig is my_fig
    plt.close(my_fig)
