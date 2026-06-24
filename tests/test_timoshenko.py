"""
Tests for the Timoshenko (shear-deformable) beam element.

A member becomes a Timoshenko element when given a finite transverse shear
rigidity ``GAv``; with ``GAv=None`` it stays on the exact Euler–Bernoulli (EB)
path.

Coverage
--------
* Element stiffness: the prismatic ``k_FF_timo`` matches the textbook
  shear-flexible matrix exactly; the flexibility-integrated ``k_timoshenko``
  reproduces it (and the released variants) to machine precision; every element
  type reduces to the EB closed forms as ``GAv -> inf``.
* Analytical deflections: simply-supported central point load
  ``PL^3/48EI + PL/4GAv`` and cantilever tip load ``PL^3/3EI + PL/GAv``.
* Phi-dependent fixed-end forces: a propped-cantilever UDL clamp moment of
  ``wL^2 / (2(4 + Phi))`` (EB limit ``wL^2/8``), confirming shear redistributes
  moments correctly through the continuous-beam solve.
* EB-equivalence regression: ``GAv=None`` is bit-for-bit identical to omitting
  ``GAv``; a huge finite ``GAv`` matches EB reactions/forces to tight tolerance.
* Variable ``GAv``: a constant ``SectionEI`` shear rigidity equals the scalar
  path; a variable ``EI``/``GAv`` reduces to the EB non-prismatic element as
  ``GAv -> inf``; the prismatic closed-form fixed-end forces match the general
  flexibility-integration path.
* Argument handling: ``GAv`` broadcasting and ``SectionEI`` length validation.
"""

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import pycba as cba
from pycba.section import SectionEI


EI = 1.0e5
L = 10.0


def _textbook_k_timo(EI, GAv, L):
    """The standard two-node shear-flexible element stiffness matrix."""
    Phi = 12.0 * EI / (GAv * L**2)
    c = EI / ((1.0 + Phi) * L**3)
    return c * np.array(
        [
            [12, 6 * L, -12, 6 * L],
            [6 * L, (4 + Phi) * L**2, -6 * L, (2 - Phi) * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, (2 - Phi) * L**2, -6 * L, (4 + Phi) * L**2],
        ]
    )


# ---------------------------------------------------------------------------
#  Element stiffness
# ---------------------------------------------------------------------------
def test_kff_timo_matches_textbook_matrix():
    b = cba.Beam([L], EI, [-1, 0, -1, 0], eletype=[1])
    GAv = 2.0e4
    np.testing.assert_allclose(b.k_FF_timo(EI, GAv, L), _textbook_k_timo(EI, GAv, L))


def test_closedform_matches_flexibility():
    """Prismatic closed-form element == flexibility-integrated element."""
    b = cba.Beam([L], EI, [-1, 0, -1, 0], eletype=[1])
    GAv = 3.0e4
    np.testing.assert_allclose(
        b.k_FF_timo(EI, GAv, L), b.k_timoshenko(EI, GAv, L, 1), atol=1e-9
    )


@pytest.mark.parametrize("eType", [1, 2, 3, 4])
def test_all_element_types_reduce_to_eb(eType):
    """As GAv -> inf every element type reduces to the EB closed form."""
    b_eb = cba.Beam([L], EI, [-1, 0, -1, 0], eletype=[eType])
    b_t = cba.Beam([L], EI, [-1, 0, -1, 0], eletype=[eType], GAv=1.0e16)
    np.testing.assert_allclose(b_t.get_span_k(0), b_eb.get_span_k(0), atol=1e-4)


# ---------------------------------------------------------------------------
#  Analytical deflections
# ---------------------------------------------------------------------------
def test_ss_central_point_load_deflection():
    """delta_mid = PL^3/48EI + PL/4GAv."""
    P, GAv = 50.0, 2.0e4
    ba = cba.BeamAnalysis([L], EI, [-1, 0, -1, 0], GAv=GAv)
    ba.add_pl(1, P, L / 2)
    ba.analyze(npts=2000)
    d = ba.beam_results.results.D.min()
    analytic = -(P * L**3 / (48 * EI) + P * L / (4 * GAv))
    assert d == pytest.approx(analytic, rel=1e-4)


def test_cantilever_tip_load_deflection():
    """delta_tip = PL^3/3EI + PL/GAv."""
    P, GAv = 50.0, 2.0e4
    ba = cba.BeamAnalysis([L], EI, [-1, -1, 0, 0], GAv=GAv)
    ba.add_pl(1, P, L)
    ba.analyze(npts=2000)
    d = ba.beam_results.results.D.min()
    analytic = -(P * L**3 / (3 * EI) + P * L / GAv)
    assert d == pytest.approx(analytic, rel=1e-4)


def test_shear_increases_deflection():
    """A finite GAv must deflect more than the EB limit."""
    P = 50.0
    ba_t = cba.BeamAnalysis([L], EI, [-1, 0, -1, 0], GAv=2.0e4)
    ba_t.add_pl(1, P, L / 2)
    ba_t.analyze(npts=500)
    ba_eb = cba.BeamAnalysis([L], EI, [-1, 0, -1, 0])
    ba_eb.add_pl(1, P, L / 2)
    ba_eb.analyze(npts=500)
    assert abs(ba_t.beam_results.results.D.min()) > abs(
        ba_eb.beam_results.results.D.min()
    )


# ---------------------------------------------------------------------------
#  Phi-dependent fixed-end forces through the continuous-beam solve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("GAv", [2.0e4, 5.0e4, 1.0e5])
def test_propped_cantilever_phi_dependent_clamp_moment(GAv):
    """Propped-cantilever UDL clamp moment = wL^2 / (2(4 + Phi))."""
    w = 12.0
    ba = cba.BeamAnalysis([L], EI, [-1, -1, -1, 0], GAv=GAv)
    ba.add_udl(1, w)
    ba.analyze(npts=4000)
    Phi = 12.0 * EI / (GAv * L**2)
    analytic = w * L**2 / (2.0 * (4.0 + Phi))  # magnitude
    clamp = abs(ba.beam_results.results.M).max()
    assert clamp == pytest.approx(analytic, rel=1e-3)


def test_symmetric_udl_fem_unchanged_prismatic():
    """A symmetric UDL on a prismatic FF member has Phi-independent FEMs."""
    w = 10.0
    m_eb = cba.BeamAnalysis([L], EI, [-1, -1, -1, -1])
    m_eb.add_udl(1, w)
    m_eb.analyze()
    m_t = cba.BeamAnalysis([L], EI, [-1, -1, -1, -1], GAv=2.0e4)
    m_t.add_udl(1, w)
    m_t.analyze()
    # Fixed-end moments (member end forces) are independent of shear here.
    assert m_t.beam_results.results.M.max() == pytest.approx(
        m_eb.beam_results.results.M.max(), rel=1e-9
    )


# ---------------------------------------------------------------------------
#  Euler–Bernoulli equivalence / regression
# ---------------------------------------------------------------------------
def test_gav_none_is_bit_identical():
    """GAv=None must give results identical to omitting GAv entirely."""
    w = 15.0
    a = cba.BeamAnalysis([L, L], EI, [-1, 0, -1, 0, -1, 0])
    a.add_udl(1, w)
    a.add_udl(2, w)
    a.analyze()
    b = cba.BeamAnalysis([L, L], EI, [-1, 0, -1, 0, -1, 0], GAv=None)
    b.add_udl(1, w)
    b.add_udl(2, w)
    b.analyze()
    np.testing.assert_array_equal(a.beam_results.results.D, b.beam_results.results.D)
    np.testing.assert_array_equal(a.beam_results.R, b.beam_results.R)


def test_multispan_reactions_reduce_to_eb():
    w = 12.0
    eb = cba.BeamAnalysis([L, L, L], EI, [-1, 0, -1, 0, -1, 0, -1, 0])
    for i in (1, 2, 3):
        eb.add_udl(i, w)
    eb.analyze()
    t = cba.BeamAnalysis([L, L, L], EI, [-1, 0, -1, 0, -1, 0, -1, 0], GAv=1.0e16)
    for i in (1, 2, 3):
        t.add_udl(i, w)
    t.analyze()
    np.testing.assert_allclose(t.beam_results.R, eb.beam_results.R, rtol=1e-6)


@pytest.mark.parametrize("eType", [1, 2, 3])
def test_release_reactions_reduce_to_eb(eType):
    """Released-member reactions (exact from the solve) reduce to EB."""
    w = 12.0
    eb = cba.BeamAnalysis([L], EI, [-1, -1, -1, -1], eletype=[eType])
    eb.add_udl(1, w)
    eb.analyze()
    t = cba.BeamAnalysis([L], EI, [-1, -1, -1, -1], eletype=[eType], GAv=1.0e16)
    t.add_udl(1, w)
    t.analyze()
    np.testing.assert_allclose(t.beam_results.R, eb.beam_results.R, rtol=1e-6)


# ---------------------------------------------------------------------------
#  Variable (non-prismatic) shear rigidity
# ---------------------------------------------------------------------------
def test_const_sectionei_gav_equals_scalar():
    GAv = 2.5e4
    b_scalar = cba.Beam([L], EI, [-1, 0, -1, 0], eletype=[1], GAv=GAv)
    b_sec = cba.Beam(
        [L], EI, [-1, 0, -1, 0], eletype=[1], GAv=SectionEI([("const", [0.0, L], GAv)])
    )
    np.testing.assert_allclose(b_sec.get_span_k(0), b_scalar.get_span_k(0), atol=1e-8)


def test_variable_ei_gav_reduces_to_eb_nonprismatic():
    w = 12.0
    sec_EI = SectionEI([("linear", [0.0, L], [2 * EI, EI])])
    eb = cba.BeamAnalysis([L], sec_EI, [-1, 0, -1, 0])
    eb.add_udl(1, w)
    eb.analyze(npts=1000)
    sec_GAv = SectionEI([("linear", [0.0, L], [2.0e16, 1.0e16])])
    t = cba.BeamAnalysis([L], sec_EI, [-1, 0, -1, 0], GAv=sec_GAv)
    t.add_udl(1, w)
    t.analyze(npts=1000)
    np.testing.assert_allclose(
        t.beam_results.results.D, eb.beam_results.results.D, atol=1e-9
    )


def test_prismatic_ref_matches_integration_path():
    """The prismatic closed-form fixed-end forces equal the general
    flexibility-integration path (forced via a constant SectionEI).

    A full-span trapezoidal load gives a smooth (kink-free) asymmetric moment
    diagram, so the two routes agree to integration precision.
    """
    from pycba.load import LoadTrapez

    b = cba.Beam([L], EI, [-1, 0, -1, 0], eletype=[1])
    GAv = 3.0e4
    load = LoadTrapez(0, 10.0, 30.0)  # asymmetric so Ma != Mb
    ref_closed = b._ref_timoshenko(load, EI, GAv, L, 1)
    ref_integ = b._ref_timoshenko(
        load,
        SectionEI([("const", [0.0, L], EI)]),
        SectionEI([("const", [0.0, L], GAv)]),
        L,
        1,
    )
    # The closed form is exact; the integration route is accurate to the
    # quadrature's endpoint handling (the SS shear term).  Agreement to ~1e-4
    # confirms the closed-form transform has no derivation error.
    np.testing.assert_allclose(ref_closed, ref_integ, rtol=1e-3)


# ---------------------------------------------------------------------------
#  Argument handling
# ---------------------------------------------------------------------------
def test_gav_broadcast_scalar_applies_to_all_spans():
    b = cba.Beam([L, L], EI, [-1, 0, -1, 0, -1, 0], eletype=[1, 1], GAv=2.0e4)
    assert b.mbr_GAv == [2.0e4, 2.0e4]


def test_gav_per_span_list():
    b = cba.Beam([L, L], EI, [-1, 0, -1, 0, -1, 0], eletype=[1, 1], GAv=[2.0e4, None])
    assert b.mbr_GAv == [2.0e4, None]
    # First span is Timoshenko, second is Euler–Bernoulli.
    assert not np.allclose(b.get_span_k(0), b.get_span_k(1))


def test_gav_length_mismatch_raises():
    with pytest.raises(ValueError):
        cba.Beam([L, L], EI, [-1, 0, -1, 0, -1, 0], eletype=[1, 1], GAv=[2.0e4])


def test_sectionei_gav_length_validation():
    with pytest.raises(ValueError):
        cba.Beam(
            [L],
            EI,
            [-1, 0, -1, 0],
            eletype=[1],
            GAv=SectionEI([("const", [0.0, L / 2], 2.0e4)]),
        )
