"""
Tests for the additive load-pattern ergonomics API.

These exercise the new convenience surface (LoadCombination basis binding,
Envelopes.combine / from_beam_analysis / operators / at / per_span,
LoadCases.envelope, and the add_all_spans_udl helpers) and assert that each
new one-call "after" path produces the same numbers as the verbose "before"
path it replaces.
"""

import numpy as np
import pytest

import pycba as cba
from pycba.results import Envelopes


def two_span_beam():
    return cba.BeamAnalysis(L=[10.0, 10.0], EI=30e6, R=[-1, 0, -1, 0, -1, 0])


def basis_load_cases(beam):
    load_cases = cba.LoadCases(beam)
    load_cases.add("G", [[1, 1, 5.0], [2, 1, 5.0]])
    load_cases.add("Q1", [[1, 1, 10.0]])
    load_cases.add("Q2", [[2, 1, 10.0]])
    return load_cases


# ---------------------------------------------------------------------------
# LoadCombination: optional basis (bound by target_combination)
# ---------------------------------------------------------------------------
def test_target_combination_binds_basis_for_no_arg_calls():
    beam = cba.BeamAnalysis(L=[8.0, 4.0], EI=30e6, R=[-1, 0, -1, 0, -1, 0])
    udl_basis = cba.make_patterned_udl(beam, w=10.0, n_segments=8)
    hogging = udl_basis.target_combination(
        "Hogging at first internal support",
        x=8.0,
        sense="min",
        response="M",
    )

    # No-arg calls equal the explicit-arg calls for all five methods.
    assert np.allclose(hogging.response()[1], hogging.response(udl_basis)[1])
    assert np.allclose(hogging.factor_vector(), hogging.factor_vector(udl_basis))
    assert hogging.to_LM() == hogging.to_LM(udl_basis)
    assert hogging.to_load_case().loads == hogging.to_load_case(udl_basis).loads
    assert hogging.analyze().beam_results is not None


def test_unbound_combination_raises_clear_error():
    beam = two_span_beam()
    load_cases = basis_load_cases(beam)
    combo = cba.LoadCombination("1.2G+1.5Q", {"G": 1.2, "Q1": 1.5})

    with pytest.raises(ValueError, match="No LoadCases bound"):
        combo.response()
    with pytest.raises(ValueError, match="No LoadCases bound"):
        combo.to_LM()

    # Explicit-arg form still works unchanged.
    x, y = combo.response(load_cases, response="M")
    assert len(x) == y.shape[0]


def test_analyse_alias_and_dataclass_integrity():
    assert cba.LoadCombination.analyse is cba.LoadCombination.analyze

    # _bound must not have leaked into the dataclass identity.
    a = cba.LoadCombination("a", {"G": 1.2})
    b = cba.LoadCombination("a", {"G": 1.2})
    assert a == b
    assert repr(a) == repr(b)
    assert "_bound" not in repr(a)

    # Positional construction still works (as exercised by existing tests).
    positional = cba.LoadCombination("ULS left", {"G": 1.2, "Q1": 1.5})
    assert positional.name == "ULS left"


def test_combination_envelope_matches_manual_envelopes():
    beam = cba.BeamAnalysis(L=[8.0, 4.0], EI=30e6, R=[-1, 0, -1, 0, -1, 0])
    udl_basis = cba.make_patterned_udl(beam, w=10.0, n_segments=8)
    sagging = udl_basis.target_combination(
        "Sagging span 1", x=4.0, sense="max", response="M"
    )

    env = sagging.envelope()
    manual = Envelopes([sagging.analyze(udl_basis).beam_results])

    assert isinstance(env, Envelopes)
    assert np.allclose(env.Mmax, manual.Mmax)
    assert np.allclose(env.Mmin, manual.Mmin)


# ---------------------------------------------------------------------------
# Envelopes.combine / from_beam_analysis / operators
# ---------------------------------------------------------------------------
def make_two_envelopes():
    beam = two_span_beam()
    ba_a = cba.BeamAnalysis(L=[10.0, 10.0], EI=30e6, R=[-1, 0, -1, 0, -1, 0])
    ba_a.add_udl(1, 10.0)
    ba_a.analyze()
    ba_b = cba.BeamAnalysis(L=[10.0, 10.0], EI=30e6, R=[-1, 0, -1, 0, -1, 0])
    ba_b.add_udl(2, 8.0)
    ba_b.analyze()
    return (
        Envelopes([ba_a.beam_results]),
        Envelopes([ba_b.beam_results]),
    )


def test_combine_envelope_mode_matches_zero_like_augment():
    a, b = make_two_envelopes()
    a_before = a.Mmax.copy()
    b_before = b.Mmax.copy()

    manual = Envelopes.zero_like(a)
    manual.augment(a)
    manual.augment(b)

    out = Envelopes.combine([a, b], mode="envelope")
    assert np.allclose(out.Mmax, manual.Mmax)
    assert np.allclose(out.Mmin, manual.Mmin)
    assert np.allclose(out.Vmax, manual.Vmax)

    # Inputs are not mutated.
    assert np.allclose(a.Mmax, a_before)
    assert np.allclose(b.Mmax, b_before)


def test_combine_sum_mode_matches_zero_like_sum():
    a, b = make_two_envelopes()
    manual = Envelopes.zero_like(a)
    manual.sum(a)
    manual.sum(b)

    out = Envelopes.combine([a, b], mode="sum")
    assert np.allclose(out.Mmax, manual.Mmax)
    assert np.allclose(out.Mmin, manual.Mmin)


def test_combine_errors():
    a, _ = make_two_envelopes()
    with pytest.raises(ValueError, match="at least one envelope"):
        Envelopes.combine([])
    with pytest.raises(ValueError, match="mode must be"):
        Envelopes.combine([a], mode="bogus")


def test_operators_equal_combine_and_are_non_mutating():
    a, b = make_two_envelopes()
    a_before = a.Mmax.copy()
    b_before = b.Mmax.copy()

    enclosing = a | b
    superposed = a + b

    assert np.allclose(enclosing.Mmax, Envelopes.combine([a, b], "envelope").Mmax)
    assert np.allclose(superposed.Mmax, Envelopes.combine([a, b], "sum").Mmax)

    # Operators do not mutate either operand.
    assert np.allclose(a.Mmax, a_before)
    assert np.allclose(b.Mmax, b_before)


def test_from_beam_analysis():
    beam = two_span_beam()
    ba = cba.BeamAnalysis(L=[10.0, 10.0], EI=30e6, R=[-1, 0, -1, 0, -1, 0])
    ba.add_udl(1, 10.0)
    ba.analyze()

    env = Envelopes.from_beam_analysis(ba)
    manual = Envelopes([ba.beam_results])
    assert np.allclose(env.Mmax, manual.Mmax)

    unanalysed = cba.BeamAnalysis(L=[10.0], EI=30e6, R=[-1, 0, -1, 0])
    with pytest.raises(ValueError, match="no results"):
        Envelopes.from_beam_analysis(unanalysed)


# ---------------------------------------------------------------------------
# Envelopes.at / per_span
# ---------------------------------------------------------------------------
def test_per_span_matches_manual_n_plus_3_slicing():
    beam = two_span_beam()
    env = cba.make_span_udl_cases(beam, w=10.0).envelope()

    n = env.vResults[0].npts
    nspans = beam.beam.no_spans
    manual_vmax = np.array(
        [np.max(env.Vmax[i * (n + 3) : (i + 1) * (n + 3)]) for i in range(nspans)]
    )
    manual_vmin = np.array(
        [np.min(env.Vmin[i * (n + 3) : (i + 1) * (n + 3)]) for i in range(nspans)]
    )

    assert np.allclose(env.per_span("Vmax"), manual_vmax)
    assert np.allclose(env.per_span("Vmin"), manual_vmin)

    # reduce="auto" picks min for *min attrs and max for *max attrs.
    assert np.allclose(env.per_span("Mmax"), env.per_span("Mmax", reduce="max"))
    assert np.allclose(env.per_span("Mmin"), env.per_span("Mmin", reduce="min"))

    chunks = env.per_span("Mmax", reduce="none")
    assert len(chunks) == nspans
    assert sum(len(c) for c in chunks) == len(env.Mmax)

    with pytest.raises(ValueError, match="reduce must be"):
        env.per_span("Mmax", reduce="bogus")


def test_at_matches_interpolated_envelope():
    beam = two_span_beam()
    env = cba.make_span_udl_cases(beam, w=10.0).envelope()

    unique_x, unique_index = np.unique(env.x, return_index=True)
    for x in [3.0, 6.0, 8.0, 10.0, 13.0]:
        got = env.at(x)
        assert got["Mmax"] == pytest.approx(
            np.interp(x, unique_x, env.Mmax[unique_index])
        )
        assert got["Vmin"] == pytest.approx(
            np.interp(x, unique_x, env.Vmin[unique_index])
        )

    # Custom attrs subset.
    sub = env.at(5.0, attrs=("Mmax",))
    assert set(sub) == {"Mmax"}


# ---------------------------------------------------------------------------
# LoadCases.envelope
# ---------------------------------------------------------------------------
def test_loadcases_envelope_matches_manual_build():
    beam = two_span_beam()
    cases = cba.make_span_udl_cases(beam, w=10.0)

    env = cases.envelope()

    from pycba.load_cases import build_pycba_model

    results = []
    for case in cases:
        model = build_pycba_model(beam, case)
        model.analyze(beam.npts)
        results.append(model.beam_results)
    manual = Envelopes(results)

    assert np.allclose(env.Mmax, manual.Mmax)
    assert np.allclose(env.Mmin, manual.Mmin)
    assert np.allclose(env.Vmax, manual.Vmax)

    # A full, plottable Envelopes.
    fig, ax = env.plot()
    assert fig is not None and ax is not None


# ---------------------------------------------------------------------------
# add_all_spans_udl
# ---------------------------------------------------------------------------
def test_loadcases_add_all_spans_udl_matches_explicit_loop():
    beam = cba.BeamAnalysis(L=[6.0, 6.0, 6.0], EI=30e6, R=[-1, 0, -1, 0, -1, 0, -1, 0])

    # New one-call path.
    new = cba.LoadCases(beam)
    new.add_all_spans_udl("Gk", 25.0)
    new.add_all_spans_udl("Qk", 10.0)

    # Verbose explicit-loop path.
    old = cba.LoadCases(beam)
    gk = old.add_case("Gk")
    qk = old.add_case("Qk")
    for i_span in range(1, beam.beam.no_spans + 1):
        gk.add_udl(i_span, 25.0)
        qk.add_udl(i_span, 10.0)

    assert new.to_LM() == old.to_LM()
    assert new.case("Gk").loads == [[1, 1, 25.0], [2, 1, 25.0], [3, 1, 25.0]]


def test_loadcase_add_all_spans_udl_is_fluent_and_distinct_from_add_span_udl():
    beam = cba.BeamAnalysis(L=[6.0, 6.0], EI=30e6, R=[-1, 0, -1, 0, -1, 0])

    case = cba.LoadCase("Gk")
    returned = case.add_all_spans_udl(beam, 25.0)
    assert returned is case  # fluent
    assert case.loads == [[1, 1, 25.0], [2, 1, 25.0]]

    # Distinct from add_span_udl: one combined case vs one case per span.
    per_span = cba.LoadCases(beam)
    added = per_span.add_span_udl(25.0)
    assert len(added) == beam.beam.no_spans
    assert len(case.loads) == beam.beam.no_spans  # but all in one case


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
def test_sign_selective_envelope_exported_at_top_level():
    from pycba import load_cases as load_case_tools

    assert hasattr(cba, "sign_selective_envelope")
    assert cba.sign_selective_envelope is load_case_tools.sign_selective_envelope
