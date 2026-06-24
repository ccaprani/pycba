# -*- coding: utf-8 -*-
"""Tests for the friendly named ``supports=`` API and ``supports_to_R``."""

import numpy as np
import pytest
import pycba as cba
from pycba.utils import supports_to_R


# ---------------------------------------------------------------------------
# supports_to_R: the lowering helper
# ---------------------------------------------------------------------------
def test_names_lower_to_expected_R():
    assert supports_to_R(["p", "r", "r", "e"]) == [-1, 0, -1, 0, -1, 0, -1, -1]
    assert supports_to_R(["f", "p"]) == [0, 0, -1, 0]


def test_letter_word_aliases_are_equivalent():
    # pin/roller -> [-1, 0]; encastre/fixed/clamped -> [-1, -1]; free -> [0, 0]
    assert supports_to_R(["pin", "roller", "encastre", "free"]) == supports_to_R(
        ["p", "r", "e", "f"]
    )
    assert supports_to_R(["pinned"]) == supports_to_R(["p"])
    assert (
        supports_to_R(["fixed"]) == supports_to_R(["clamped"]) == supports_to_R(["e"])
    )


def test_case_insensitive_and_whitespace_tolerant():
    assert supports_to_R([" Pin ", "ROLLER", "Encastre"]) == [-1, 0, -1, 0, -1, -1]


def test_spring_given_as_raw_pair():
    # A raw [vertical, rotation] pair is the escape hatch (and the spring form).
    assert supports_to_R(["p", [5e4, 0], "p"]) == [-1, 0, 5e4, 0, -1, 0]
    # A raw pair may also restate a named support.
    assert supports_to_R([[-1, 0], [-1, -1]]) == [-1, 0, -1, -1]


def test_n_nodes_validation():
    assert supports_to_R(["p", "r", "r"], n_nodes=3) == [-1, 0, -1, 0, -1, 0]
    with pytest.raises(ValueError, match="Expected 3 supports"):
        supports_to_R(["p", "r"], n_nodes=3)


def test_unknown_name_rejected():
    with pytest.raises(ValueError, match="Unknown support 'x'"):
        supports_to_R(["p", "x"])


def test_hinge_rejected_with_guidance():
    # A hinge is a member release, not a support: it must not appear in supports.
    with pytest.raises(ValueError, match="hinge is an internal moment release"):
        supports_to_R(["p", "hinge", "r"])
    with pytest.raises(ValueError, match="hinge"):
        supports_to_R(["p", "h", "r"])


def test_bad_pair_length_rejected():
    with pytest.raises(ValueError, match="exactly 2 entries"):
        supports_to_R(["p", [1, 2, 3]])


# ---------------------------------------------------------------------------
# supports= on the constructors
# ---------------------------------------------------------------------------
def _two_span_inputs():
    L = [7.5, 7.0]
    EI = 30 * 600e7 * 1e-6
    LM = [[1, 1, 20, 0, 0], [2, 1, 20, 0, 0]]
    return L, EI, LM


def test_supports_equivalent_to_R_full_analysis():
    L, EI, LM = _two_span_inputs()
    ba_R = cba.BeamAnalysis(L, EI, [-1, 0, -1, 0, -1, 0], LM)
    ba_S = cba.BeamAnalysis(L, EI, LM=LM, supports=["p", "r", "r"])
    assert ba_R.analyze() == 0
    assert ba_S.analyze() == 0
    assert list(ba_S.beam.restraints) == list(ba_R.beam.restraints)
    assert np.allclose(ba_S.beam_results.R, ba_R.beam_results.R)
    assert np.allclose(ba_S.beam_results.results.M, ba_R.beam_results.results.M)


def test_supports_spring_equivalent_to_R_spring():
    L, EI, LM = _two_span_inputs()
    ba_R = cba.BeamAnalysis(L, EI, [-1, 0, 5e4, 0, -1, 0], LM)
    ba_S = cba.BeamAnalysis(L, EI, LM=LM, supports=["p", [5e4, 0], "p"])
    ba_R.analyze()
    ba_S.analyze()
    assert np.allclose(ba_S.beam_results.R, ba_R.beam_results.R)
    assert np.allclose(ba_S.beam_results.Rs, ba_R.beam_results.Rs)


def test_supports_on_beam_object():
    L, EI, _ = _two_span_inputs()
    beam = cba.Beam(L=L, EI=EI, eletype=np.ones((len(L), 1)), supports=["p", "r", "r"])
    assert list(beam.restraints) == [-1, 0, -1, 0, -1, 0]


def test_cantilever_free_end_via_supports():
    # Propped cantilever: encastre then free tip.
    L, EI = [5.0], 1e8
    ba = cba.BeamAnalysis(L, EI, LM=[[1, 1, 10]], supports=["e", "f"])
    assert ba.analyze() == 0
    assert list(ba.beam.restraints) == [-1, -1, 0, 0]


def test_both_R_and_supports_rejected():
    L, EI, LM = _two_span_inputs()
    with pytest.raises(ValueError, match="either R or supports"):
        cba.BeamAnalysis(L, EI, [-1, 0, -1, 0, -1, 0], LM, supports=["p", "r", "r"])


def test_neither_R_nor_supports_rejected():
    L, EI, _ = _two_span_inputs()
    with pytest.raises(ValueError, match="R or supports"):
        cba.BeamAnalysis(L, EI)


def test_wrong_support_count_rejected():
    L, EI, _ = _two_span_inputs()  # 2 spans -> 3 nodes expected
    with pytest.raises(ValueError, match="Expected 3 supports"):
        cba.BeamAnalysis(L, EI, supports=["p", "r"])


# ---------------------------------------------------------------------------
# parse_beam_string still agrees after the shared-table refactor
# ---------------------------------------------------------------------------
def test_parse_beam_string_unchanged():
    L, EI, R, eType = cba.parse_beam_string("P40R20R")
    assert R == [-1, 0, -1, 0, -1, 0]
    # An internal hinge: unsupported node + released previous member.
    L, EI, R, eType = cba.parse_beam_string("E20H30R10F")
    assert R == [-1, -1, 0, 0, -1, 0, 0, 0]
    assert eType == [2, 1, 1]
