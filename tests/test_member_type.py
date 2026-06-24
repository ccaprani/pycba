# -*- coding: utf-8 -*-
"""Tests for the MemberType enum and friendlier element-type API."""

import enum
import numpy as np
import pytest
import pycba as cba


def test_membertype_is_int_enum_with_codes():
    assert issubclass(cba.MemberType, enum.IntEnum)
    assert (
        int(cba.MemberType.FF),
        int(cba.MemberType.FP),
        int(cba.MemberType.PF),
        int(cba.MemberType.PP),
    ) == (1, 2, 3, 4)


def test_coerce_accepts_int_enum_and_name():
    assert cba.MemberType.coerce(2) == 2
    assert cba.MemberType.coerce(cba.MemberType.FP) == 2
    assert cba.MemberType.coerce("FP") == 2
    assert cba.MemberType.coerce("fp") == 2  # case-insensitive
    assert cba.MemberType.coerce(np.array([3.0])) == 3  # default eletype row shape


def test_coerce_rejects_bad_values():
    with pytest.raises(ValueError, match="Unknown member type"):
        cba.MemberType.coerce("XX")
    with pytest.raises(ValueError, match="eletype must be 1-4"):
        cba.MemberType.coerce(7)


def _Ma(eletype):
    """Hogging moment at the fixed end of a fixed-pinned span under a central PL."""
    P, L, EI = 10.0, 10.0, 30 * 600e7 * 1e-6
    ba = cba.BeamAnalysis(
        [L], EI, [-1, -1, -1, -1], [[1, 2, P, 0.5 * L, 0]], eletype=eletype
    )
    ba.analyze()
    return ba.beam_results.results.M[1], ba.beam.mbr_eletype


def test_eletype_forms_are_equivalent():
    m_int, e_int = _Ma([2])
    m_enum, e_enum = _Ma([cba.MemberType.FP])
    m_str, e_str = _Ma(["FP"])
    assert m_int == pytest.approx(-3 * 10.0 * 10.0 / 16)  # -3PL/16
    assert m_enum == pytest.approx(m_int)
    assert m_str == pytest.approx(m_int)
    # stored as plain ints regardless of the input form
    assert e_int == e_enum == e_str == [2]
    assert all(isinstance(x, int) for x in e_str)


def test_default_eletype_still_works():
    ba = cba.BeamAnalysis(
        [5.0, 6.0], 1e8, [-1, 0, -1, 0, -1, 0], [[1, 1, 10], [2, 1, 10]]
    )
    assert ba.analyze() == 0
    assert ba.beam.mbr_eletype == [1, 1]


def test_add_member_friendly_api():
    beam = cba.Beam()
    out = beam.add_member(5.0, 1e8, "PP")
    beam.add_member(6.0, 1e8, cba.MemberType.FF)
    beam.add_member(4.0, 1e8)  # default FF
    assert beam.mbr_eletype == [4, 1, 1]
    assert out is None  # add_member mirrors add_span (no return)
