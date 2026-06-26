# -*- coding: utf-8 -*-
"""
Tests for the member-naming conversion: ``no_members`` (with the kept
``no_spans`` alias), the ``i_member`` load-method parameter, and the member
wording in the reprs.
"""

import pycba as cba


def test_no_members_with_span_alias():
    ba = cba.BeamAnalysis([5.0, 6.0], 1e7, [-1, 0, -1, 0, -1, 0])
    assert ba.beam.no_members == 2
    assert ba.beam.no_spans == 2  # deprecated alias still works


def test_i_member_keyword_and_positional():
    ba = cba.BeamAnalysis([10.0], 1e7, [-1, 0, -1, 0])
    ba.add_udl(i_member=1, w=10)  # new keyword name
    ba.add_pl(1, 20, 5.0)  # positional call is unchanged
    ba.add_trap(i_member=1, w1=2, w2=4)
    ba.analyze()
    assert ba.beam_results is not None


def test_reprs_say_members():
    ba = cba.BeamAnalysis([5.0, 6.0], 1e7, [-1, 0, -1, 0, -1, 0])
    assert "member" in repr(ba) and "span" not in repr(ba)
    assert "member" in repr(ba.beam) and "span" not in repr(ba.beam)


def test_add_member_builds_beam():
    b = cba.Beam()
    b.add_member(10.0, 1e7)
    b.add_span(8.0, 1e7)  # legacy alias
    assert b.no_members == 2
