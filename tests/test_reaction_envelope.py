# -*- coding: utf-8 -*-
"""Tests for the reaction-envelope plotting improvements (extreme markers)."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pycba as cba


def _bridge(spans=(20.0, 25.0, 20.0)):
    L = list(spans)
    EI = 30e11 * 1e-6 * np.ones(len(L))
    R = [-1, 0] * (len(L) + 1)
    bridge = cba.BeamAnalysis(L, EI, R)
    return cba.BridgeAnalysis(bridge, cba.Vehicle([1.2, 1.2], [60, 60, 60]))


def test_mark_reaction_extreme_adds_marker_and_label():
    fig, ax = plt.subplots()
    pos = np.linspace(0.0, 10.0, 11)
    series = np.array([0, 1, 2, 9, 4, 3, 2, 1, 0, 0, 0], dtype=float)  # peak 9 @ idx 3
    cba.BridgeAnalysis._mark_reaction_extreme(ax, pos, series, "r", np.argmax)
    # a marker line at the extreme
    markers = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert markers and markers[0].get_xydata()[0][1] == 9.0
    # a value annotation "9"
    assert any("9" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_plot_envelopes_consistent_path_runs_and_marks():
    ba = _bridge()
    env = ba.run_vehicle(2.0)
    ba.plot_envelopes(env)  # consistent path (pos length == Rmax columns)
    # the figure now exists with marked reaction axes; just assert no error and
    # that some axes carry the 'o' extreme markers
    fig = plt.gcf()
    has_markers = any(
        ln.get_marker() == "o" for axx in fig.get_axes() for ln in axx.lines
    )
    assert has_markers
    plt.close("all")


def test_plot_envelopes_incompatible_bar_path_runs():
    ba = _bridge(spans=(20.0, 25.0))
    e1 = ba.run_vehicle(1.0)
    e2 = ba.run_vehicle(2.5)
    envenv = cba.Envelopes.zero_like(e1)
    envenv.augment(e1)
    envenv.augment(e2)
    # incompatible reaction shape -> bar (envelope-of-envelopes) path with labels
    ba.plot_envelopes(envenv)
    plt.close("all")
