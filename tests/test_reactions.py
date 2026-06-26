# -*- coding: utf-8 -*-
"""
Tests for the reaction recovery (``_reactions_by_node``) and the reaction
plots (``plot_reactions`` standalone and the reactions panel in
``plot_results``).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pycba as cba


def test_reactions_simple_span():
    # UDL w=10 over a 10 m simple span -> +50 / +50 (positive upward)
    ba = cba.BeamAnalysis([10.0], 1.0, [-1, 0, -1, 0], LM=[[1, 1, 10, 0, 0]])
    ba.analyze()
    vert, mom = ba._reactions_by_node()
    assert vert[0] == pytest.approx(50.0)
    assert vert[1] == pytest.approx(50.0)
    assert not mom  # pinned/roller -> no moment reactions
    assert sum(vert.values()) == pytest.approx(100.0)  # equilibrium


def test_reactions_propped_cantilever():
    # Fixed at A, roller at B, UDL w=12 over 8 m.
    ba = cba.BeamAnalysis([8.0], 1.0, [-1, -1, -1, 0], LM=[[1, 1, 12, 0, 0]])
    ba.analyze()
    vert, mom = ba._reactions_by_node()
    assert vert[0] == pytest.approx(60.0)  # 5wL/8
    assert vert[1] == pytest.approx(36.0)  # 3wL/8
    assert mom[0] == pytest.approx(96.0)  # wL^2/8 fixing moment
    assert sum(vert.values()) == pytest.approx(96.0)


def test_reactions_two_span_continuous():
    ba = cba.BeamAnalysis(
        [10.0, 10.0],
        1e7,
        [-1, 0, -1, 0, -1, 0],
        LM=[[1, 1, 20, 0, 0], [2, 1, 20, 0, 0]],
    )
    ba.analyze()
    vert, _ = ba._reactions_by_node()
    assert vert[0] == pytest.approx(75.0)  # 0.375 wL
    assert vert[1] == pytest.approx(250.0)  # 1.25 wL
    assert vert[2] == pytest.approx(75.0)
    assert sum(vert.values()) == pytest.approx(400.0)


def test_reactions_include_spring_forces():
    # Vertical spring at each end (k=1e3), point load P=10 at mid-span.
    ba = cba.BeamAnalysis([10.0], 1e6, [1e3, 0, 1e3, 0], LM=[[1, 2, 10, 5.0, 0]])
    ba.analyze()
    vert, _ = ba._reactions_by_node()
    # symmetric -> 5 kN each, upward, summing to the applied 10 kN
    assert vert[0] == pytest.approx(5.0, abs=1e-6)
    assert vert[1] == pytest.approx(5.0, abs=1e-6)
    assert sum(vert.values()) == pytest.approx(10.0, abs=1e-6)


def test_plot_reactions_standalone():
    ba = cba.BeamAnalysis([8.0], 1.0, [-1, -1, -1, 0], LM=[[1, 1, 12, 0, 0]])
    ba.analyze()
    fig, ax = ba.plot_reactions(show=False)
    # arrows (annotations) for the vertical reactions + a label per reaction
    assert len(ax.texts) > 0
    assert ax.patches or ax.lines  # the schematic was drawn
    plt.close("all")


def test_plot_reactions_before_analysis_returns_none():
    ba = cba.BeamAnalysis([8.0], 1.0, [-1, 0, -1, 0])
    assert ba.plot_reactions(show=False) is None


def test_plot_results_reactions_panel():
    ba = cba.BeamAnalysis(
        [10.0, 10.0],
        1e7,
        [-1, 0, -1, 0, -1, 0],
        LM=[[1, 1, 20, 0, 0], [2, 1, 20, 0, 0]],
    )
    ba.analyze()
    # default: schematic + M + V + D + reactions = 5 panels
    fig, axs = ba.plot_results(show=False)
    assert len(axs) == 5
    plt.close("all")
    # opt out of the reactions panel -> back to 4
    fig, axs = ba.plot_results(show=False, show_reactions=False)
    assert len(axs) == 4
    plt.close("all")
    # bare diagrams only
    fig, axs = ba.plot_results(show=False, show_beam=False, show_reactions=False)
    assert len(axs) == 3
    plt.close("all")
