"""
Tests for the beam/loading schematic renderers (:mod:`pycba.render`).

The matplotlib backend is exercised with the non-interactive ``Agg`` backend.
The TikZ/stanli backend is checked by string assertions (no LaTeX needed); an
optional end-to-end compile test runs only when ``pdflatex`` is available.
"""
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import pycba as cba
from pycba.render import (
    BeamPlotter,
    FIXED,
    PIN,
    ROLLER,
    SPRING,
)
from pycba.utils import parse_beam_string

# The renderer supports the high-level load-case classes by duck typing, so the
# high-level tests skip gracefully on any build where they are unavailable.
try:
    from pycba import LoadCase, LoadCases, LoadCombination

    HAS_LOADCASES = True
except ImportError:  # pragma: no cover - depends on pycba version
    HAS_LOADCASES = False

requires_loadcases = pytest.mark.skipif(
    not HAS_LOADCASES, reason="pycba LoadCase/LoadCombination not available"
)


def make_beam(beam_str, LM=None):
    L, EI, R, eType = parse_beam_string(beam_str)
    return cba.Beam(L, EI, R, LM=LM if LM is not None else [], eletype=eType)


# --------------------------------------------------------------------------- #
# Support inference
# --------------------------------------------------------------------------- #
def test_support_inference_pin_then_rollers():
    bp = BeamPlotter(make_beam("P7.5R7.0R"))
    kinds = [s.kind for s in bp.supports]
    assert kinds == [PIN, ROLLER, ROLLER]
    assert [round(s.x, 2) for s in bp.supports] == [0.0, 7.5, 14.5]


def test_support_inference_encastre_and_free():
    bp = BeamPlotter(make_beam("E10F"))  # cantilever
    assert [s.kind for s in bp.supports] == [FIXED]
    assert bp.supports[0].at_left_end


def test_support_inference_propped_cantilever_has_no_pin():
    # A fixed support provides horizontal restraint, so the other support is a
    # roller, not a pin.
    bp = BeamPlotter(make_beam("E10R"))
    assert [s.kind for s in bp.supports] == [FIXED, ROLLER]


def test_support_inference_spring():
    # Pin at left, vertical spring (k=1500) at right.
    beam = cba.Beam([10.0], [30e3], [-1, 0, 1500, 0], LM=[], eletype=[1])
    assert [s.kind for s in BeamPlotter(beam).supports] == [PIN, SPRING]


def test_internal_hinge_inferred_from_eletype():
    bp = BeamPlotter(make_beam("E20R20H20R"))
    assert [round(h.x, 1) for h in bp.hinges] == [40.0]


def test_showcase_beam_all_features():
    # The documented showcase: encastre + rollers + spring + internal hinge,
    # with a UDL, point load, partial UDL and moment.
    L, EI, R, eType = parse_beam_string("E6R6H6R6R")
    R[6] = 2000.0  # node 3 -> vertical spring support
    beam = cba.Beam(
        L,
        EI,
        R,
        eletype=eType,
        LM=[[1, 1, 20], [2, 2, 50, 3.0], [3, 3, 12, 1.0, 4.0], [4, 4, 40, 3.0]],
    )
    bp = BeamPlotter(beam)
    assert [s.kind for s in bp.supports] == [FIXED, ROLLER, SPRING, ROLLER]
    assert [round(h.x, 1) for h in bp.hinges] == [12.0]
    # Both backends render the full beam without error.
    ax = beam.plot()
    assert ax.patches
    plt.close(ax.figure)
    tx = beam.to_tikz()
    assert tx.count(r"\support") == 4
    assert r"\support{4}{a}[-90]" in tx  # left-end fixed -> vertical wall
    assert r"\support{5}" in tx  # spring
    assert r"\hinge" in tx
    # The partial UDL (span 3, the 2nd distributed load) has its extent
    # dimensioned; the full-span UDL on span 1 does not.
    assert r"\dimensioning{1}{dl1a}{dl1b}" in tx
    assert r"\dimensioning{1}{dl0a}{dl0b}" not in tx


def test_tikz_fixed_support_rotation():
    # The encastre wall is rotated to vertical, and the left/right ends differ
    # by 180 degrees so the wall faces outward.
    assert r"\support{4}{a}[-90]" in make_beam("E10R").to_tikz()
    assert r"\support{4}{b}[90]" in make_beam("R10E").to_tikz()


def test_partial_udl_extent_dimensioned():
    # A partial UDL gets an extent dimension; a full-span UDL does not.
    beam = make_beam("P10R", LM=[[1, 3, 10, 2, 5]])
    # The extent follows load_values (a load annotation), so it shows even when
    # the span dimensions are off, and is removed with load_values=False.
    assert r"\dimensioning{1}{dl0a}{dl0b}" in beam.to_tikz(dimensions=False)
    assert r"\dimensioning{1}{dl0a}{dl0b}" not in beam.to_tikz(load_values=False)
    full = make_beam("P10R", LM=[[1, 1, 10]]).to_tikz()
    assert r"\dimensioning{1}{dl0a}{dl0b}" not in full


# --------------------------------------------------------------------------- #
# Load normalisation
# --------------------------------------------------------------------------- #
def test_load_normalisation_to_global_coords():
    beam = make_beam("P10R10R", LM=[[1, 1, 20], [2, 2, 50, 4], [1, 4, 30, 5]])
    bp = BeamPlotter(beam)
    assert len(bp.dist_loads) == 1
    assert (bp.dist_loads[0].x0, bp.dist_loads[0].x1) == (0.0, 10.0)
    # point load in span 2 (offset 10) at a=4 -> global x = 14
    assert len(bp.point_loads) == 1 and bp.point_loads[0].x == 14.0
    # moment in span 1 at a=5 -> global x = 5
    assert len(bp.moment_loads) == 1 and bp.moment_loads[0].x == 5.0


def test_partial_and_trapezoidal_loads():
    beam = make_beam("P10R", LM=[[1, 3, 10, 2, 5], [1, 5, 0, 12]])
    bp = BeamPlotter(beam)
    # Two distributed loads: a partial UDL [2, 7] and a triangular 0->12 [0, 10]
    spans = sorted((round(d.x0, 1), round(d.x1, 1), d.w0, d.w1) for d in bp.dist_loads)
    assert spans == [(0.0, 10.0, 0, 12), (2.0, 7.0, 10, 10)]


# --------------------------------------------------------------------------- #
# Load source resolution (LM / LoadCase / LoadCombination / structure-only)
# --------------------------------------------------------------------------- #
def test_default_uses_beam_lm():
    beam = make_beam("P10R", LM=[[1, 1, 5]])
    assert len(BeamPlotter(beam).dist_loads) == 1


def test_structure_only_with_empty_loads():
    beam = make_beam("P10R", LM=[[1, 1, 5]])
    bp = BeamPlotter(beam, loads=[])
    assert bp.dist_loads == [] and bp.point_loads == []


@requires_loadcases
def test_loadcase_source():
    beam = make_beam("P10R10R")
    lc = LoadCase("dead").add_udl(1, 15).add_udl(2, 15)
    bp = BeamPlotter(beam, lc)
    assert len(bp.dist_loads) == 2


@requires_loadcases
def test_loadcombination_requires_load_cases():
    beam = make_beam("P10R10R")
    cases = LoadCases(
        beam,
        [LoadCase("dead").add_udl(1, 15), LoadCase("live").add_pl(2, 60, 5)],
    )
    comb = LoadCombination("uls", factors=[1.35, 1.5])
    bp = BeamPlotter(beam, comb, load_cases=cases)
    assert len(bp.dist_loads) == 1 and len(bp.point_loads) == 1
    # factored magnitude
    assert bp.dist_loads[0].w0 == pytest.approx(1.35 * 15)
    assert bp.point_loads[0].P == pytest.approx(1.5 * 60)

    with pytest.raises(ValueError):
        BeamPlotter(beam, comb)  # missing load_cases


# --------------------------------------------------------------------------- #
# matplotlib backend
# --------------------------------------------------------------------------- #
def test_plot_returns_axes_and_draws():
    beam = make_beam("P7.5R7.0R", LM=[[1, 1, 20], [2, 1, 20]])
    ax = beam.plot()
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.lines  # beam line + supports drawn as lines
    assert ax.patches  # support polygons / load fills
    assert ax.get_xlabel() == "Distance along beam (m)"
    plt.close(ax.figure)


def test_plot_accepts_existing_axes():
    fig, ax = plt.subplots()
    out = make_beam("P10R").plot(ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_toggles_do_not_error():
    beam = make_beam("P10R10R", LM=[[1, 1, 10], [2, 2, 30, 5]])
    ax = beam.plot(dimensions=False, labels=False, load_values=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)


def _make_analysis(beam_str="P7.5R7.0R", LM=None):
    L, EI, R, eType = parse_beam_string(beam_str)
    if LM is None:
        LM = [[1, 1, 20], [2, 2, 50, 3]]
    return cba.BeamAnalysis(L, EI, R, LM=LM, eletype=eType)


def test_beamanalysis_plot_beam_convenience():
    # plot_beam() on the analysis object mirrors beam.plot() without
    # reaching through .beam, just as plot_results() does.
    ba = _make_analysis()
    ax = ba.plot_beam()
    assert isinstance(ax, matplotlib.axes.Axes)
    assert ax.patches  # supports / loads drawn
    plt.close(ax.figure)

    # load source can still be overridden, e.g. bare structure
    ax2 = ba.plot_beam(loads=[])
    assert isinstance(ax2, matplotlib.axes.Axes)
    plt.close(ax2.figure)


def test_plot_beam_tikz_returns_source():
    # tikz=True selects the TikZ backend and returns LaTeX, no .beam indirection
    ba = _make_analysis()
    src = ba.plot_beam(tikz=True)
    assert isinstance(src, str)
    assert "\\begin{tikzpicture}" in src
    assert "\\documentclass" in src  # standalone by default


def test_plot_beam_tex_extension_infers_tikz(tmp_path):
    # a .tex target selects the TikZ backend without needing tikz=True
    ba = _make_analysis()
    out = ba.plot_beam(save=tmp_path / "beam.tex")
    assert out == tmp_path / "beam.tex"
    assert out.exists()
    assert "\\begin{tikzpicture}" in out.read_text()


def test_plot_beam_mpl_save_writes_image(tmp_path):
    # a non-.tex target stays on the matplotlib backend
    ba = _make_analysis()
    png = tmp_path / "schematic.png"
    ax = ba.plot_beam(save=png)
    assert isinstance(ax, matplotlib.axes.Axes)  # mpl backend still returns Axes
    assert png.exists() and png.stat().st_size > 0
    plt.close(ax.figure)


def test_beam_plot_tikz_equivalent_to_to_tikz():
    # the unified front door agrees with the explicit utility method
    beam = make_beam("P7.5R7.0R", LM=[[1, 1, 20]])
    assert beam.plot(tikz=True) == beam.to_tikz()


def test_plot_results_overlay_returns_handles():
    ba = _make_analysis()
    ba.analyze()
    fig, axs = ba.plot_results(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(axs) == 5  # schematic + M + V + D + reactions
    # top panel carries the schematic (support/load patches)
    assert axs[0].patches
    plt.close(fig)


def test_plot_results_without_beam_is_three_panels():
    ba = _make_analysis()
    ba.analyze()
    fig, axs = ba.plot_results(show_beam=False, show_reactions=False, show=False)
    assert len(axs) == 3
    assert not axs[0].patches  # no schematic, just the moment diagram
    plt.close(fig)


def test_plot_results_before_analyze_warns_and_returns_none():
    ba = _make_analysis()
    assert ba.plot_results(show=False) is None


def test_plot_bmd_sfd_deflection_axes_and_convention():
    ba = cba.BeamAnalysis([10.0], 1.0e5, [-1, -1, -1, 0])
    ba.add_udl(1, 10.0)
    ba.analyze()
    ax_m = ba.plot_bmd()
    assert ax_m.yaxis_inverted()  # sagging-positive: moment y-axis inverted
    ax_v = ba.plot_sfd()
    assert not ax_v.yaxis_inverted()
    ax_d = ba.plot_dsd()
    assert ax_d is not None and not ax_d.yaxis_inverted()
    plt.close("all")


def test_plot_bmd_overlay_on_existing_axes():
    ti = cba.BeamAnalysis([10.0], 1.0e5, [-1, -1, -1, 0], GAv=2.0e4)
    ti.add_udl(1, 10.0)
    ti.analyze()
    eb = cba.BeamAnalysis([10.0], 1.0e5, [-1, -1, -1, 0])
    eb.add_udl(1, 10.0)
    eb.analyze()
    ax = ti.plot_bmd(label="Timoshenko")
    n0 = len(ax.lines)
    out = eb.plot_bmd(ax=ax, color="0.5", ls="--", label="Euler-Bernoulli")
    assert out is ax  # overlay returns the same axes
    assert len(ax.lines) == n0 + 1  # only the curve is added, no re-setup
    assert ax.yaxis_inverted()  # overlay must not flip the axis back
    plt.close("all")


def test_plot_diagram_before_analyze_returns_none():
    ba = cba.BeamAnalysis([10.0], 1.0e5, [-1, -1, -1, 0])
    assert ba.plot_bmd() is None
    assert ba.plot_sfd() is None
    assert ba.plot_dsd() is None


def test_repr_beam_and_analysis():
    ba = _make_analysis("P7.5R7.0R", LM=[[1, 1, 20], [2, 2, 50, 3]])
    assert repr(ba) == "BeamAnalysis(2 spans, 3 supports, 2 loads, not analysed)"
    assert repr(ba.beam) == "Beam(2 spans, L=14.5, 3 supports, 2 loads)"
    ba.analyze()
    assert "analysed" in repr(ba) and "not analysed" not in repr(ba)


# --------------------------------------------------------------------------- #
# TikZ / stanli backend (string assertions)
# --------------------------------------------------------------------------- #
def test_tikz_basic_structure():
    s = make_beam("P7.5R7.0R", LM=[[1, 1, 20], [2, 1, 20]]).to_tikz()
    assert r"\documentclass" in s  # standalone by default
    assert r"\begin{tikzpicture}" in s and r"\end{tikzpicture}" in s
    assert r"\usepackage{stanli}" in s
    assert s.count(r"\beam{4}") == 2  # two spans
    assert s.count(r"\support") == 3  # three supports
    assert r"\lineload{1}" in s


def test_tikz_point_coordinates_formatted():
    s = make_beam("P7.5R7.0R").to_tikz()
    assert r"\point{a}{0}{0}" in s
    assert r"\point{b}{7.5}{0}" in s
    assert r"\point{c}{14.5}{0}" in s


def test_tikz_support_macros():
    s = make_beam("E10R").to_tikz()
    assert r"\support{4}" in s  # encastre
    assert r"\support{2ooo}" in s  # roller


def test_tikz_spring_support_macro():
    beam = cba.Beam([10.0], [30e3], [-1, 0, 1500, 0], LM=[], eletype=[1])
    assert r"\support{5}" in beam.to_tikz()


def test_tikz_point_and_moment_macros():
    s = make_beam("P10R10R", LM=[[1, 2, 50, 5], [2, 4, 30, 5]]).to_tikz()
    assert r"\load{1}" in s  # point load
    assert r"\load{2}" in s or r"\load{3}" in s  # moment (sign-dependent)


def test_tikz_internal_hinge_macro():
    s = make_beam("E20R20H20R").to_tikz()
    assert r"\hinge{2}" in s


def test_tikz_standalone_toggle():
    beam = make_beam("P10R")
    assert r"\documentclass" not in beam.to_tikz(standalone=False)
    assert r"\begin{tikzpicture}" in beam.to_tikz(standalone=False)


def test_tikz_dimensions_and_labels_toggle():
    beam = make_beam("P10R")
    assert r"\dimensioning" in beam.to_tikz(dimensions=True)
    assert r"\dimensioning" not in beam.to_tikz(dimensions=False)
    assert r"\notation{1}" in beam.to_tikz(labels=True)


def test_tikz_load_values_toggle():
    beam = make_beam("P10R", LM=[[1, 1, 20]])
    assert "kN/m" in beam.to_tikz(load_values=True)
    assert "kN/m" not in beam.to_tikz(load_values=False)


# --------------------------------------------------------------------------- #
# save_tikz (+ optional compile)
# --------------------------------------------------------------------------- #
def test_save_tikz_writes_file(tmp_path):
    out = make_beam("P10R", LM=[[1, 1, 20]]).save_tikz(tmp_path / "beam")
    assert out.suffix == ".tex" and out.exists()
    assert r"\begin{tikzpicture}" in out.read_text()


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_save_tikz_compiles(tmp_path):
    beam = make_beam("P7.5R7.0R", LM=[[1, 1, 20], [2, 1, 20]])
    pdf = beam.save_tikz(tmp_path / "beam.tex", compile=True)
    assert pdf.suffix == ".pdf" and pdf.exists() and pdf.stat().st_size > 0
