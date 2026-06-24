"""
PyCBA - Post-tensioning equivalent-load preprocessor.

A **preprocessor** (not an analysis object): it turns a draped post-tensioning
tendon into the equivalent ("balanced") loads the tendon exerts on the concrete,
returned as an ordinary PyCBA load matrix that you apply to a :class:`Beam` /
:class:`BeamAnalysis` like any other loading.

The tendon is described **span by span** by a profile object whose geometry is
given as **eccentricities from the section centroid, positive below the
centroid** (so a sagging tendon balances gravity).  The supported profiles
mirror the standard library used by RAPT / PT Designer (12 profile types, 7 for
spans and 5 for cantilevers) - see the *PT Designer Theory Manual*, Chapters 5
and 6 (https://secure.skghoshassociates.com/product/PT/download/TheoryManual.pdf).

Rather than tabulate twelve sign-sensitive closed forms, the equivalent loads
are generated from first principles from the piecewise tendon profile ``e(x)``
(parabolic or straight segments):

* a parabolic segment of constant curvature ``e''`` contributes a uniform
  transverse load ``w = F·e''`` over its length;
* a slope change ``Δe'`` at an interior kink contributes a point load
  ``P = F·Δe'`` there (slope changes *at* a support are reacted directly and
  are not applied to the beam);
* the tendon eccentricity at each end **anchorage** contributes a moment
  ``M = F·e`` there.

With the eccentricity-positive-below convention and PyCBA's downward-positive
loads, a tendon sagging ``a`` below the chord of a simply-supported span gives an
upward ``w = -8 F a / L²`` and a midspan moment ``-F a``; a constant eccentricity
``e`` gives a uniform ``-F e``.  Applying the returned matrix and analysing gives
the **balanced** moment ``M_bal``; the secondary (parasitic) moment is
``M₂ = M_bal − F·e``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from .beam import Beam
from .load import LoadMatrix

__all__ = [
    "Parabola",
    "CompoundParabola",
    "Harp",
    "DoubleHarp",
    "equivalent_loads",
    "plot_tendon",
]

_TOL = 1e-9


@dataclass
class _Seg:
    """A tendon profile piece over span-local ``[x1, x2]`` (eccentricity +down)."""

    x1: float
    x2: float
    e1: float
    e2: float
    epp: float = 0.0  # constant second derivative e'' (0 => straight)

    @property
    def length(self) -> float:
        return self.x2 - self.x1

    def slope_start(self) -> float:
        # e(x2) = e1 + e'(x1) L + 1/2 e'' L^2  ->  e'(x1)
        L = self.length
        return (self.e2 - self.e1) / L - 0.5 * self.epp * L

    def slope_end(self) -> float:
        return self.slope_start() + self.epp * self.length


def _parabola_through(x1, e1, x2, e2, xm, em) -> float:
    """Constant ``e''`` of the parabola through three points (xm interior)."""
    # second divided difference * 2
    f01 = (em - e1) / (xm - x1)
    f12 = (e2 - em) / (x2 - xm)
    return 2.0 * (f12 - f01) / (x2 - x1)


# --------------------------------------------------------------------------- #
# Profile objects (geometry only; eccentricities are +below the centroid)
# --------------------------------------------------------------------------- #
@dataclass
class Parabola:
    """
    Parabolic tendon in a span (RAPT/PT-Designer Types 1, 2; 8, 9 on cantilevers).

    A single parabola between the two **high points**.  With ``c_left = c_right
    = 0`` the high points are at the support centrelines (Type 1 / centreline
    parabola); positive ``c_left`` / ``c_right`` move the high points inside the
    supports by those distances, with a straight tendon from each support to its
    high point (Type 2 / face-to-face parabola), which adds a downward point
    load at each high point.

    Parameters
    ----------
    e_left, e_mid, e_right : float
        Tendon eccentricity (positive below the centroid) at the left high
        point, the parabola mid-point (low point), and the right high point.
    c_left, c_right : float, optional
        Distance of each high point inside its support centreline (default 0).
    """

    e_left: float
    e_mid: float
    e_right: float
    c_left: float = 0.0
    c_right: float = 0.0

    def segments(self, L: float, cantilever: Optional[str] = None) -> List[_Seg]:
        cl, cr = self.c_left, self.c_right
        if cantilever == "right":  # support at x=0 (high point, zero slope), tip at L
            x1, x2 = cl, L
            Le = x2 - x1
            epp = 2.0 * (self.e_right - self.e_left) / Le**2  # zero slope at x1
            segs = [_Seg(x1, x2, self.e_left, self.e_right, epp)]
            if cl > _TOL:
                segs.insert(0, _Seg(0.0, x1, self.e_left, self.e_left, 0.0))
            return segs
        if cantilever == "left":  # tip at x=0, support at x=L (zero slope at L)
            x1, x2 = 0.0, L - cr
            Le = x2 - x1
            epp = 2.0 * (self.e_left - self.e_right) / Le**2  # zero slope at x2
            segs = [_Seg(x1, x2, self.e_left, self.e_right, epp)]
            if cr > _TOL:
                segs.append(_Seg(x2, L, self.e_right, self.e_right, 0.0))
            return segs
        x1, x2 = cl, L - cr
        Le = x2 - x1
        if Le <= _TOL:
            raise ValueError("Parabola: high points overlap (c_left + c_right >= L)")
        xm = 0.5 * (x1 + x2)
        epp = _parabola_through(x1, self.e_left, x2, self.e_right, xm, self.e_mid)
        segs = []
        if cl > _TOL:  # straight, horizontal, from support to left high point
            segs.append(_Seg(0.0, x1, self.e_left, self.e_left, 0.0))
        segs.append(_Seg(x1, x2, self.e_left, self.e_right, epp))
        if cr > _TOL:
            segs.append(_Seg(x2, L, self.e_right, self.e_right, 0.0))
        return segs


@dataclass
class CompoundParabola:
    """
    Compound parabola (RAPT/PT-Designer Type 3): four parabolas - concave down
    over each support and concave up between - with points of contraflexure.

    Parameters
    ----------
    e_left, e_mid, e_right : float
        Eccentricity (+below) at the left support, the low point, and the right
        support.
    a, b : float
        Points of contraflexure measured from the left and right support
        centrelines respectively.
    c : float
        Low-point location measured from the left support centreline
        (``b < c < L - a`` is required so the four pieces are well formed).
    """

    e_left: float
    e_mid: float
    e_right: float
    a: float
    b: float
    c: float

    def segments(self, L: float, cantilever: Optional[str] = None) -> List[_Seg]:
        if cantilever is not None:
            raise NotImplementedError(
                "CompoundParabola is not supported on a cantilever span (RAPT "
                "Type 12); use Parabola for a cantilever tendon."
            )
        a, b, c = self.a, self.b, self.c
        d = L - b  # right contraflexure measured from the left
        if not (0 < a < c < d < L):
            raise ValueError(
                "CompoundParabola: require 0 < a < c < (L-b) < L (check a, b, c)"
            )
        # The four parabolas join with matching slope at the contraflexures a, d
        # and at the low point c. Reproduce PT Designer's sub-sags:
        el, em, er = self.e_left, self.e_mid, self.e_right
        a1 = (a / c) * (em - el)
        a2 = (em - el) - a1
        a4 = (b / d) * (em - er)
        a3 = (em - er) - a4
        # epp of each segment from sag over its half-span (sag a_i over length):
        # piece 1 [0,a] concave down, piece 2 [a,c] concave up,
        # piece 3 [c,d] concave up, piece 4 [d,L] concave down.
        e_a = el + a1  # eccentricity at contraflexure a
        e_d = er + a4  # eccentricity at contraflexure d
        epp1 = 2.0 * a1 / (a**2)  # concave down near support -> e'' < 0? sign via sag
        epp2 = -2.0 * a2 / ((c - a) ** 2)
        epp3 = -2.0 * a3 / ((d - c) ** 2)
        epp4 = 2.0 * a4 / (b**2)
        return [
            _Seg(0.0, a, el, e_a, epp1),
            _Seg(a, c, e_a, em, epp2),
            _Seg(c, d, em, e_d, epp3),
            _Seg(d, L, e_d, er, epp4),
        ]


@dataclass
class Harp:
    """
    Single-point harped tendon (RAPT/PT-Designer Types 4, 5; 10, 11 cantilever):
    straight segments with one bend.

    Parameters
    ----------
    e_left, e_mid, e_right : float
        Eccentricity (+below) at the left high point, the bend, and the right
        high point.
    a : float
        Bend location measured from the left support centreline.
    c_left, c_right : float, optional
        High points inside the supports (Type 5) instead of at the centrelines
        (Type 4).
    """

    e_left: float
    e_mid: float
    e_right: float
    a: float
    c_left: float = 0.0
    c_right: float = 0.0

    def segments(self, L: float, cantilever: Optional[str] = None) -> List[_Seg]:
        cl, cr = self.c_left, self.c_right
        xb = self.a
        xl, xr = cl, L - cr
        if not (xl < xb < xr):
            raise ValueError("Harp: require c_left < a < L - c_right")
        _ = cantilever  # straight-segment profile is identical on a cantilever
        segs: List[_Seg] = []
        if cl > _TOL:
            segs.append(_Seg(0.0, xl, self.e_left, self.e_left, 0.0))
        segs.append(_Seg(xl, xb, self.e_left, self.e_mid, 0.0))
        segs.append(_Seg(xb, xr, self.e_mid, self.e_right, 0.0))
        if cr > _TOL:
            segs.append(_Seg(xr, L, self.e_right, self.e_right, 0.0))
        return segs


@dataclass
class DoubleHarp:
    """
    Double-point harped tendon (RAPT/PT-Designer Types 6, 7): straight segments
    with two bends.

    Parameters
    ----------
    e_left, e_1, e_2, e_right : float
        Eccentricity (+below) at the left high point, the two bends, and the
        right high point.
    a, b : float
        The two bend locations measured from the left support centreline.
    c_left, c_right : float, optional
        High points inside the supports (Type 7) instead of at the centrelines
        (Type 6).
    """

    e_left: float
    e_1: float
    e_2: float
    e_right: float
    a: float
    b: float
    c_left: float = 0.0
    c_right: float = 0.0

    def segments(self, L: float, cantilever: Optional[str] = None) -> List[_Seg]:
        cl, cr = self.c_left, self.c_right
        xa, xb = self.a, self.b
        xl, xr = cl, L - cr
        if not (xl < xa < xb < xr):
            raise ValueError("DoubleHarp: require c_left < a < b < L - c_right")
        _ = cantilever  # straight-segment profile is identical on a cantilever
        segs: List[_Seg] = []
        if cl > _TOL:
            segs.append(_Seg(0.0, xl, self.e_left, self.e_left, 0.0))
        segs.append(_Seg(xl, xa, self.e_left, self.e_1, 0.0))
        segs.append(_Seg(xa, xb, self.e_1, self.e_2, 0.0))
        segs.append(_Seg(xb, xr, self.e_2, self.e_right, 0.0))
        if cr > _TOL:
            segs.append(_Seg(xr, L, self.e_right, self.e_right, 0.0))
        return segs


Profile = Union[Parabola, CompoundParabola, Harp, DoubleHarp]


# --------------------------------------------------------------------------- #
# Equivalent-load assembly
# --------------------------------------------------------------------------- #
def _span_lengths(model) -> List[float]:
    beam = model.beam if hasattr(model, "beam") else model
    if not isinstance(beam, Beam):
        raise TypeError("equivalent_loads needs a Beam or BeamAnalysis")
    return list(beam.mbr_lengths)


def equivalent_loads(
    model,
    force: Union[float, Sequence[float]],
    profiles: Sequence[Optional[Profile]],
) -> LoadMatrix:
    """
    Build the post-tensioning equivalent-load matrix for a draped tendon.

    Parameters
    ----------
    model : pycba.Beam or pycba.BeamAnalysis
        Supplies the span lengths (and number of spans).
    force : float or sequence of float
        The (effective) prestress force ``F``.  A scalar applies to every span;
        otherwise one value per span (constant-force method).
    profiles : sequence of profile or None
        One profile per span (``None`` for an unstressed span).  Eccentricities
        are positive below the section centroid.

    Returns
    -------
    LoadMatrix
        A PyCBA load matrix (UDLs, partial UDLs, point loads and the two end
        anchorage moments) ready to pass to ``set_loads``/``add`` and analyse.
    """
    beam = model.beam if hasattr(model, "beam") else model
    R = list(beam.restraints)
    L = _span_lengths(model)
    n = len(L)
    if len(profiles) != n:
        raise ValueError(f"expected {n} profiles (one per span), got {len(profiles)}")
    F = [float(force)] * n if np.isscalar(force) else [float(f) for f in force]
    if len(F) != n:
        raise ValueError(f"expected {n} force values, got {len(F)}")

    LM: LoadMatrix = []
    first_e: Optional[float] = None
    first_F = 0.0
    last_e: Optional[float] = None
    last_F = 0.0

    for i, prof in enumerate(profiles):
        if prof is None:
            continue
        span = i + 1
        Li, Fi = L[i], F[i]
        # a span with a free end node is a cantilever (tendon anchored at the tip)
        if R[2 * i] == 0 and R[2 * i + 1] == 0:
            cant = "left"
        elif R[2 * (i + 1)] == 0 and R[2 * (i + 1) + 1] == 0:
            cant = "right"
        else:
            cant = None
        segs = prof.segments(Li, cant)
        # distributed loads from curvature; point loads from interior kinks
        prev_end_slope = None
        for k, s in enumerate(segs):
            if abs(s.epp) > _TOL:
                w = Fi * s.epp  # +down (pycba); upward tendon load is negative
                if s.x1 <= _TOL and s.x2 >= Li - _TOL:
                    LM.append([span, 1, w])  # full-span UDL
                else:
                    LM.append([span, 3, w, s.x1, s.length])  # partial UDL
            # interior kink between previous segment end and this segment start
            if prev_end_slope is not None:
                d_slope = s.slope_start() - prev_end_slope
                if abs(d_slope) > _TOL and s.x1 > _TOL and s.x1 < Li - _TOL:
                    P = Fi * d_slope  # +down
                    LM.append([span, 2, P, s.x1])
            prev_end_slope = s.slope_end()

        # remember tendon eccentricity AND slope at the two ends of the beam
        if first_e is None:
            first_e, first_F = segs[0].e1, Fi
            first_slope = segs[0].slope_start()
        last_e, last_F = segs[-1].e2, Fi
        last_slope = segs[-1].slope_end()

    # Anchorage actions  M = F e (couple) and V = F e' (vertical).  At the left
    # end +F e; at the right end -F e, so the moment pair is self-equilibrating
    # (a constant eccentricity gives the correct uniform -F e).  The vertical
    # anchorage force is reacted directly by any support there, so it is applied
    # only at a free tip (where it is not absorbed).
    left_free = R[0] == 0 and R[1] == 0
    right_free = R[2 * n] == 0 and R[2 * n + 1] == 0
    if first_e is not None and abs(first_e) > _TOL:
        LM.append([1, 4, first_F * first_e, 0.0])
    if last_e is not None and abs(last_e) > _TOL:
        LM.append([n, 4, -last_F * last_e, L[-1]])
    if left_free and abs(first_slope) > _TOL:
        LM.append([1, 2, first_F * first_slope, 0.0])
    if right_free and abs(last_slope) > _TOL:
        LM.append([n, 2, -last_F * last_slope, L[-1]])

    return LM


def _eval_e(segs: List[_Seg], x: float) -> float:
    """Tendon eccentricity at span-local ``x`` from its segments."""
    for s in segs:
        if s.x1 - _TOL <= x <= s.x2 + _TOL:
            return s.e1 + s.slope_start() * (x - s.x1) + 0.5 * s.epp * (x - s.x1) ** 2
    return 0.0


def _cantilever_side(R, i, n):
    if R[2 * i] == 0 and R[2 * i + 1] == 0:
        return "left"
    if R[2 * (i + 1)] == 0 and R[2 * (i + 1) + 1] == 0:
        return "right"
    return None


def plot_tendon(
    model,
    force: Union[float, Sequence[float]],
    profiles: Sequence[Optional[Profile]],
    *,
    units=None,
    color: str = "tab:red",
    show: bool = False,
):
    """
    Draw the tendon and its equivalent loads in three stacked, x-aligned panels.

    (a) the beam with its supports (no loads); (b) the specified cable drape -
    the tendon eccentricity profile, on its own (exaggerated) vertical scale,
    measured positive below the centroid; and (c) the equivalent ("balanced")
    loads the tendon exerts, drawn on the bare beam (no support symbols).  Read
    top to bottom, this shows how the drape becomes the balancing loads.

    Parameters
    ----------
    model : pycba.Beam or pycba.BeamAnalysis
        Supplies the geometry.
    force, profiles
        As for :func:`equivalent_loads`.
    units : str or pycba.units.UnitSystem, optional
        Display unit system for the load labels and the length axis.
    color : str
        Colour for the load arrows/labels.
    show : bool
        Call ``matplotlib.pyplot.show()`` before returning.

    Returns
    -------
    matplotlib.figure.Figure, tuple(matplotlib.axes.Axes)
        The figure and its three axes (beam, drape, loads).
    """
    import matplotlib.pyplot as plt

    from .render import BeamPlotter
    from .units import resolve

    us = resolve(units)
    beam = model.beam if hasattr(model, "beam") else model
    Ls = list(beam.mbr_lengths)
    R = list(beam.restraints)
    n = len(Ls)
    offs = np.concatenate([[0.0], np.cumsum(Ls)])
    total = float(offs[-1])
    LM = equivalent_loads(model, force, profiles)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(9, 6.6),
        gridspec_kw={"height_ratios": [0.6, 1.3, 0.85], "hspace": 0.35},
    )

    def _strip_xaxis(ax):  # no x ticks / no bottom spine; the vertical grid stays
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)

    # (a) beam + supports, no loads -- no x-axis, keep the vertical grid
    BeamPlotter(beam, []).render_mpl(
        ax=ax_a, equal_aspect=False, dimensions=False, units=us
    )
    ax_a.set_xlabel("")
    ax_a.set_title("Beam")
    y0, y1 = ax_a.get_ylim()  # trim the empty height above the beam (keep supports)
    ax_a.set_ylim(y0, 0.35 * y1)
    ax_a.grid(True, axis="x", ls=":", alpha=0.4)
    _strip_xaxis(ax_a)

    # (b) the cable drape e(x), exaggerated (its own y-scale), + below centroid
    ax_b.plot([0, total], [0, 0], "k-", lw=2, zorder=3)  # the beam (centroid)
    for i in range(n + 1):
        if not (R[2 * i] == 0 and R[2 * i + 1] == 0):  # a supported node
            ax_b.plot([offs[i]], [0], marker="^", color="0.35", ms=8, zorder=4)
    for i, prof in enumerate(profiles):
        if prof is None:
            continue
        segs = prof.segments(Ls[i], _cantilever_side(R, i, n))
        xs = np.linspace(0.0, Ls[i], 80)
        es = np.array([_eval_e(segs, x) for x in xs])
        ax_b.plot(offs[i] + xs, es, color="tab:blue", lw=2.0, zorder=5)
    ax_b.axhline(0, color="0.6", ls="--", lw=0.8, zorder=1)
    ax_b.invert_yaxis()  # positive e (below centroid) drawn downward
    lu = f" ({us.length})" if us.length else ""
    ax_b.set_ylabel(f"Tendon eccentricity{lu}\n(+ve below centroid)")
    ax_b.set_xlabel("")
    ax_b.set_title("Cable drape (exaggerated)")
    ax_b.grid(True, axis="x", ls=":", alpha=0.4)
    # keep the y-axis (and its label) but drop the surrounding box and the x-axis
    for sp in ("top", "right", "bottom"):
        ax_b.spines[sp].set_visible(False)
    _strip_xaxis(ax_b)

    # (c) the equivalent loads on the bare beam (no supports) -- keep the x-axis
    BeamPlotter(beam, LM).render_mpl(
        ax=ax_c, equal_aspect=False, show_supports=False, color=color, units=us
    )
    ax_c.set_title("Equivalent (balanced) loads")
    ax_c.grid(True, axis="x", ls=":", alpha=0.4)

    if show:
        plt.show()
    return fig, (ax_a, ax_b, ax_c)
