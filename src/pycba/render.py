"""
PyCBA - Beam & loading schematic rendering.

Two backends draw the *structural model* (the beam, its supports and the applied
loads) directly from a :class:`pycba.beam.Beam`:

* :meth:`BeamPlotter.render_mpl` - a native matplotlib schematic drawn on
  labelled axes (distance along the beam), using structural symbols for the
  supports and arrows for the loads.
* :meth:`BeamPlotter.render_tikz` - a TikZ/``stanli`` ``.tex`` document for
  publication-quality output, optionally compiled to PDF with ``pdflatex`` via
  :meth:`BeamPlotter.save_tikz`.

Both backends share a single inference pass (:class:`BeamPlotter`) that turns the
restraint vector and load matrix into backend-agnostic *support* and *load*
descriptors, so the structural interpretation is written once.

The support symbols follow the usual drafting convention: the first
vertical-restraining support is drawn as a pin and the remainder as rollers
(PyCBA's 2-DOF-per-node model does not distinguish them mechanically, and
:func:`pycba.utils.parse_beam_string` deliberately discards the ``P``/``R``
distinction); a fully-restrained node is an encastre wall; a positive
(spring) stiffness is a spring.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .load import LoadIC, LoadML, LoadPL, LoadPUDL, LoadTrapez, LoadUDL, parse_LM

# Support kinds (backend-agnostic)
PIN = "pin"
ROLLER = "roller"
FIXED = "fixed"
SPRING = "spring"
GUIDED = "guided"
TORSION_SPRING = "torsion_spring"


@dataclass
class Support:
    """A support inferred from the restraint vector, in global coordinates."""

    x: float
    kind: str
    node: int
    at_left_end: bool = False
    at_right_end: bool = False


@dataclass
class DistLoad:
    """A (possibly trapezoidal / partial) distributed load in global coords."""

    i_span: int
    x0: float
    x1: float
    w0: float
    w1: float


@dataclass
class PointLoad:
    """A concentrated transverse load (``+`` is downward) in global coords."""

    i_span: int
    x: float
    P: float


@dataclass
class MomentLoad:
    """A concentrated moment (``+`` is counter-clockwise) in global coords."""

    i_span: int
    x: float
    M: float


@dataclass
class Hinge:
    """An internal moment release at a node, in global coordinates."""

    x: float
    node: int


class BeamPlotter:
    """
    Render a :class:`pycba.beam.Beam` as a structural schematic.

    The beam *structure* (geometry, supports, internal hinges) is always drawn
    from the beam itself.  The *loads* are an independent layer whose source is
    chosen with ``loads``:

    * ``None`` (default) - use the beam's own load matrix (``beam.LM``), which
      may be empty, in which case only the structure is drawn.
    * an empty list ``[]`` - draw the bare structure, ignoring ``beam.LM``.
    * a low-level load matrix (list of PyCBA load rows).
    * a high-level :class:`pycba.load_cases.LoadCase`.
    * a high-level :class:`pycba.load_cases.LoadCombination` - pass its
      :class:`pycba.load_cases.LoadCases` collection via ``load_cases`` (or convert
      it first with ``combination.to_load_case(load_cases)``).

    Parameters
    ----------
    beam : pycba.beam.Beam
        The beam to render.  Only its geometry, restraints, element types and
        (optionally) load matrix are used, so the beam need not have been
        analysed.
    loads : list | LoadCase | LoadCombination, optional
        The load source to draw (see above).
    load_cases : pycba.load_cases.LoadCases, optional
        Required only when ``loads`` is a :class:`LoadCombination`.
    """

    def __init__(self, beam, loads=None, *, load_cases=None):
        self.beam = beam
        self.L = float(beam.length)
        self.node_x = [float(x) for x in beam._terminal_coords]
        self.LM = self._resolve_lm(loads, load_cases)
        self.supports = self._infer_supports()
        (
            self.dist_loads,
            self.point_loads,
            self.moment_loads,
            self.skipped_loads,
        ) = self._infer_loads()
        self.hinges = self._infer_hinges()

    def _resolve_lm(self, loads, load_cases):
        """Resolve the ``loads`` argument to a plain PyCBA load matrix.

        Accepts a raw load matrix, a :class:`~pycba.load_cases.LoadCase`, or a
        :class:`~pycba.load_cases.LoadCombination` (with its ``load_cases``).  The
        high-level classes are resolved by duck typing so this renderer does
        not hard-depend on their presence in :mod:`pycba.load_cases`.
        """
        if loads is None:
            return self.beam.LM or []

        to_LM = getattr(loads, "to_LM", None)
        if callable(to_LM):
            # A LoadCombination needs the surrounding LoadCases to resolve its
            # factors; an ordinary LoadCase resolves on its own.  Distinguish by
            # the to_LM() signature rather than importing the concrete classes.
            import inspect

            needs_cases = "load_cases" in inspect.signature(to_LM).parameters
            if needs_cases:
                if load_cases is None:
                    raise ValueError(
                        "Plotting a LoadCombination requires its LoadCases "
                        "collection; pass load_cases=..., or convert it first "
                        "with combination.to_load_case(load_cases) or "
                        "combination.to_LM(load_cases)."
                    )
                return to_LM(load_cases)
            return to_LM()
        return list(loads)

    # ------------------------------------------------------------------ #
    # Inference (shared by both backends)
    # ------------------------------------------------------------------ #
    def _infer_supports(self) -> List[Support]:
        """Map the restraint vector to a list of :class:`Support` descriptors."""
        R = list(self.beam.restraints)
        n_nodes = len(self.node_x)
        # A fully-fixed (encastre) node provides horizontal restraint, so when one
        # exists no separate pin is needed - every vertical-only support is a roller.
        has_fixed = any(R[2 * i] == -1 and R[2 * i + 1] == -1 for i in range(n_nodes))
        pin_assigned = has_fixed

        supports: List[Support] = []
        for i in range(n_nodes):
            v = R[2 * i]
            r = R[2 * i + 1]
            kind: Optional[str] = None
            if v == -1 and r == -1:
                kind = FIXED
            elif v == -1:  # vertical translation restrained, rotation free or sprung
                if pin_assigned:
                    kind = ROLLER
                else:
                    kind = PIN
                    pin_assigned = True
            elif v > 0:  # vertical spring
                kind = SPRING
            elif v == 0 and r == -1:  # rotation restrained, translation free
                kind = GUIDED
            elif v == 0 and r > 0:  # rotational spring only
                kind = TORSION_SPRING
            # v == 0 and r == 0  -> free node (no support): nothing to draw
            if kind is not None:
                supports.append(
                    Support(
                        x=self.node_x[i],
                        kind=kind,
                        node=i,
                        at_left_end=(i == 0),
                        at_right_end=(i == n_nodes - 1),
                    )
                )
        return supports

    def _infer_loads(
        self,
    ) -> Tuple[List[DistLoad], List[PointLoad], List[MomentLoad], list]:
        """Normalise the load matrix into global-coordinate descriptors."""
        dist: List[DistLoad] = []
        points: List[PointLoad] = []
        moments: List[MomentLoad] = []
        skipped: list = []

        loads = parse_LM(self.LM) if self.LM else []
        for ld in loads:
            off = self.node_x[ld.i_span]
            span_len = float(self.beam.mbr_lengths[ld.i_span])
            if isinstance(ld, LoadUDL):
                dist.append(DistLoad(ld.i_span, off, off + span_len, ld.w, ld.w))
            elif isinstance(ld, LoadPUDL):
                dist.append(
                    DistLoad(ld.i_span, off + ld.a, off + ld.a + ld.c, ld.w, ld.w)
                )
            elif isinstance(ld, LoadTrapez):
                w1, w2, _dw, a, c = ld._resolve(span_len)
                if c > 0:
                    dist.append(DistLoad(ld.i_span, off + a, off + a + c, w1, w2))
            elif isinstance(ld, LoadPL):
                points.append(PointLoad(ld.i_span, off + ld.a, ld.P))
            elif isinstance(ld, LoadML):
                moments.append(MomentLoad(ld.i_span, off + ld.a, ld.M))
            elif isinstance(ld, LoadIC):
                # Imposed-curvature loads have no clean schematic glyph.
                skipped.append(ld)
        return dist, points, moments, skipped

    def _infer_hinges(self) -> List[Hinge]:
        """Find internal moment releases from the per-span element types.

        Element types: 1=FF, 2=FP (release at right end), 3=PF (release at left
        end), 4=PP (both ends released).
        """
        et = list(self.beam.mbr_eletype)
        hinges: List[Hinge] = []
        for i in range(1, len(self.node_x) - 1):
            left_span, right_span = i - 1, i  # spans meeting at node i
            released = (left_span < len(et) and et[left_span] in (2, 4)) or (
                right_span < len(et) and et[right_span] in (3, 4)
            )
            if released:
                hinges.append(Hinge(self.node_x[i], i))
        return hinges

    def _is_partial_dist(self, d: DistLoad) -> bool:
        """True if a distributed load covers less than its full span (so its
        extent is worth dimensioning)."""
        eps = 1e-6 * max(self.L, 1.0)
        span_start = self.node_x[d.i_span]
        span_end = self.node_x[d.i_span + 1]
        return d.x0 > span_start + eps or d.x1 < span_end - eps

    # ------------------------------------------------------------------ #
    # matplotlib backend
    # ------------------------------------------------------------------ #
    def render_mpl(
        self,
        ax=None,
        *,
        dimensions: bool = False,
        labels: bool = True,
        load_values: bool = True,
        color: str = "tab:red",
        equal_aspect: bool = True,
        units=None,
        show_supports: bool = True,
        figsize=(10, 3.2),
    ):
        """
        Draw the beam schematic with matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure/axes is created if omitted.
        dimensions : bool
            Draw span-length dimension lines below the beam.  Off by default
            because the x-axis already shows distance along the beam (the
            extent of a partial distributed load is still dimensioned with
            ``load_values``).
        labels : bool
            Label the support nodes ``A``, ``B``, ...
        load_values : bool
            Annotate the load magnitudes (and the extent of any partial
            distributed load).
        color : str
            Colour used for the load arrows/labels.
        equal_aspect : bool
            Keep an equal data aspect ratio (the default) so the support and
            load glyphs are drawn true to shape.  Set ``False`` to let the
            schematic stretch to fill its axes — used when embedding the
            schematic as a strip above the result diagrams (see
            :meth:`pycba.analysis.BeamAnalysis.plot_results`).

        Returns
        -------
        matplotlib.axes.Axes
            The axes the schematic was drawn into.
        """
        import matplotlib.pyplot as plt
        from .units import resolve

        self._us = resolve(units)
        L = self.L
        if L <= 0:
            raise ValueError("Cannot render a beam of zero length")
        if ax is None:
            _fig, ax = plt.subplots(figsize=figsize)

        sh = 0.05 * L  # support symbol unit height

        # The beam itself: a depth-varying grey elevation for a non-prismatic
        # member (depth proportional to EI**(1/3), the equivalent rectangular
        # section depth), otherwise the usual flat line.
        prof = self._ei_depth_profile(sh)
        if prof is not None:
            xb, hb = prof
            ax.fill_between(xb, -hb, hb, facecolor="0.82", edgecolor="none", zorder=2)
            ax.plot(xb, hb, "k-", lw=1.5, zorder=3)
            ax.plot(xb, -hb, "k-", lw=1.5, zorder=3)
            ax.plot([0, 0], [-hb[0], hb[0]], "k-", lw=1.5, zorder=3)
            ax.plot([L, L], [-hb[-1], hb[-1]], "k-", lw=1.5, zorder=3)
        else:
            ax.plot([0, L], [0, 0], "k-", lw=3, zorder=5, solid_capstyle="round")

        if show_supports:
            self._draw_foundations_mpl(ax, sh)
            for s in self.supports:
                self._draw_support_mpl(ax, s, sh)
            for h in self.hinges:
                self._draw_hinge_mpl(ax, h, sh)

        # Loads (scaled within each family so the figure stays balanced)
        wmax = max((max(abs(d.w0), abs(d.w1)) for d in self.dist_loads), default=0.0)
        pmax = max((abs(p.P) for p in self.point_loads), default=0.0)
        for d in self.dist_loads:
            self._draw_dist_mpl(ax, d, wmax, color, load_values)
        for p in self.point_loads:
            self._draw_point_mpl(ax, p, pmax, sh, color, load_values)
        for m in self.moment_loads:
            self._draw_moment_mpl(ax, m, 0.045 * L, sh, color, load_values)

        if labels and show_supports:
            # Node letters sit just below the beam and offset to one side, so
            # they clear the loads (above) and the support symbol (directly
            # below): to the right for the right-hand end, to the left otherwise.
            for s in self.supports:
                right = s.at_right_end
                ax.text(
                    s.x + (0.8 * sh if right else -0.8 * sh),
                    -0.55 * sh,
                    chr(ord("A") + s.node),
                    ha="left" if right else "right",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    zorder=8,
                )

        if dimensions:
            self._draw_dimensions_mpl(ax, sh)
        # The extent of a partial distributed load is a load annotation, not a
        # span dimension, so it follows ``load_values`` and shows even when the
        # span dimensions are off.
        partial = [d for d in self.dist_loads if self._is_partial_dist(d)]
        if load_values:
            for d in partial:
                self._draw_load_extent_mpl(ax, d, sh)

        # Axes cosmetics: labelled x, no meaningful y.  Reserve vertical room on
        # the side each glyph is actually drawn (positive loads above the beam,
        # negative below), so upward loads do not collide with the x-axis; the
        # magnitude labels sit a little beyond each glyph tip.
        above = [0.0]
        below = [0.0]
        for p in self.point_loads:
            ln = 0.16 * L * (0.55 + 0.45 * abs(p.P) / pmax if pmax else 1.0)
            (above if p.P >= 0 else below).append(ln)
        for d in self.dist_loads:
            if d.w0 >= 0 or d.w1 >= 0:
                above.append(0.07 * L)
            if d.w0 < 0 or d.w1 < 0:
                below.append(0.07 * L)
        for _m in self.moment_loads:  # the moment arc straddles the beam
            above.append(0.045 * L)
            below.append(0.045 * L)
        any_loads = self.point_loads or self.dist_loads or self.moment_loads
        lbl = 1.4 * sh  # room beyond a glyph tip for its magnitude label
        ymax = max(above) + (lbl if any_loads else 0.9 * sh)
        if not show_supports:
            sup = -0.3 * sh  # no support glyphs hang below the beam
        elif dimensions:
            sup = -3.0 * sh
        elif load_values and partial:
            sup = -2.2 * sh
        else:
            sup = -1.6 * sh
        ymin = min(sup, -(max(below) + lbl)) if max(below) > 1e-12 else sup

        ax.set_xlim(-0.06 * L, 1.06 * L)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box" if equal_aspect else "datalim")
        if not equal_aspect:
            ax.set_aspect("auto")
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel(self._us.distance_axis)
        ax.grid(True, axis="x", ls=":", alpha=0.4)
        return ax

    def render_reactions_mpl(
        self, ax, vert, mom, sh=None, color="tab:green", show_val=True
    ):
        """
        Overlay support-reaction arrows on a schematic already drawn by
        :meth:`render_mpl` (which must have been called first, so ``self._us``
        and the axis limits are set).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The schematic axes to draw onto.
        vert : dict[int, float]
            Node index -> vertical reaction (positive **upward**); drawn as a
            straight arrow pointing in the force direction.
        mom : dict[int, float]
            Node index -> moment reaction (positive **counter-clockwise**);
            drawn as a curved arrow at the support.
        sh : float, optional
            The schematic unit height; defaults to ``0.05 * L``.
        color : str
            Colour for the reaction arrows and labels.
        show_val : bool
            Annotate each reaction with its magnitude.
        """
        if sh is None:
            sh = 0.05 * self.L
        fvmax = max((abs(v) for v in vert.values()), default=0.0)
        lmax = 2.2 * sh
        for node, Rv in vert.items():
            x = self.node_x[node]
            length = sh * (2.2 + 1.3 * (abs(Rv) / fvmax if fvmax else 1.0))
            lmax = max(lmax, length)
            # Arrow points in the force direction with its head at the beam for
            # an upward reaction (tail below), and below the beam for a
            # downward one - in both cases the label sits below the arrow.
            if Rv >= 0:
                xy, xytext = (x, 0.0), (x, -length)
            else:
                xy, xytext = (x, -length), (x, 0.0)
            ax.annotate(
                "",
                xy=xy,
                xytext=xytext,
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16
                ),
                zorder=10,
            )
            if show_val:
                ax.text(
                    x,
                    -length - 0.3 * sh,
                    self._us.fmt_force(abs(Rv)),
                    ha="center",
                    va="top",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    zorder=11,
                )
        for node, Mr in mom.items():
            ml = MomentLoad(i_span=0, x=self.node_x[node], M=Mr)
            self._draw_moment_mpl(ax, ml, 1.2 * sh, sh, color, show_val)
        # Make sure the arrows and their labels are within the view.
        ymin, ymax = ax.get_ylim()
        need = -(lmax + 1.0 * sh)
        if ymin > need:
            ax.set_ylim(need, ymax)
        return ax

    def _ei_depth_profile(self, sh: float):
        """
        Half-depth profile ``(x, h)`` for a non-prismatic beam, or ``None``.

        Depth is taken proportional to ``EI(x) ** (1/3)`` (the depth of an
        equivalent rectangular section), normalised so the stiffest section is
        ``~1.2 sh`` deep.  Returns ``None`` for a uniform beam so it keeps the
        plain centre-line.
        """
        from .section import SectionEI

        eis = getattr(self.beam, "mbr_EIs", None)
        if not eis:
            return None
        Ls = self.beam.mbr_lengths
        offs = np.concatenate([[0.0], np.cumsum(Ls)])
        xs, vals = [], []
        for i, ei in enumerate(eis):
            if isinstance(ei, SectionEI):
                xl = np.linspace(0.0, Ls[i], 41)
                v = np.asarray(ei(xl), dtype=float)
            else:
                xl = np.array([0.0, Ls[i]])
                v = np.array([float(ei), float(ei)])
            xs.append(offs[i] + xl)
            vals.append(v)
        xs = np.concatenate(xs)
        vals = np.concatenate(vals)
        if vals.max() <= 0.0 or (vals.max() - vals.min()) <= 1e-6 * vals.max():
            return None  # uniform -> flat line
        h = 0.6 * sh * (vals / vals.max()) ** (1.0 / 3.0)
        return xs, h

    # --- matplotlib support glyphs ------------------------------------- #
    def _draw_support_mpl(self, ax, s: Support, sh: float):
        from matplotlib.patches import Circle, Polygon

        x = s.x
        if s.kind in (PIN, ROLLER):
            end = s.at_left_end or s.at_right_end
            # Internal supports are seated a touch below the beam so the pin
            # half-disc marker reads clearly below the beam line; end supports
            # sit at the beam with a full-circle pin straddling it.
            top = 0.0 if end else -0.20 * sh
            if s.kind == PIN:
                ax.add_patch(
                    Polygon(
                        [[x, top], [x - 0.5 * sh, top - sh], [x + 0.5 * sh, top - sh]],
                        closed=True,
                        fc="white",
                        ec="k",
                        lw=1.3,
                        zorder=4,
                    )
                )
                self._ground_mpl(ax, x, top - sh, 1.5 * sh)
            else:  # ROLLER
                ax.add_patch(
                    Polygon(
                        [
                            [x, top],
                            [x - 0.5 * sh, top - 0.7 * sh],
                            [x + 0.5 * sh, top - 0.7 * sh],
                        ],
                        closed=True,
                        fc="white",
                        ec="k",
                        lw=1.3,
                        zorder=4,
                    )
                )
                for dx in (-0.27 * sh, 0.27 * sh):
                    ax.add_patch(
                        Circle(
                            (x + dx, top - 0.86 * sh),
                            0.15 * sh,
                            fc="white",
                            ec="k",
                            lw=1.0,
                            zorder=4,
                        )
                    )
                self._ground_mpl(ax, x, top - 1.02 * sh, 1.5 * sh)
            self._pin_marker_mpl(ax, x, sh, end)
        elif s.kind == FIXED:
            side = 1 if s.at_right_end else -1
            ax.plot([x, x], [-sh, sh], "k-", lw=2, zorder=4)
            n = 6
            ys = np.linspace(-sh, sh, n)
            for yi in ys:
                ax.plot(
                    [x, x + side * 0.4 * sh],
                    [yi, yi + 0.4 * sh],
                    "k-",
                    lw=0.8,
                    zorder=3,
                )
        elif s.kind in (SPRING, TORSION_SPRING):
            self._spring_mpl(ax, x, sh)
            self._ground_mpl(ax, x, -1.2 * sh, 1.3 * sh)
        elif s.kind == GUIDED:
            ax.plot(
                [x - 0.5 * sh, x + 0.5 * sh],
                [-0.3 * sh, -0.3 * sh],
                "k-",
                lw=1.2,
                zorder=4,
            )
            ax.plot(
                [x - 0.5 * sh, x + 0.5 * sh],
                [-0.6 * sh, -0.6 * sh],
                "k-",
                lw=1.2,
                zorder=4,
            )
            self._ground_mpl(ax, x, -0.6 * sh, 1.3 * sh)

    def _pin_marker_mpl(self, ax, x: float, sh: float, end: bool):
        """Little pin at a pin/roller support: full circle at an end, a
        half-disc below the beam (seated on the support) at an internal one."""
        from matplotlib.patches import Circle, Wedge

        r = 0.17 * sh
        if end:
            ax.add_patch(Circle((x, 0), r, fc="white", ec="k", lw=1.1, zorder=7))
        else:
            ax.add_patch(
                Wedge((x, 0), r, 180, 360, fc="white", ec="k", lw=1.1, zorder=7)
            )

    def _draw_foundations_mpl(self, ax, sh: float):
        """
        Draw a Winkler-foundation indication - a row of springs standing on a
        hatched ground line - under any member that has an elastic foundation
        (a finite ``kf``).
        """
        mbr_kf = getattr(self.beam, "mbr_kf", None)
        if not mbr_kf:
            return
        yg = -1.25 * sh  # ground line, just below the spring feet
        hs = 0.16 * sh
        for i, kf in enumerate(mbr_kf):
            if kf is None:
                continue
            x0, x1 = self.node_x[i], self.node_x[i + 1]
            n = int(np.clip(round((x1 - x0) / (1.4 * sh)), 3, 18))
            for xc in np.linspace(x0, x1, n):
                self._spring_mpl(ax, xc, sh)
            ax.plot([x0, x1], [yg, yg], "k-", lw=1.0, zorder=3)
            nh = int(np.clip(round((x1 - x0) / (0.45 * sh)), 6, 80))
            for xi in np.linspace(x0, x1 - (x1 - x0) / nh, nh):
                ax.plot([xi, xi + hs], [yg, yg - hs], "k-", lw=0.6, zorder=3)

    def _ground_mpl(self, ax, xc: float, ytop: float, width: float, n: int = 6):
        """A hatched ground line centred at ``xc`` with its top at ``ytop``."""
        ax.plot([xc - width / 2, xc + width / 2], [ytop, ytop], "k-", lw=1.0, zorder=3)
        hs = width * 0.16
        for xi in np.linspace(xc - width / 2, xc + width / 2 - width / n, n):
            ax.plot([xi, xi + hs], [ytop, ytop - hs], "k-", lw=0.7, zorder=3)

    def _spring_mpl(self, ax, xc: float, sh: float):
        """A vertical zig-zag spring from the beam down to a short stem."""
        n = 6
        ys = np.linspace(0, -0.95 * sh, 2 * n + 1)
        xs = []
        for k in range(len(ys)):
            if k == 0 or k == len(ys) - 1:
                xs.append(xc)
            else:
                xs.append(xc + (0.22 * sh if k % 2 else -0.22 * sh))
        ax.plot(xs, ys, "k-", lw=1.0, zorder=4)
        ax.plot([xc, xc], [-0.95 * sh, -1.2 * sh], "k-", lw=1.0, zorder=4)

    def _draw_hinge_mpl(self, ax, h: Hinge, sh: float):
        from matplotlib.patches import Circle

        ax.add_patch(Circle((h.x, 0), 0.18 * sh, fc="white", ec="k", lw=1.2, zorder=6))

    # --- matplotlib load glyphs ---------------------------------------- #
    def _draw_point_mpl(self, ax, p: PointLoad, pmax: float, sh, color, show_val):
        length = 0.16 * self.L * (0.55 + 0.45 * abs(p.P) / pmax if pmax else 1.0)
        x = p.x
        tail_y = length if p.P >= 0 else -length
        ax.annotate(
            "",
            xy=(x, 0),
            xytext=(x, tail_y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=14),
            zorder=6,
        )
        if show_val:
            ax.text(
                x,
                tail_y + (0.3 * sh if p.P >= 0 else -0.3 * sh),
                self._us.fmt_force(abs(p.P)),
                ha="center",
                va="bottom" if p.P >= 0 else "top",
                color=color,
                fontsize=8,
                zorder=7,
            )

    def _draw_dist_mpl(self, ax, d: DistLoad, wmax: float, color, show_val):
        from matplotlib.patches import Polygon

        w_h = 0.07 * self.L

        def height(w):
            if not wmax:
                return 0.0
            return np.sign(w) * w_h * (0.35 + 0.65 * abs(w) / wmax)

        x0, x1 = d.x0, d.x1
        h0, h1 = height(d.w0), height(d.w1)
        ax.plot([x0, x1], [h0, h1], color=color, lw=1.5, zorder=4)
        ax.add_patch(
            Polygon(
                [[x0, 0], [x0, h0], [x1, h1], [x1, 0]],
                closed=True,
                fc=color,
                ec="none",
                alpha=0.12,
                zorder=2,
            )
        )
        n = max(2, int((x1 - x0) / (0.06 * self.L))) if x1 > x0 else 2
        for xi in np.linspace(x0, x1, n + 1):
            hi = h0 if x1 == x0 else h0 + (h1 - h0) * (xi - x0) / (x1 - x0)
            if abs(hi) < 1e-9:
                continue
            ax.annotate(
                "",
                xy=(xi, 0),
                xytext=(xi, hi),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=1.1, mutation_scale=9
                ),
                zorder=4,
            )
        if show_val:
            same = abs(d.w0 - d.w1) < 1e-9
            lbl = (
                self._us.fmt_distributed(abs(d.w0))
                if same
                else self._us.fmt_distributed(abs(d.w0), abs(d.w1))
            )
            ax.text(
                0.5 * (x0 + x1),
                max(h0, h1) + 0.25 * w_h,
                lbl,
                ha="center",
                va="bottom",
                color=color,
                fontsize=8,
                zorder=7,
            )

    def _draw_moment_mpl(self, ax, m: MomentLoad, r: float, sh, color, show_val):
        from matplotlib.patches import Arc, Polygon

        x = m.x
        # Tick marking the point of application on the beam.
        ax.plot([x, x], [-0.22 * sh, 0.22 * sh], color=color, lw=1.5, zorder=6)
        th1, th2 = -50.0, 230.0
        ax.add_patch(
            Arc(
                (x, 0),
                2 * r,
                2 * r,
                angle=0,
                theta1=th1,
                theta2=th2,
                color=color,
                lw=2,
                zorder=6,
            )
        )
        ccw = m.M >= 0
        a = np.radians(th2 if ccw else th1)
        tip = (x + r * np.cos(a), r * np.sin(a))
        tang = a + (np.pi / 2 if ccw else -np.pi / 2)
        # arrowhead triangle
        back = tang + np.pi
        size = 0.5 * sh
        p2 = (
            tip[0] + size * np.cos(back + np.radians(22)),
            tip[1] + size * np.sin(back + np.radians(22)),
        )
        p3 = (
            tip[0] + size * np.cos(back - np.radians(22)),
            tip[1] + size * np.sin(back - np.radians(22)),
        )
        ax.add_patch(Polygon([tip, p2, p3], closed=True, fc=color, ec=color, zorder=6))
        if show_val:
            ax.text(
                x,
                r + 0.5 * sh,
                self._us.fmt_moment(abs(m.M)),
                ha="center",
                va="bottom",
                color=color,
                fontsize=8,
                zorder=7,
            )

    def _draw_dimensions_mpl(self, ax, sh: float):
        yd = -2.3 * sh
        for i in range(self.beam.no_spans):
            x0, x1 = self.node_x[i], self.node_x[i + 1]
            ax.annotate(
                "",
                xy=(x0, yd),
                xytext=(x1, yd),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1),
            )
            for xb in (x0, x1):
                ax.plot([xb, xb], [yd + 0.3 * sh, yd - 0.3 * sh], color="0.35", lw=0.6)
            ax.text(
                0.5 * (x0 + x1),
                yd - 0.15 * sh,
                f"{x1 - x0:g} m",
                ha="center",
                va="top",
                color="0.35",
                fontsize=8,
            )

    def _draw_load_extent_mpl(self, ax, d: DistLoad, sh: float):
        """Dimension the loaded length of a partial distributed load, on a
        stacked dimension row just above the span-length dimensions."""
        yd = -1.55 * sh
        x0, x1 = d.x0, d.x1
        ax.annotate(
            "",
            xy=(x0, yd),
            xytext=(x1, yd),
            arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.8),
        )
        for xb in (x0, x1):
            ax.plot([xb, xb], [yd + 0.22 * sh, yd - 0.22 * sh], color="0.45", lw=0.6)
        ax.text(
            0.5 * (x0 + x1),
            yd - 0.12 * sh,
            f"{x1 - x0:g} m",
            ha="center",
            va="top",
            color="0.45",
            fontsize=7.5,
        )

    # ------------------------------------------------------------------ #
    # TikZ / stanli backend
    # ------------------------------------------------------------------ #
    def render_tikz(
        self,
        *,
        standalone: bool = True,
        scale: Optional[float] = None,
        dimensions: bool = True,
        labels: bool = True,
        load_values: bool = True,
        units=None,
    ) -> str:
        """
        Generate a TikZ/``stanli`` representation of the beam.

        Parameters
        ----------
        standalone : bool
            If ``True`` (default) return a complete compilable document; if
            ``False`` return only the ``tikzpicture`` environment for embedding.
        scale : float, optional
            Emit a ``stanli`` ``\\scaling`` factor.
        dimensions, labels, load_values : bool
            Toggle span dimensions, node labels and load-magnitude annotations.
        units : str or pycba.units.UnitSystem, optional
            Display unit system for the load and dimension labels.  Defaults to
            the global default (see :func:`pycba.set_units`).

        Returns
        -------
        str
            The LaTeX source.
        """
        from .units import resolve

        self._us = resolve(units)
        nodes = [self._node_name(i) for i in range(len(self.node_x))]
        lines: List[str] = []
        lines.append(
            "\\begin{tikzpicture}[background rectangle/.style={fill=white}, "
            "show background rectangle]"
        )
        if scale:
            lines.append(f"\t\\scaling{{{scale:g}}};")

        lines.append("\t% Nodes")
        for i, x in enumerate(self.node_x):
            lines.append(f"\t\\point{{{nodes[i]}}}{{{x:g}}}{{0}};")

        lines.append("\t% Beam")
        for i in range(self.beam.no_spans):
            lines.append(f"\t\\beam{{4}}{{{nodes[i]}}}{{{nodes[i + 1]}}};")

        lines.append("\t% Supports")
        for s in self.supports:
            lines.append("\t" + self._support_tikz(s, nodes))

        hinge_lines = self._hinges_tikz(nodes)
        if hinge_lines:
            lines.append("\t% Hinges")
            lines.extend("\t" + h for h in hinge_lines)

        load_lines = self._loads_tikz(nodes, load_values)
        if load_lines:
            lines.append("\t% Loads")
            lines.extend("\t" + l for l in load_lines)

        if dimensions:
            lines.append("\t% Dimensions")
            dist = -max(1.0, 0.09 * self.L)
            for i in range(self.beam.no_spans):
                span = self.node_x[i + 1] - self.node_x[i]
                length_u = f"~{self._us.length}" if self._us.length else ""
                lines.append(
                    f"\t\\dimensioning{{1}}{{{nodes[i]}}}{{{nodes[i + 1]}}}"
                    f"{{{dist:g}}}[${span:g}${length_u}];"
                )

        if labels:
            lines.append("\t% Labels")
            for s in self.supports:
                # Offset below and to one side so the letter clears the loads
                # (above) and the support symbol (directly below).
                pos = (
                    "below right=1mm and 4mm"
                    if s.at_right_end
                    else "below left=1mm and 4mm"
                )
                letter = chr(ord("A") + s.node)
                lines.append(
                    f"\t\\notation{{1}}{{{nodes[s.node]}}}{{${letter}$}}[{pos}];"
                )

        lines.append("\\end{tikzpicture}")
        body = "\n".join(lines)
        return body if not standalone else self._wrap_standalone(body)

    def save_tikz(
        self, path: Union[str, Path], *, compile: bool = False, **opts
    ) -> Path:
        """
        Write the TikZ/``stanli`` source to ``path`` (and optionally compile it).

        Parameters
        ----------
        path : str or pathlib.Path
            Output path; a ``.tex`` suffix is enforced.
        compile : bool
            If ``True`` run ``pdflatex`` in the file's directory to produce a
            PDF.  Requires a LaTeX installation with the ``stanli`` package.
        **opts
            Forwarded to :meth:`render_tikz` (``standalone`` is forced to
            ``True``).

        Returns
        -------
        pathlib.Path
            The written ``.tex`` path, or the produced ``.pdf`` when
            ``compile`` is ``True``.
        """
        path = Path(path)
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        opts.pop("standalone", None)
        path.write_text(self.render_tikz(standalone=True, **opts))
        if not compile:
            return path

        exe = shutil.which("pdflatex")
        if exe is None:
            raise RuntimeError(
                "pdflatex not found on PATH; cannot compile. Install a LaTeX "
                "distribution that provides the 'stanli' package, or call "
                "save_tikz(..., compile=False) to emit the .tex only."
            )
        proc = subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error", path.name],
            cwd=str(path.parent),
            capture_output=True,
            text=True,
        )
        pdf = path.with_suffix(".pdf")
        if proc.returncode != 0 or not pdf.exists():
            raise RuntimeError(
                "pdflatex failed to compile the figure. Tail of the log:\n"
                + (proc.stdout or "")[-2000:]
            )
        return pdf

    # --- TikZ helpers --------------------------------------------------- #
    @staticmethod
    def _node_name(i: int) -> str:
        return chr(ord("a") + i) if i < 26 else f"p{i}"

    def _support_tikz(self, s: Support, nodes: List[str]) -> str:
        n = nodes[s.node]
        if s.kind == PIN:
            return f"\\support{{1}}{{{n}}};"
        if s.kind == ROLLER:
            return f"\\support{{2ooo}}{{{n}}};"
        if s.kind == FIXED:
            # stanli's fixed support defaults to a horizontal wall (beam
            # vertical); rotate it to a vertical wall at the beam end - the
            # wall faces outward, so the left and right ends differ by 180.
            if s.at_right_end:
                rot = "[90]"
            elif s.at_left_end:
                rot = "[-90]"
            else:
                rot = ""  # interior fixed node: keep the default orientation
            return f"\\support{{4}}{{{n}}}{rot};"
        if s.kind == SPRING:
            return f"\\support{{5}}{{{n}}};"
        if s.kind == GUIDED:
            return f"\\support{{3}}{{{n}}}[90];"
        if s.kind == TORSION_SPRING:
            return f"\\support{{6}}{{{n}}};"
        return ""

    def _hinges_tikz(self, nodes: List[str]) -> List[str]:
        out: List[str] = []
        support_nodes = set()
        # Little pin at every pin/roller support: a full circle at an end
        # support, and a half-circle (clipped to the member side) at an internal
        # support - matching the doc examples (e.g. B in intro_ex_1).
        for s in self.supports:
            if s.kind in (PIN, ROLLER):
                i = s.node
                support_nodes.add(i)
                if s.at_left_end or s.at_right_end:
                    out.append(f"\\hinge{{1}}{{{nodes[i]}}};")
                else:
                    out.append(
                        f"\\hinge{{2}}{{{nodes[i]}}}[{nodes[i - 1]}][{nodes[i + 1]}];"
                    )
        # Free internal moment-release hinges (not at a support) are full circles.
        for h in self.hinges:
            if h.node not in support_nodes:
                out.append(f"\\hinge{{1}}{{{nodes[h.node]}}};")
        return out

    def _span_fraction(self, i_span: int, x: float) -> float:
        off = self.node_x[i_span]
        span_len = self.node_x[i_span + 1] - off
        return (x - off) / span_len if span_len else 0.0

    def _loads_tikz(self, nodes: List[str], show_val: bool) -> List[str]:
        out: List[str] = []

        if self.dist_loads:
            wmax = max(max(abs(d.w0), abs(d.w1)) for d in self.dist_loads) or 1.0
            out.append("\\begin{scope}[color=red]")
            for k, d in enumerate(self.dist_loads):
                ni, nj = nodes[d.i_span], nodes[d.i_span + 1]
                f0 = self._span_fraction(d.i_span, d.x0)
                f1 = self._span_fraction(d.i_span, d.x1)
                out.append(f"\\node (dl{k}a) at ($({ni})!{f0:g}!({nj})$){{}};")
                out.append(f"\\node (dl{k}b) at ($({ni})!{f1:g}!({nj})$){{}};")
                h0 = 0.3 + 0.4 * abs(d.w0) / wmax
                h1 = 0.3 + 0.4 * abs(d.w1) / wmax
                out.append(f"\\lineload{{1}}{{dl{k}a}}{{dl{k}b}}[{h0:g}][{h1:g}];")
            out.append("\\end{scope}")
            if show_val:
                pdist = -max(0.6, 0.055 * self.L)
                du = f" {self._us.distributed}" if self._us.distributed else ""
                lu = f"~{self._us.length}" if self._us.length else ""
                for k, d in enumerate(self.dist_loads):
                    if abs(d.w0 - d.w1) < 1e-9:
                        lbl = f"${abs(d.w0):g}${du}"
                    else:
                        lbl = f"${abs(d.w0):g}\\rightarrow{abs(d.w1):g}${du}"
                    out.append(
                        f"\\notation{{5}}{{dl{k}a}}{{dl{k}b}}[{lbl}][0.5][above=8mm];"
                    )
                    # The extent of a partial distributed load is a load
                    # annotation (shown with the magnitudes), not a span
                    # dimension.
                    if self._is_partial_dist(d):
                        out.append(
                            f"\\dimensioning{{1}}{{dl{k}a}}{{dl{k}b}}"
                            f"{{{pdist:g}}}[${d.x1 - d.x0:g}${lu}];"
                        )

        for k, p in enumerate(self.point_loads):
            ni, nj = nodes[p.i_span], nodes[p.i_span + 1]
            f = self._span_fraction(p.i_span, p.x)
            ang = "90" if p.P >= 0 else "270"
            out.append(f"\\node (pl{k}) at ($({ni})!{f:g}!({nj})$){{}};")
            # 4th arg lengthens the force arrow (default \forceLength is short).
            out.append(
                f"\\begin{{scope}}[color=red]\\load{{1}}{{pl{k}}}[{ang}][18mm]\\end{{scope}}"
            )
            if show_val:
                fu = f" {self._us.force}" if self._us.force else ""
                out.append(
                    f"\\notation{{1}}{{pl{k}}}{{${abs(p.P):g}${fu}}}[above=20mm];"
                )

        for k, m in enumerate(self.moment_loads):
            ni, nj = nodes[m.i_span], nodes[m.i_span + 1]
            f = self._span_fraction(m.i_span, m.x)
            # stanli \load{2} (<-) reads clockwise and \load{3} (->) reads
            # anticlockwise; positive M is anticlockwise (matching the
            # matplotlib backend), so positive -> 3.
            mtype = "3" if m.M >= 0 else "2"
            out.append(f"\\node (ml{k}) at ($({ni})!{f:g}!({nj})$){{}};")
            out.append(
                f"\\begin{{scope}}[color=red]\\load{{{mtype}}}{{ml{k}}}\\end{{scope}}"
            )
            if show_val:
                mu = f" {self._us.moment}" if self._us.moment else ""
                out.append(
                    f"\\notation{{2}}{{ml{k}}}{{${abs(m.M):g}${mu}}}[above=5mm];"
                )
        return out

    @staticmethod
    def _wrap_standalone(body: str) -> str:
        preamble = (
            "\\documentclass[tikz,border=5pt]{standalone}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{stanli}\n"
            "\\usepackage{tikz}\n"
            "\\usetikzlibrary{arrows.meta,arrows,positioning,calc,backgrounds}\n"
            "\\begin{document}\n"
        )
        return preamble + body + "\n\\end{document}\n"
