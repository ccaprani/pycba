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
        has_fixed = any(
            R[2 * i] == -1 and R[2 * i + 1] == -1 for i in range(n_nodes)
        )
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

    # ------------------------------------------------------------------ #
    # matplotlib backend
    # ------------------------------------------------------------------ #
    def render_mpl(
        self,
        ax=None,
        *,
        dimensions: bool = True,
        labels: bool = True,
        load_values: bool = True,
        color: str = "tab:red",
    ):
        """
        Draw the beam schematic with matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure/axes is created if omitted.
        dimensions : bool
            Draw span-length dimension lines below the beam.
        labels : bool
            Label the support nodes ``A``, ``B``, ...
        load_values : bool
            Annotate the load magnitudes.
        color : str
            Colour used for the load arrows/labels.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the schematic was drawn into.
        """
        import matplotlib.pyplot as plt

        L = self.L
        if L <= 0:
            raise ValueError("Cannot render a beam of zero length")
        if ax is None:
            _fig, ax = plt.subplots(figsize=(10, 3.2))

        sh = 0.05 * L  # support symbol unit height

        # The beam itself
        ax.plot(
            [0, L], [0, 0], "k-", lw=3, zorder=5, solid_capstyle="round"
        )

        for s in self.supports:
            self._draw_support_mpl(ax, s, sh)

        for h in self.hinges:
            self._draw_hinge_mpl(ax, h, sh)

        # Loads (scaled within each family so the figure stays balanced)
        wmax = max(
            (max(abs(d.w0), abs(d.w1)) for d in self.dist_loads), default=0.0
        )
        pmax = max((abs(p.P) for p in self.point_loads), default=0.0)
        for d in self.dist_loads:
            self._draw_dist_mpl(ax, d, wmax, color, load_values)
        for p in self.point_loads:
            self._draw_point_mpl(ax, p, pmax, sh, color, load_values)
        for m in self.moment_loads:
            self._draw_moment_mpl(ax, m, 0.06 * L, sh, color, load_values)

        if labels:
            for s in self.supports:
                ax.text(
                    s.x,
                    0.5 * sh,
                    chr(ord("A") + s.node),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    zorder=7,
                )

        if dimensions:
            self._draw_dimensions_mpl(ax, sh)

        # Axes cosmetics: labelled x, no meaningful y
        load_top = [0.0]
        if self.point_loads:
            load_top.append(0.14 * L)
        if self.dist_loads:
            load_top.append(0.10 * L)
        if self.moment_loads:
            load_top.append(0.06 * L)
        any_loads = self.point_loads or self.dist_loads or self.moment_loads
        ymax = max(load_top) + (1.4 * sh if any_loads else 0.9 * sh)
        ymin = -3.0 * sh if dimensions else -1.6 * sh

        ax.set_xlim(-0.06 * L, 1.06 * L)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("Distance along beam (m)")
        ax.grid(True, axis="x", ls=":", alpha=0.4)
        return ax

    # --- matplotlib support glyphs ------------------------------------- #
    def _draw_support_mpl(self, ax, s: Support, sh: float):
        from matplotlib.patches import Circle, Polygon

        x = s.x
        if s.kind == PIN:
            ax.add_patch(
                Polygon(
                    [[x, 0], [x - 0.5 * sh, -sh], [x + 0.5 * sh, -sh]],
                    closed=True,
                    fc="white",
                    ec="k",
                    lw=1.3,
                    zorder=4,
                )
            )
            self._ground_mpl(ax, x, -sh, 1.5 * sh)
        elif s.kind == ROLLER:
            ax.add_patch(
                Polygon(
                    [[x, 0], [x - 0.5 * sh, -0.7 * sh], [x + 0.5 * sh, -0.7 * sh]],
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
                        (x + dx, -0.86 * sh),
                        0.15 * sh,
                        fc="white",
                        ec="k",
                        lw=1.0,
                        zorder=4,
                    )
                )
            self._ground_mpl(ax, x, -1.02 * sh, 1.5 * sh)
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

    def _ground_mpl(self, ax, xc: float, ytop: float, width: float, n: int = 6):
        """A hatched ground line centred at ``xc`` with its top at ``ytop``."""
        ax.plot(
            [xc - width / 2, xc + width / 2], [ytop, ytop], "k-", lw=1.0, zorder=3
        )
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

        ax.add_patch(
            Circle((h.x, 0), 0.18 * sh, fc="white", ec="k", lw=1.2, zorder=6)
        )

    # --- matplotlib load glyphs ---------------------------------------- #
    def _draw_point_mpl(self, ax, p: PointLoad, pmax: float, sh, color, show_val):
        length = 0.14 * self.L * (0.45 + 0.55 * abs(p.P) / pmax if pmax else 1.0)
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
                f"{abs(p.P):g} kN",
                ha="center",
                va="bottom" if p.P >= 0 else "top",
                color=color,
                fontsize=8,
                zorder=7,
            )

    def _draw_dist_mpl(self, ax, d: DistLoad, wmax: float, color, show_val):
        from matplotlib.patches import Polygon

        w_h = 0.10 * self.L

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
                f"{abs(d.w0):g} kN/m"
                if same
                else f"{abs(d.w0):g}→{abs(d.w1):g} kN/m"
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
        ax.add_patch(
            Polygon([tip, p2, p3], closed=True, fc=color, ec=color, zorder=6)
        )
        if show_val:
            ax.text(
                x,
                r + 0.5 * sh,
                f"{abs(m.M):g} kNm",
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
                ax.plot(
                    [xb, xb], [yd + 0.3 * sh, yd - 0.3 * sh], color="0.35", lw=0.6
                )
            ax.text(
                0.5 * (x0 + x1),
                yd - 0.15 * sh,
                f"{x1 - x0:g} m",
                ha="center",
                va="top",
                color="0.35",
                fontsize=8,
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

        Returns
        -------
        str
            The LaTeX source.
        """
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
                lines.append(
                    f"\t\\dimensioning{{1}}{{{nodes[i]}}}{{{nodes[i + 1]}}}"
                    f"{{{dist:g}}}[${span:g}$~m];"
                )

        if labels:
            lines.append("\t% Labels")
            for s in self.supports:
                pos = "above right" if s.at_right_end else "above left"
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
            rot = "[180]" if s.at_right_end else ""
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
        # Pin marker at simply-supported extremities (matches the doc figures)
        for s in self.supports:
            if s.kind in (PIN, ROLLER) and (s.at_left_end or s.at_right_end):
                out.append(f"\\hinge{{1}}{{{nodes[s.node]}}};")
        # Internal moment releases
        for h in self.hinges:
            i = h.node
            out.append(
                f"\\hinge{{2}}{{{nodes[i]}}}[{nodes[i - 1]}][{nodes[i + 1]}];"
            )
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
                out.append(
                    f"\\lineload{{1}}{{dl{k}a}}{{dl{k}b}}[{h0:g}][{h1:g}];"
                )
            out.append("\\end{scope}")
            if show_val:
                for k, d in enumerate(self.dist_loads):
                    if abs(d.w0 - d.w1) < 1e-9:
                        lbl = f"${abs(d.w0):g}$ kN/m"
                    else:
                        lbl = f"${abs(d.w0):g}\\rightarrow{abs(d.w1):g}$ kN/m"
                    out.append(
                        f"\\notation{{5}}{{dl{k}a}}{{dl{k}b}}[{lbl}][0.5][above=8mm];"
                    )

        for k, p in enumerate(self.point_loads):
            ni, nj = nodes[p.i_span], nodes[p.i_span + 1]
            f = self._span_fraction(p.i_span, p.x)
            ang = "90" if p.P >= 0 else "270"
            out.append(f"\\node (pl{k}) at ($({ni})!{f:g}!({nj})$){{}};")
            out.append(f"\\begin{{scope}}[color=red]\\load{{1}}{{pl{k}}}[{ang}]\\end{{scope}}")
            if show_val:
                out.append(
                    f"\\notation{{1}}{{pl{k}}}{{${abs(p.P):g}$ kN}}[above=12mm];"
                )

        for k, m in enumerate(self.moment_loads):
            ni, nj = nodes[m.i_span], nodes[m.i_span + 1]
            f = self._span_fraction(m.i_span, m.x)
            mtype = "2" if m.M >= 0 else "3"  # 2 = ccw, 3 = cw
            out.append(f"\\node (ml{k}) at ($({ni})!{f:g}!({nj})$){{}};")
            out.append(
                f"\\begin{{scope}}[color=red]\\load{{{mtype}}}{{ml{k}}}\\end{{scope}}"
            )
            if show_val:
                out.append(
                    f"\\notation{{2}}{{ml{k}}}{{${abs(m.M):g}$ kNm}}[above=5mm];"
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
