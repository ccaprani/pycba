"""
PyCBA - Variable (non-prismatic) section properties
====================================================

This module provides :class:`SectionEI`, a **segment builder** describing the
flexural rigidity :math:`EI(x)` of a single **non-prismatic** span.  The
rigidity is assembled from one or more contiguous *segments*, each of which is
constant, linear, piecewise-linear (``pwl``) or polynomial (``poly``).  The
resulting piecewise function :math:`EI(x)` is consumed by the
flexibility-integrated element stiffness in :class:`pycba.beam.Beam` and by the
curvature-based post-processing in :class:`pycba.results.BeamResults`.

The prismatic (constant-:math:`EI`, scalar) path in PyCBA is completely
unaffected: a member is treated as non-prismatic *only* when its rigidity is
supplied as a :class:`SectionEI` object rather than a scalar ``float``.

Coordinate convention
----------------------
``x`` is the **span-local physical coordinate**: ``0`` at the start (``i``-end)
of the span and ``L`` (the real span length, in length units) at the end.  It
is *not* normalised and *not* a global multi-span coordinate.

Segments and breakpoints
------------------------
Segments are added head-to-tail and must be **contiguous**: each new segment's
``x[0]`` must equal the running end coordinate (the first segment starts at
``0``).  A coincident ``x`` carrying a *different* ``EI`` across a join is an
allowed **step** (a genuine discontinuity in ``EI``).  The
:attr:`~SectionEI.breakpoints` are the sorted span-local boundary coordinates,
including every segment join and every interior ``pwl`` kink; the element
flexibility is integrated piece-by-piece *between* consecutive breakpoints so
that kinks and steps are captured exactly.

The total coverage (the running end coordinate of the last segment) is
validated against the span length when the section is attached to a
:class:`~pycba.beam.Beam`.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

#: Supported segment types.
_SEG_TYPES = ("const", "linear", "pwl", "poly")


class _Piece:
    """A single linear or polynomial piece of ``EI(x)`` over ``[x0, x1]``.

    Internally every segment is reduced to one or more such pieces, each with
    its own polynomial coefficients (highest order first, as ``numpy.polyval``
    expects) and the polynomial degree, so the breakpoint-aware integrator can
    pick a Gauss order sufficient for the piece.
    """

    __slots__ = ("x0", "x1", "coeffs", "degree")

    def __init__(self, x0: float, x1: float, coeffs: np.ndarray, degree: int):
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.degree = int(degree)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.polyval(self.coeffs, x)


class SectionEI:
    """
    Segment-built, piecewise variable flexural rigidity over one span.

    A member assigned a :class:`SectionEI` is analysed with the
    flexibility-integrated non-prismatic element (see
    :meth:`pycba.beam.Beam.k_nonprismatic`), whose flexibility integrals are
    evaluated **piece-by-piece between consecutive breakpoints**.  When the
    section is a single constant segment the element reproduces the closed-form
    prismatic stiffness to machine precision.

    The section is built by chaining :meth:`add_segment` calls, or in one line
    by passing a list of segment specs to the constructor.
    """

    def __init__(self, segments: Optional[Sequence] = None):
        """
        Construct a (possibly empty) variable-rigidity section.

        Parameters
        ----------
        segments : sequence of segment specs, optional
            When given, each spec is added in turn via :meth:`add_segment`.  A
            spec is either a tuple/list ``(seg_type, x, ei)`` or
            ``(seg_type, x, ei, degree)``, or a mapping with keys ``seg_type``,
            ``x``, ``ei`` and optionally ``degree``.  When ``None`` (the
            default) an empty section is created, to be populated by chained
            :meth:`add_segment` calls.

        Examples
        --------
        One-liner (a straight haunch then a flat soffit)::

            sec = SectionEI([
                ("linear", [0.0, 3.0], [3.0e5, 1.2e5]),
                ("const", [3.0, 9.0], 1.2e5),
                ("linear", [9.0, 12.0], [1.2e5, 3.0e5]),
            ])

        Chained::

            sec = (SectionEI()
                   .add_segment("linear", [0.0, 3.0], [3.0e5, 1.2e5])
                   .add_segment("const", [3.0, 9.0], 1.2e5)
                   .add_segment("linear", [9.0, 12.0], [1.2e5, 3.0e5]))
        """
        self._pieces: list[_Piece] = []
        self._joins: list[float] = []  # segment-join coordinates (running ends)
        self._kinks: list[float] = []  # interior pwl kink coordinates
        self._x0: float = 0.0  # start of the very first segment (always 0)
        self._end: float = 0.0  # running end coordinate

        if segments is not None:
            for spec in segments:
                if isinstance(spec, dict):
                    self.add_segment(
                        spec["seg_type"],
                        spec["x"],
                        spec["ei"],
                        spec.get("degree"),
                    )
                else:
                    self.add_segment(*spec)

    # ------------------------------------------------------------------
    #  Builder
    # ------------------------------------------------------------------
    def add_segment(
        self,
        seg_type: str,
        x: Sequence[float],
        ei: Union[float, Sequence[float], Callable],
        degree: Optional[int] = None,
    ) -> "SectionEI":
        """
        Append a contiguous segment to the section.

        Parameters
        ----------
        seg_type : {'const', 'linear', 'pwl', 'poly'}
            The kind of variation over the segment:

            * ``'const'`` -- constant ``EI`` over ``[x0, x1]``.  ``x = [x0, x1]``
              and ``ei`` is a scalar.
            * ``'linear'`` -- a single linear piece.  ``x = [x0, x1]`` and
              ``ei = [ei0, ei1]``.
            * ``'pwl'`` -- piecewise-linear.  ``x = [x0, x1, ..., xn]`` with
              ``n >= 2`` strictly-increasing stations and ``ei`` of the same
              length; ``n - 1`` linear pieces with kinks at the interior
              stations.  (``'linear'`` is the two-point case; ``'const'`` a
              flat run.)
            * ``'poly'`` -- a single polynomial piece over ``[x[0], x[-1]]``.
              ``ei`` is either a list of sample values (same length as ``x``;
              a polynomial of order ``degree``, default ``len(x) - 1``, is
              fitted) **or** a ``callable`` ``ei(x_local)`` evaluated in the
              span-local physical coordinate (in which case ``degree`` sets the
              integration order, default 8).

        x : array_like of float
            The segment station(s) in span-local physical coordinates, strictly
            increasing.
        ei : float, array_like of float, or callable
            The rigidity value(s) or, for ``'poly'``, optionally a callable.
        degree : int, optional
            Polynomial order for a ``'poly'`` segment.  Ignored for the other
            types.

        Returns
        -------
        SectionEI
            ``self``, to allow chaining.

        Raises
        ------
        ValueError
            If ``seg_type`` is unknown; ``x`` is not strictly increasing; the
            lengths of ``x`` and ``ei`` are inconsistent; any rigidity is
            non-positive; or the segment is not contiguous with the running
            end of the section (a gap or an overlap).
        """
        if seg_type not in _SEG_TYPES:
            raise ValueError(
                f"Unknown seg_type {seg_type!r}; expected one of {_SEG_TYPES}"
            )

        x = np.asarray(x, dtype=float).ravel()
        if x.ndim != 1 or len(x) < 2:
            raise ValueError("x must list at least the segment start and end")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("x positions within a segment must be strictly increasing")

        x0, x1 = float(x[0]), float(x[-1])

        # Contiguity: first segment must start at 0; later ones at the running end.
        anchor = 0.0 if not self._pieces else self._end
        if not np.isclose(x0, anchor, rtol=0.0, atol=1e-9 * max(1.0, abs(anchor))):
            kind = "first segment must start at x = 0" if not self._pieces else (
                f"segment must start at the running end x = {anchor}"
            )
            raise ValueError(
                f"Non-contiguous segment: {kind}, got x[0] = {x0} "
                f"(gaps and overlaps are not allowed)"
            )

        new_pieces: list[_Piece] = []
        new_kinks: list[float] = []

        if seg_type == "const":
            ei_val = self._scalar(ei, "const")
            self._check_positive([ei_val])
            new_pieces.append(_Piece(x0, x1, np.array([ei_val]), degree=0))

        elif seg_type in ("linear", "pwl"):
            ei_arr = np.asarray(ei, dtype=float).ravel()
            if len(ei_arr) != len(x):
                raise ValueError(
                    f"{seg_type!r} segment needs len(ei) == len(x) "
                    f"({len(ei_arr)} != {len(x)})"
                )
            if seg_type == "linear" and len(x) != 2:
                raise ValueError("'linear' segment requires exactly 2 stations")
            self._check_positive(ei_arr)
            for k in range(len(x) - 1):
                xa, xb = float(x[k]), float(x[k + 1])
                ya, yb = float(ei_arr[k]), float(ei_arr[k + 1])
                slope = (yb - ya) / (xb - xa)
                # y = ya + slope*(x - xa) -> polyval coeffs [slope, ya - slope*xa]
                coeffs = np.array([slope, ya - slope * xa])
                new_pieces.append(_Piece(xa, xb, coeffs, degree=1))
                if k > 0:
                    new_kinks.append(xa)

        elif seg_type == "poly":
            if callable(ei):
                deg = 8 if degree is None else int(degree)
                xs = np.linspace(x0, x1, deg + 2)
                ys = np.asarray(ei(xs), dtype=float).ravel()
                self._check_positive(ys)
                coeffs = np.polyfit(xs, ys, deg)
                new_pieces.append(_Piece(x0, x1, coeffs, degree=deg))
            else:
                ei_arr = np.asarray(ei, dtype=float).ravel()
                if len(ei_arr) != len(x):
                    raise ValueError(
                        "'poly' segment with sample values needs len(ei) == len(x) "
                        f"({len(ei_arr)} != {len(x)})"
                    )
                self._check_positive(ei_arr)
                deg = (len(x) - 1) if degree is None else int(degree)
                coeffs = np.polyfit(x, ei_arr, deg)
                new_pieces.append(_Piece(x0, x1, coeffs, degree=deg))

        # Commit.  The first segment fixes the section start coordinate; every
        # segment records its end coordinate as a join (a breakpoint).
        if not self._joins:
            self._x0 = x0
        self._pieces.extend(new_pieces)
        self._kinks.extend(new_kinks)
        self._joins.append(x1)
        self._end = x1
        return self

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _scalar(ei, seg_type: str) -> float:
        arr = np.asarray(ei, dtype=float).ravel()
        if arr.size != 1:
            raise ValueError(f"'{seg_type}' segment requires a scalar ei")
        return float(arr[0])

    @staticmethod
    def _check_positive(values) -> None:
        if np.any(np.asarray(values, dtype=float) <= 0.0):
            raise ValueError("EI values must be strictly positive")

    # ------------------------------------------------------------------
    #  Geometry / query
    # ------------------------------------------------------------------
    @property
    def length(self) -> float:
        """float : total covered length (running end of the last segment)."""
        return self._end - self._x0

    @property
    def breakpoints(self) -> np.ndarray:
        """
        np.ndarray : sorted span-local breakpoint coordinates.

        Includes the section start, every segment join, and every interior
        ``pwl`` kink.  Coincident coordinates (e.g. a step at a join) are
        collapsed to a single value.  The flexibility integration is split at
        these breakpoints so kinks and steps are captured exactly.
        """
        bps = [self._x0] + list(self._joins) + list(self._kinks)
        bps = np.array(sorted(bps), dtype=float)
        if len(bps) == 0:
            return bps
        # Collapse near-duplicates (a step shares an x with the next join start).
        keep = np.concatenate([[True], np.diff(bps) > 1e-9 * max(1.0, self.length)])
        return bps[keep]

    @property
    def pieces(self) -> list:
        """list of :class:`_Piece` : the internal linear/poly pieces."""
        return self._pieces

    @property
    def is_constant(self) -> bool:
        """bool : ``True`` when ``EI(x)`` is a single constant value."""
        if not self._pieces:
            return False
        if any(p.degree > 0 for p in self._pieces):
            return False
        vals = [float(p.coeffs[-1]) for p in self._pieces]
        return bool(np.allclose(vals, vals[0]))

    def validate_length(self, L: float, atol: float = 1e-6) -> None:
        """
        Check that the section's total coverage equals the span length ``L``.

        Parameters
        ----------
        L : float
            The span length the section is attached to.
        atol : float, optional
            Absolute tolerance (scaled by ``L``).  Default ``1e-6``.

        Raises
        ------
        ValueError
            If the section is empty, or its coverage differs from ``L``.
        """
        if not self._pieces:
            raise ValueError("SectionEI has no segments")
        tol = atol * max(1.0, abs(L))
        if not np.isclose(self.length, L, rtol=0.0, atol=tol):
            raise ValueError(
                f"SectionEI coverage {self.length} does not match the span "
                f"length {L}; add segments so the pieces span the full length"
            )

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate ``EI(x)`` piecewise in the span-local physical coordinate.

        At an internal step (coincident ``x`` with differing ``EI``) the value
        of the piece *ending* at ``x`` is returned (left-continuous); at the
        section ends the natural end piece is used.

        Parameters
        ----------
        x : float or np.ndarray
            Span-local position(s).

        Returns
        -------
        float or np.ndarray
            The rigidity ``EI(x)``.
        """
        if not self._pieces:
            raise ValueError("SectionEI has no segments to evaluate")

        scalar = np.isscalar(x)
        xv = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty_like(xv)
        # x0..x1 boundaries of each piece; assign each query to the first piece
        # whose [x0, x1] contains it (left-to-right, so a step takes the left
        # piece's value at the shared coordinate).
        assigned = np.zeros(xv.shape, dtype=bool)
        for p in self._pieces:
            mask = (~assigned) & (xv >= p.x0 - 1e-12) & (xv <= p.x1 + 1e-12)
            if np.any(mask):
                out[mask] = p(xv[mask])
                assigned |= mask
        # Anything outside the covered range: clamp to the nearest end piece.
        if not np.all(assigned):
            for p, sel in ((self._pieces[0], xv < self._x0), (self._pieces[-1], xv > self._end)):
                m = (~assigned) & sel
                if np.any(m):
                    out[m] = p(xv[m])
                    assigned |= m
        return float(out[0]) if scalar else out

    # ------------------------------------------------------------------
    #  Visualisation
    # ------------------------------------------------------------------
    def plot(
        self,
        ax=None,
        n: int = 200,
        show_breakpoints: bool = True,
        annotate: bool = True,
        **kwargs,
    ):
        """
        Plot ``EI(x)`` over the whole span as an input-verification figure.

        The rigidity is drawn **piece by piece** over each piece's own
        ``[x0, x1]`` interval, so the curve faithfully reflects what was
        entered:

        * a **kink** (e.g. a ``pwl`` interior station, or a join where the
          slope changes) renders as a slope change with no break, because
          adjacent pieces share the boundary value;
        * a **step** (a coincident ``x`` carrying a different ``EI`` across a
          join) renders as a genuine discontinuity -- the left piece ends at
          its value and the right piece starts at its own value, with no
          spurious vertical line connecting them.  A thin dashed connector is
          drawn at each such step purely to make the jump easy to read.

        Constant pieces appear flat, linear pieces straight, and polynomial
        pieces are sampled smoothly.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into.  When ``None`` (default) a new figure and axes
            are created with ``plt.subplots(**kwargs)``; otherwise the given
            axes (and its parent figure) are used and ``**kwargs`` is ignored.
        n : int, optional
            Total number of sample points distributed across the pieces
            (proportional to each piece's length, at least 2 per piece).  Used
            for smooth sampling of polynomial pieces; default ``200``.
        show_breakpoints : bool, optional
            When ``True`` (default) mark the segment boundaries and ``pwl``
            kinks at :attr:`breakpoints` with light vertical gridlines.
        annotate : bool, optional
            When ``True`` (default) lightly label each piece with its degree
            (``const`` / ``linear`` / ``poly``) via a legend, so the entered
            variation can be confirmed at a glance.
        **kwargs
            Passed to :func:`matplotlib.pyplot.subplots` when ``ax is None``.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The figure and axes, matching the PyCBA plotting convention.
            ``plt.show()`` is never called.
        """
        if not self._pieces:
            raise ValueError("SectionEI has no segments to plot")

        if ax is None:
            fig, ax = plt.subplots(**kwargs)
        else:
            fig = ax.figure

        # Human-readable label for a piece, by polynomial degree.
        def _kind(degree: int) -> str:
            return {0: "const", 1: "linear"}.get(degree, f"poly (deg {degree})")

        # Distribute the sample budget across pieces by length (>= 2 each).
        total_len = sum(max(p.x1 - p.x0, 0.0) for p in self._pieces)
        total_len = total_len if total_len > 0.0 else 1.0

        # Colour each piece by its *kind* so pieces of the same kind share a
        # colour; the by-kind legend then faithfully represents every piece.
        # (A section like linear/const/linear shows a single "linear" entry that
        # covers both haunches, rather than dropping the repeated kind and
        # leaving the second linear piece unrepresented.)
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        kind_color: dict = {}
        prev_x1 = None
        prev_y1 = None
        for p in self._pieces:
            npts = max(2, int(round(n * (p.x1 - p.x0) / total_len)))
            xs = np.linspace(p.x0, p.x1, npts)
            ys = p(xs)

            label = _kind(p.degree)
            if label in kind_color:
                color, legend_label = kind_color[label], None
            else:
                color = (
                    prop_cycle[len(kind_color) % len(prop_cycle)]
                    if prop_cycle
                    else None
                )
                kind_color[label] = color
                legend_label = label if annotate else None
            ax.plot(xs, ys, lw=2, color=color, label=legend_label)

            # A step: this piece starts at the previous piece's end x but with a
            # different value.  Draw a thin dashed connector to read the jump.
            if (
                prev_x1 is not None
                and np.isclose(p.x0, prev_x1, atol=1e-9 * max(1.0, self.length))
                and not np.isclose(float(ys[0]), float(prev_y1), rtol=1e-9, atol=0.0)
            ):
                ax.plot(
                    [p.x0, p.x0], [prev_y1, ys[0]], "k--", lw=0.8, alpha=0.6
                )
            prev_x1, prev_y1 = p.x1, float(ys[-1])

        if show_breakpoints:
            for bp in self.breakpoints:
                ax.axvline(bp, color="0.7", lw=0.8, ls=":", zorder=0)

        ax.set_xlim(self._x0, self._end)
        ax.set_xlabel("distance along span (local)")
        ax.set_ylabel("EI")
        ax.set_title("SectionEI — input check")
        ax.grid(True)
        if annotate and kind_color:
            ax.legend(loc="best", fontsize="small")

        return fig, ax

    def __repr__(self) -> str:
        segs = ", ".join(
            f"[{p.x0:g},{p.x1:g}] deg{p.degree}" for p in self._pieces
        )
        return f"SectionEI({len(self._pieces)} pieces: {segs})"
