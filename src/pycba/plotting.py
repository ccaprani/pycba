"""
PyCBA - optional Plotly plotting backend.

Interactive, hover-to-read versions of the result diagrams and envelopes.
Hovering over a curve reads the value at that section; with the shared x-axis
of the combined plots, one hover reports the bending moment, shear and
deflection at the same position.

Plotly is an **optional** dependency - the default backend remains matplotlib.
Install it with ``pip install pycba[plotly]`` (or ``pip install plotly``).

Select the backend per call with the ``backend=`` argument on the plotting
methods, or globally with :func:`set_backend`::

    import pycba as cba
    cba.set_backend("plotly")          # all subsequent plots are interactive
    beam_analysis.plot_results()       # -> a plotly Figure

The bending-moment panels keep PyCBA's sagging-positive convention (the y-axis
is reversed so sagging plots below the beam line), matching the matplotlib
backend.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from .units import resolve

_BACKENDS = ("matplotlib", "plotly")
_default_backend = "matplotlib"

# Curve colours, matching the matplotlib backend's red/blue.
_RED = "#d62728"
_BLUE = "#1f77b4"
_BAND = "rgba(120,120,120,0.15)"
# Light-red diagram fill, matching the matplotlib BMD/SFD shading.
_FILL = "rgba(214,39,40,0.12)"


def set_backend(name: str) -> None:
    """
    Set the default plotting backend used by the PyCBA result plots.

    Parameters
    ----------
    name : {"matplotlib", "plotly"}
        The backend to use when a plotting method is called without an explicit
        ``backend=`` argument.  Selecting ``"plotly"`` checks that plotly is
        importable and raises ``ModuleNotFoundError`` if it is not.
    """
    global _default_backend
    name = str(name).lower()
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {_BACKENDS}")
    if name == "plotly":
        _require_plotly()
    _default_backend = name


def get_backend() -> str:
    """Return the current default plotting backend."""
    return _default_backend


def resolve_backend(backend: Optional[str]) -> str:
    """Resolve a per-call ``backend`` argument against the global default."""
    name = (backend or _default_backend).lower()
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {_BACKENDS}")
    return name


def _require_plotly():
    """Import plotly, raising a helpful error if it is not installed."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "The plotly plotting backend requires the optional 'plotly' package. "
            "Install it with `pip install plotly` (or `pip install pycba[plotly]`)."
        ) from exc
    return go, make_subplots


def _hovertemplate(name: str, unit: str) -> str:
    """Hover text for a curve under ``hovermode='x unified'`` (x is the header)."""
    suffix = f" {unit}" if unit else ""
    return f"{name}: %{{y:.4g}}{suffix}<extra></extra>"


def _layout(fig, us, title, *, bottom_row=None):
    """Apply the shared layout (unified hover, theme, axis title) to a figure.

    ``bottom_row=None`` is a single-panel figure; otherwise the x-axis title is
    placed on the given subplot row.
    """
    if bottom_row is None:
        fig.update_xaxes(title_text=us.distance_axis)
    else:
        fig.update_xaxes(title_text=us.distance_axis, row=bottom_row, col=1)
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        title=title,
        margin=dict(l=70, r=20, t=40 if title else 20, b=50),
    )
    return fig


def diagram_figure(mr, kind: str, units=None, title: Optional[str] = None, defl=None):
    """
    Build an interactive single result diagram (bending moment, shear or
    deflection) from a :class:`pycba.MemberResults`-like object.

    Parameters
    ----------
    mr : object
        Anything exposing ``x``, ``M``, ``V`` and ``D`` arrays (e.g.
        ``BeamAnalysis.beam_results.results``).
    kind : {"M", "V", "D"}
        Which load effect to draw.
    units : str or pycba.units.UnitSystem, optional
        Display unit system (see :func:`pycba.set_units`).
    title : str, optional
        Figure title.
    defl : (array, array), optional
        The de-padded ``(x, D)`` deflected shape (native units); when given and
        ``kind == "D"`` it is used instead of ``mr.D`` so the closure padding
        does not draw spurious verticals at member ends.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go, _ = _require_plotly()
    us = resolve(units)
    x = np.asarray(mr.x)
    axis_label, name, y, unit, invert, do_fill = {
        "M": (
            us.moment_axis,
            "Bending moment",
            np.asarray(mr.M),
            us.moment,
            True,
            True,
        ),
        "V": (us.shear_axis, "Shear force", np.asarray(mr.V), us.force, False, True),
        "D": (
            us.deflection_axis,
            "Deflection",
            np.asarray(mr.D) * us.disp_scale,
            us.disp_label,
            False,
            False,
        ),
    }[kind]
    if kind == "D" and defl is not None:
        x = np.asarray(defl[0])
        y = np.asarray(defl[1]) * us.disp_scale

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=_RED),
            fill="tozeroy" if do_fill else None,
            fillcolor=_FILL,
            hovertemplate=_hovertemplate(name, unit),
        )
    )
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_yaxes(title_text=axis_label)
    if invert:
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    return _layout(fig, us, title)


def results_figure(mr, units=None, title: Optional[str] = None, defl=None):
    """
    Build the combined interactive bending-moment / shear / deflection figure
    (three stacked panels sharing the x-axis) from a member-results object.

    A single hover reports all three effects at the same section.  The moment
    and shear diagrams are shaded to the baseline; ``defl`` (the de-padded
    ``(x, D)`` deflected shape) is used for the deflection panel when given.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go, make_subplots = _require_plotly()
    us = resolve(units)
    x = np.asarray(mr.x)
    if defl is not None:
        xD, yD = np.asarray(defl[0]), np.asarray(defl[1]) * us.disp_scale
    else:
        xD, yD = x, np.asarray(mr.D) * us.disp_scale
    rows = [
        (us.moment_axis, "Bending moment", x, np.asarray(mr.M), us.moment, True, True),
        (us.shear_axis, "Shear force", x, np.asarray(mr.V), us.force, False, True),
        (us.deflection_axis, "Deflection", xD, yD, us.disp_label, False, False),
    ]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for i, (axis_label, name, xrow, y, unit, invert, do_fill) in enumerate(
        rows, start=1
    ):
        fig.add_trace(
            go.Scatter(
                x=xrow,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=_RED),
                fill="tozeroy" if do_fill else None,
                fillcolor=_FILL,
                hovertemplate=_hovertemplate(name, unit),
            ),
            row=i,
            col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=1), row=i, col=1)
        fig.update_yaxes(title_text=axis_label, row=i, col=1)
        if invert:
            fig.update_yaxes(autorange="reversed", row=i, col=1)
    fig.update_layout(showlegend=False)
    return _layout(fig, us, title, bottom_row=3)


def envelope_figure(env, units=None, title: Optional[str] = None):
    """
    Build the interactive moment/shear envelope figure (two stacked panels) from
    a :class:`pycba.Envelopes` object, with the max/min band shaded.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go, make_subplots = _require_plotly()
    us = resolve(units)
    x = np.asarray(env.x)
    panels = [
        (us.moment_axis, "M", env.Mmax, env.Mmin, us.moment, True),
        (us.shear_axis, "V", env.Vmax, env.Vmin, us.force, False),
    ]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for i, (axis_label, sym, ymax, ymin, unit, invert) in enumerate(panels, start=1):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.asarray(ymax),
                mode="lines",
                name=f"{sym} max",
                line=dict(color=_RED),
                hovertemplate=_hovertemplate(f"{sym} max", unit),
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.asarray(ymin),
                mode="lines",
                name=f"{sym} min",
                line=dict(color=_BLUE),
                fill="tonexty",
                fillcolor=_BAND,
                hovertemplate=_hovertemplate(f"{sym} min", unit),
            ),
            row=i,
            col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=1), row=i, col=1)
        fig.update_yaxes(title_text=axis_label, row=i, col=1)
        if invert:
            fig.update_yaxes(autorange="reversed", row=i, col=1)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.18))
    return _layout(fig, us, title, bottom_row=2)
