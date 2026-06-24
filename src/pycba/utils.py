"""
PyCBA - Utility functions for interacting with PyCBA
"""
import re
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union


# Per-node support-name vocabulary, shared by :func:`parse_beam_string` and the
# friendlier ``supports=`` constructor argument.  Each name lowers to a
# ``[vertical, rotation]`` restraint pair in the ``R`` convention (``-1`` = fixed,
# ``0`` = free); a 1D beam node carries only these two DOFs.  Pin and roller are
# identical in a beam (no horizontal DOF distinguishes them); both names are kept
# for readability and visualisation intent.
_SUPPORT_DOF = {
    "p": [-1, 0],
    "pin": [-1, 0],
    "pinned": [-1, 0],
    "r": [-1, 0],
    "roller": [-1, 0],
    "e": [-1, -1],
    "encastre": [-1, -1],
    "fixed": [-1, -1],
    "clamped": [-1, -1],
    "f": [0, 0],
    "free": [0, 0],
}

# Support type that is *not* a ground restraint but an internal moment release;
# only meaningful inside a beam string (where it also pins the previous member).
_HINGE_NAMES = {"h", "hinge"}

SupportType = Union[str, Sequence[float]]


def _support_pair(entry: SupportType) -> List[float]:
    """
    Lower a single per-node support to its ``[vertical, rotation]`` DOF pair.

    Parameters
    ----------
    entry : str or sequence of float
        Either a support name (case-insensitive; see :data:`_SUPPORT_DOF`) or a
        raw two-element ``[vertical, rotation]`` restraint pair in the ``R``
        convention.  A pair is the escape hatch for elastic springs, e.g.
        ``[5e4, 0]`` is a vertical spring of stiffness ``5e4`` with the rotation
        free.

    Raises
    ------
    ValueError
        If a name is unknown, names an internal hinge (which is a member
        release, not a support), or a numeric pair does not have exactly two
        entries.
    """
    if isinstance(entry, str):
        key = entry.strip().lower()
        if key in _HINGE_NAMES:
            raise ValueError(
                "A hinge is an internal moment release, not a support; do not "
                "list it in `supports`. Release the moment on the adjacent "
                "member via its element type instead (e.g. eletype 'FP'/'PF')."
            )
        try:
            return list(_SUPPORT_DOF[key])
        except KeyError:
            names = ", ".join(sorted(_SUPPORT_DOF))
            raise ValueError(
                f"Unknown support {entry!r}; use one of: {names}; or a raw "
                f"[vertical, rotation] DOF pair (e.g. [5e4, 0] for a spring)."
            )
    pair = list(entry)
    if len(pair) != 2:
        raise ValueError(
            f"A support DOF pair must have exactly 2 entries "
            f"[vertical, rotation], got {entry!r}."
        )
    return pair


def supports_to_R(
    supports: Sequence[SupportType], n_nodes: Optional[int] = None
) -> list:
    """
    Lower a per-node list of named supports to a raw restraint vector ``R``.

    This is the friendly front-end to the low-level ``R`` vector used throughout
    PyCBA: one entry per node (left to right), each either a support *name* or a
    raw ``[vertical, rotation]`` DOF pair.  The result is exactly the ``R`` you
    would otherwise write by hand, so ``supports=`` and ``R=`` are
    interchangeable on the :class:`~pycba.Beam` / :class:`~pycba.BeamAnalysis`
    constructors.

    Recognised names (case-insensitive):

    * ``"p"`` / ``"pin"`` / ``"pinned"`` and ``"r"`` / ``"roller"`` -> ``[-1, 0]``
      (vertical held, rotation free).
    * ``"e"`` / ``"encastre"`` / ``"fixed"`` / ``"clamped"`` -> ``[-1, -1]``
      (fully fixed).
    * ``"f"`` / ``"free"`` -> ``[0, 0]`` (unrestrained, e.g. a cantilever tip).

    Elastic springs are given as a raw pair, e.g. ``[5e4, 0]`` for a vertical
    spring (stiffness ``5e4``) with the rotation free.

    An internal hinge is **not** a support (it is a member moment release) and is
    rejected; release the moment on the adjacent member via its element type
    instead.

    Parameters
    ----------
    supports : sequence of (str or [float, float])
        One entry per node, ordered left to right.
    n_nodes : int, optional
        Expected number of nodes (``= number of spans + 1``).  If given, the
        length of ``supports`` is validated against it for a clear early error.

    Returns
    -------
    list
        The restraint vector ``R`` of length ``2 * len(supports)``.

    Raises
    ------
    ValueError
        If ``n_nodes`` is given and does not match ``len(supports)``, or any
        entry is not a recognised name or valid ``[vertical, rotation]`` pair.
    """
    if n_nodes is not None and len(supports) != n_nodes:
        raise ValueError(
            f"Expected {n_nodes} supports (one per node), got {len(supports)}."
        )
    R = []
    for entry in supports:
        R.extend(_support_pair(entry))
    return R


def parse_beam_string(
    beam_string: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function parses a beam descriptor string and returns CBA input vectors.

    The beam descriptor string uses a specific format: spans lengths in float are
    separated by single characters describing the terminals of that beam element.
    The terminal characters are:

        - P - pinned (effectively the same as roller, but retained for visualisations)
        - R - roller (can occur at any terminal)
        - E - encastre (i.e. fully-fixed) - can only occur at beam extremity
        - F - free (e.g. cantilever end) - can only occur at beam extremity
        - H - hinge - can only occur internally in the beam

    Examples of beam strings are:

        - *P40R20R* - 2-span, 60 m long, with pinned-roller-roller supports
        - *E20H30R10F* - 3-span, 60 m long, encastre-hinge-roller-free

    **Complex beam configurations may not be describable using the beam string.**

    The function returns a tuple containing the necessary beam inputs for
    :class:`pycba.analysis.BeamAnalysis`: `(L, EI, R, eType)`

    Parameters
    ----------
    beam_string :
        The string to be parsed.

    Raises
    ------
    ValueError
        When the beam string does not meet basic structural requirements.

    Returns
    -------
    (L, EI, R, eType) : tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        In which:
            - `L` is a vector of span lengths.
            - `EI` is A vector of member flexural rigidities (prismatic).
            - `R` is a vector describing the support conditions at each member end.
            - `eType` is a vector of the member types.

    Example
    -------
    This example creates a four-span beam with fixed extreme supports and
    an internal hinge. ::

        beam_str = "E30R30H30R30E"
        (L, EI, R, eType) = cba.parse_beam_string(beam_str)
        ils = cba.InfluenceLines(L, EI, R, eType)
        ils.create_ils(step=0.1)
        ils.plot_il(0.0, "R")

    """

    beam_string = beam_string.lower()
    terminals = re.findall(r"[efhpr]", beam_string)
    spans_str = [m.end() for m in re.finditer(r"[efhpr]", beam_string)]

    if len(terminals) < 2:
        raise ValueError("At least two terminals must be defined")
    if terminals[0] == "h" or terminals[-1] == "h":
        raise ValueError("Cannot have a hinge at an extremity")
    if len(terminals) > 2:
        if any(t == "f" or t == "e" for t in terminals[1:-1]):
            raise ValueError("Do not define internal free or encastre terminals")

    # Get and check the span lengths
    L = [
        float(beam_string[spans_str[i] : spans_str[i + 1] - 1])
        for i in range(len(spans_str) - 1)
    ]
    if len(terminals) - 1 != len(L):
        raise ValueError("Inconsistent terminal count and span count")

    EI = 30 * 1e10 * np.ones(len(L)) * 1e-6  # kNm2 - arbitrary value
    R = []
    eType = [1 for l in L]
    for i, t in enumerate(terminals):
        if t == "h":  # internal hinge: unsupported node + release of prev member
            R.extend([0, 0])
            eType[i - 1] = 2
        else:  # p / r / e / f map via the shared support vocabulary
            R.extend(_SUPPORT_DOF[t])

    return (L, EI, R, eType)
