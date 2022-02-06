"""
PyCBA - Utility functions for interacting with PyCBA
"""
import re
import numpy as np
from typing import Tuple


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
        if t == "p" or t == "r":  # pin or roller
            R.append([-1, 0])
        elif t == "e":  # encastre
            R.append([-1, -1])
        elif t == "f":  # free
            R.append([0, 0])
        elif t == "h":  # hinge
            R.append([0, 0])
            eType[i - 1] = 2
    R = [elem for sublist in R for elem in sublist]

    return (L, EI, R, eType)
