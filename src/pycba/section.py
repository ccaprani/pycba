"""
PyCBA - Variable (non-prismatic) section properties
====================================================

This module provides :class:`SectionEI`, a lightweight container describing a
**non-prismatic** member whose flexural rigidity :math:`EI` varies along the
length of the span.  The rigidity is defined at a set of points along the
member and is polynomial-interpolated to a continuous function :math:`EI(x)`
which is then consumed by the flexibility-integrated element stiffness in
:class:`pycba.beam.Beam` and by the curvature-based post-processing in
:class:`pycba.results.BeamResults`.

The prismatic (constant-:math:`EI`, scalar) path in PyCBA is completely
unaffected: a member is treated as non-prismatic *only* when its rigidity is
supplied as a :class:`SectionEI` object rather than a scalar ``float``.

Interpolation
-------------
``EI(x)`` is built with :func:`numpy.polyfit` of degree
``min(len(points) - 1, max_degree)``.  Two points give a linear variation,
three a parabola, and so on, capped at ``max_degree`` (default 6) to avoid
ill-conditioned high-order fits.  A single point degenerates to a constant
(i.e. prismatic) rigidity.

Sign / coordinate convention
----------------------------
Positions are measured from the **left (i) end** of the member in local
member coordinates, consistent with the rest of PyCBA.
"""

from __future__ import annotations

from typing import Sequence, Union
import numpy as np


class SectionEI:
    """
    Point-defined, polynomial-interpolated variable flexural rigidity.

    A member assigned a :class:`SectionEI` is analysed with the
    flexibility-integrated non-prismatic element (see
    :meth:`pycba.beam.Beam.k_nonprismatic`).  When all supplied rigidity
    values are equal the interpolation collapses to a constant and the element
    reproduces the closed-form prismatic stiffness to machine precision.
    """

    def __init__(
        self,
        x: Sequence[float],
        EI: Sequence[float],
        max_degree: int = 6,
    ):
        """
        Construct a variable-rigidity section description.

        Parameters
        ----------
        x : array_like of float
            Positions along the member (local coordinates, measured from the
            left/``i`` end) at which the rigidity is specified.  Need not be
            equally spaced, but must be strictly increasing.
        EI : array_like of float
            Flexural rigidity values at the corresponding ``x`` positions.
            Must be strictly positive.
        max_degree : int, optional
            Maximum polynomial degree used to interpolate ``EI(x)``.  The
            actual degree is ``min(len(x) - 1, max_degree)``.  Default 6.

        Raises
        ------
        ValueError
            If ``x`` and ``EI`` differ in length, fewer than one point is
            given, ``x`` is not strictly increasing, or any ``EI`` is
            non-positive.
        """
        x = np.asarray(x, dtype=float)
        EI = np.asarray(EI, dtype=float)

        if x.ndim != 1 or EI.ndim != 1:
            raise ValueError("x and EI must be one-dimensional")
        if len(x) != len(EI):
            raise ValueError("x and EI must have the same length")
        if len(x) < 1:
            raise ValueError("At least one (x, EI) point must be supplied")
        if np.any(EI <= 0.0):
            raise ValueError("EI values must be strictly positive")
        if len(x) > 1 and np.any(np.diff(x) <= 0.0):
            raise ValueError("x positions must be strictly increasing")

        self.x = x
        self.EI = EI
        self.max_degree = max_degree

        if len(x) == 1:
            # Degenerate (prismatic) case: constant polynomial.
            self._coeffs = np.array([EI[0]])
        else:
            degree = min(len(x) - 1, max_degree)
            self._coeffs = np.polyfit(x, EI, degree)

    @property
    def is_constant(self) -> bool:
        """bool : ``True`` when all supplied rigidity values are identical."""
        return bool(np.allclose(self.EI, self.EI[0]))

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the interpolated rigidity ``EI(x)``.

        Parameters
        ----------
        x : float or np.ndarray
            Position(s) along the member (local coordinates).

        Returns
        -------
        float or np.ndarray
            The interpolated flexural rigidity at ``x``.
        """
        return np.polyval(self._coeffs, x)

    def __repr__(self) -> str:
        return (
            f"SectionEI(x={self.x.tolist()}, EI={self.EI.tolist()}, "
            f"max_degree={self.max_degree})"
        )
