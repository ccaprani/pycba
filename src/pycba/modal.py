"""
PyCBA - free-vibration (modal) analysis.

Assembles a consistent mass matrix alongside the (Euler-Bernoulli) stiffness
matrix and solves the generalized eigenproblem

.. math:: \\mathbf{K}\\,\\boldsymbol{\\phi} = \\omega^2\\,\\mathbf{M}\\,\\boldsymbol{\\phi}

for the natural circular frequencies :math:`\\omega` and mode shapes
:math:`\\boldsymbol{\\phi}`.  Each span is internally refined into ``nseg``
Euler-Bernoulli sub-elements so the frequencies and mode shapes are accurate
(a single element per span is far too coarse).  Supports (including elastic
springs) are applied at the original span nodes.

This first version covers prismatic, fixed-fixed spans (no moment releases,
``GAv`` or non-prismatic ``EI``); the mass is given per span as a mass per unit
length.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence
import numpy as np
from scipy.linalg import eigh
from .section import SectionEI


def _k_ebeam(EI: float, L: float) -> np.ndarray:
    """Prismatic Euler-Bernoulli element stiffness (PyCBA DOF order)."""
    L2 = L * L
    L3 = L2 * L
    return EI * np.array(
        [
            [12 / L3, 6 / L2, -12 / L3, 6 / L2],
            [6 / L2, 4 / L, -6 / L2, 2 / L],
            [-12 / L3, -6 / L2, 12 / L3, -6 / L2],
            [6 / L2, 2 / L, -6 / L2, 4 / L],
        ]
    )


def _m_consistent(m_bar: float, L: float) -> np.ndarray:
    """Consistent mass matrix of an Euler-Bernoulli element (mass/length m̄)."""
    L2 = L * L
    return (m_bar * L / 420.0) * np.array(
        [
            [156, 22 * L, 54, -13 * L],
            [22 * L, 4 * L2, 13 * L, -3 * L2],
            [54, 13 * L, 156, -22 * L],
            [-13 * L, -3 * L2, -22 * L, 4 * L2],
        ]
    )


class ModalResults:
    """
    Natural frequencies and mode shapes from a free-vibration analysis.

    Attributes
    ----------
    omega : np.ndarray
        Natural circular frequencies (rad/s), ascending.
    f : np.ndarray
        Natural frequencies (Hz) = ``omega / (2 pi)``.
    periods : np.ndarray
        Natural periods (s) = ``1 / f``.
    x : np.ndarray
        Coordinates of the refined-mesh nodes along the beam.
    phi : np.ndarray
        Mode-shape matrix, shape ``(2 * n_nodes, n_modes)`` over the full DOF
        set (restrained DOFs zero); column ``i`` is mode ``i``.
    n_modes : int
        Number of modes returned.
    """

    def __init__(self, omega, phi, x):
        self.omega = np.asarray(omega)
        self.f = self.omega / (2 * np.pi)
        with np.errstate(divide="ignore"):
            self.periods = np.where(self.f > 0, 1.0 / self.f, np.inf)
        self.phi = np.asarray(phi)
        self.x = np.asarray(x)
        self.n_modes = len(self.omega)

    def mode_shape(self, mode: int = 0):
        """
        Return ``(x, v)`` for a mode: the vertical mode shape sampled at the
        refined-mesh nodes, normalised to unit maximum amplitude.
        """
        v = self.phi[0::2, mode]
        vmax = np.max(np.abs(v))
        if vmax > 0:
            v = v / vmax
        # Orient consistently (largest lobe positive).
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        return self.x, v

    def plot(self, modes: Union[int, Sequence[int]] = 0, ax=None, units=None):
        """Plot one or several mode shapes."""
        import matplotlib.pyplot as plt
        from .units import resolve

        us = resolve(units)
        if isinstance(modes, int):
            modes = [modes]
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot([self.x[0], self.x[-1]], [0, 0], "k", lw=1)
        for m in modes:
            x, v = self.mode_shape(m)
            ax.plot(x, v, label=f"Mode {m + 1}: {self.f[m]:.3g} Hz")
        ax.set_xlabel(us.distance_axis)
        ax.set_yticks([])
        ax.set_ylabel("Mode shape")
        ax.legend(loc="best", fontsize=8)
        return ax

    def __repr__(self):
        fs = ", ".join(f"{v:.4g}" for v in self.f[: min(5, self.n_modes)])
        more = ", ..." if self.n_modes > 5 else ""
        return f"ModalResults({self.n_modes} modes; f [Hz] = {fs}{more})"


def _broadcast_mass(mass, n_spans: int) -> list:
    """Normalise the mass-per-length argument to one entry per span."""
    if isinstance(mass, (float, int)):
        return [float(mass)] * n_spans
    mass = list(mass)
    if len(mass) != n_spans:
        raise ValueError("Define mass for each span (or pass a scalar).")
    return [float(m) for m in mass]


def solve_modal(beam, mass, n_modes: int = 10, nseg: int = 12) -> ModalResults:
    """
    Free-vibration analysis of a :class:`pycba.beam.Beam`.

    Parameters
    ----------
    beam : Beam
        The beam (prismatic, fixed-fixed spans without ``GAv``).
    mass : float or sequence
        Mass per unit length, a scalar for every span or one value per span.
    n_modes : int, optional
        Number of (lowest) modes to return. The default is 10.
    nseg : int, optional
        Number of sub-elements per span for the refined mesh. The default is 12.

    Returns
    -------
    ModalResults
    """
    N = beam.no_spans
    for i in range(N):
        if isinstance(beam.mbr_EIs[i], SectionEI):
            raise NotImplementedError("Modal analysis supports prismatic spans only.")
        if beam.mbr_GAv[i] is not None:
            raise NotImplementedError(
                "Modal analysis does not yet support shear-deformable (GAv) spans."
            )
        if int(np.asarray(beam.mbr_eletype[i]).item()) != 1:
            raise NotImplementedError(
                "Modal analysis supports fixed-fixed spans only (no moment releases)."
            )

    mass_list = _broadcast_mass(mass, N)
    n_nodes = N * nseg + 1
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    node_x = np.zeros(n_nodes)

    for i in range(N):
        EIi = float(beam.mbr_EIs[i])
        Li = beam.mbr_lengths[i]
        h = Li / nseg
        ke = _k_ebeam(EIi, h)
        me = _m_consistent(mass_list[i], h)
        base = i * nseg
        for s in range(nseg):
            n0 = base + s
            node_x[n0 + 1] = node_x[n0] + h
            sdof = 2 * n0
            K[sdof : sdof + 4, sdof : sdof + 4] += ke
            M[sdof : sdof + 4, sdof : sdof + 4] += me

    # Supports at the original span nodes: remove fixed DOFs; add spring stiffness.
    restraints = np.asarray(beam.restraints)
    fixed = set()
    for j in range(N + 1):
        gnode = j * nseg
        for d in range(2):
            gdof = 2 * gnode + d
            r = restraints[2 * j + d]
            if r == -1:
                fixed.add(gdof)
            elif r > 0:
                K[gdof, gdof] += r

    free = [i for i in range(ndof) if i not in fixed]
    Kff = K[np.ix_(free, free)]
    Mff = M[np.ix_(free, free)]

    w2, vecs = eigh(Kff, Mff)
    w2 = np.clip(w2, 0.0, None)
    omega = np.sqrt(w2)
    nm = int(min(n_modes, len(omega)))

    phi = np.zeros((ndof, nm))
    phi[np.ix_(free, range(nm))] = vecs[:, :nm]
    return ModalResults(omega[:nm], phi, node_x)
