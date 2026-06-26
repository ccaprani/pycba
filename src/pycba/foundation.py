"""
PyCBA - beam on elastic (Winkler) foundation.

A foundation *member* (PyCBA uses "member" and "span" interchangeably for the
element between two nodes) is modelled as a *super-element*: internally it is
meshed into ``n`` Euler-Bernoulli **sub-elements**, each carrying the standard
consistent Winkler foundation stiffness, and the internal nodes are statically
condensed out so the member still presents the usual two-node (4-DOF) stiffness
to the global assembly.  Applied-load handling and the per-sub-element
member-result recovery are inherited from PyCBA's existing Euler-Bernoulli
machinery, so no per-load-type closed forms are re-derived; accuracy improves
with mesh refinement and the mesh defaults to the foundation characteristic
length.  (This internal meshing mirrors the super-element used by the nonlinear
analysis.)

Supported in this version: *prismatic*, fixed-fixed (default ``eletype``)
members without shear flexibility (``GAv``), carrying UDL, point-load and
partial-UDL loads.  Other combinations raise a clear error.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from scipy import integrate
from .load import LoadUDL, LoadPL, LoadPUDL, LoadMaMb
from .section import SectionEI
from .types import MemberResults


def characteristic_length(EI: float, kf: float) -> float:
    """Winkler characteristic length :math:`\\lambda = (4 EI / k_f)^{1/4}`."""
    return (4.0 * EI / kf) ** 0.25


def auto_subdivisions(
    L: float,
    EI: float,
    kf: float,
    per_wave: int = 8,
    n_min: int = 8,
    n_max: int = 400,
) -> int:
    """
    Number of sub-elements for a foundation member: at least ``per_wave`` per
    characteristic length, clipped to ``[n_min, n_max]``.
    """
    lam = characteristic_length(EI, kf)
    n = int(np.ceil(per_wave * L / lam))
    return int(np.clip(n, n_min, n_max))


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


def _k_found(kf: float, L: float) -> np.ndarray:
    """Consistent Winkler foundation stiffness (cubic Hermite shape functions)."""
    L2 = L * L
    return (kf * L / 420.0) * np.array(
        [
            [156, 22 * L, 54, -13 * L],
            [22 * L, 4 * L2, 13 * L, -3 * L2],
            [54, 13 * L, 156, -22 * L],
            [-13 * L, -3 * L2, -22 * L, 4 * L2],
        ]
    )


def _eb_member_values(L, EI, f, d, loads, npts) -> MemberResults:
    """
    Euler-Bernoulli member-result recovery for one (fixed-fixed) sub-element,
    given its end forces ``f = [Va, Ma, Vb, Mb]``, end displacements
    ``d = [v_i, θ_i, v_j, θ_j]`` and the loads acting on it (sub-element-local).
    Mirrors the prismatic branch of ``BeamResults._member_values``.
    """
    dx = L / npts
    x = np.zeros(npts + 3)
    x[1 : npts + 2] = dx * np.arange(0, npts + 1)
    x[npts + 2] = L

    res = LoadMaMb(i_span=0, Ma=f[1], Mb=f[3]).get_mbr_results(x, L)
    for load in loads:
        res += load.get_mbr_results(x, L)

    curv = res.M / EI
    psi_prov = integrate.cumulative_trapezoid(curv[1:-1], dx=dx, initial=0)
    psi = psi_prov + d[1]
    D = integrate.cumulative_trapezoid(psi, dx=dx, initial=0) + d[0]
    res.R[1:-1] = psi
    res.D[1:-1] = D
    return res


def _concat_results(res_list: List[MemberResults]) -> MemberResults:
    """Concatenate per-sub-element results into a single member result."""
    arrays = [
        np.concatenate([getattr(r, a) for r in res_list])
        for a in ("x", "M", "V", "R", "D")
    ]
    return MemberResults(vals=tuple(arrays))


class FoundationElement:
    """
    Beam-on-elastic-foundation super-element for one prismatic member.

    Parameters
    ----------
    L : float
        Member length.
    EI : float
        Flexural rigidity (prismatic; a ``SectionEI`` raises ``NotImplementedError``).
    kf : float
        Winkler foundation modulus (modulus of subgrade reaction per unit beam
        length: force / length / length).
    n : int, optional
        Number of internal sub-elements.  Defaults to :func:`auto_subdivisions`.
    """

    def __init__(self, L: float, EI, kf: float, n: Optional[int] = None):
        if isinstance(EI, SectionEI):
            raise NotImplementedError(
                "A Winkler foundation currently requires a prismatic (scalar EI) "
                "member; non-prismatic foundation members are not yet supported."
            )
        self.L = float(L)
        self.EI = float(EI)
        self.kf = float(kf)
        self.n = int(n) if n else auto_subdivisions(self.L, self.EI, self.kf)
        self.h = self.L / self.n
        self.ndof = 2 * (self.n + 1)

        ke = _k_ebeam(self.EI, self.h) + _k_found(self.kf, self.h)
        K = np.zeros((self.ndof, self.ndof))
        for e in range(self.n):
            s = 2 * e
            K[s : s + 4, s : s + 4] += ke

        self.r = [0, 1, self.ndof - 2, self.ndof - 1]  # retained (span ends)
        self.c = [i for i in range(self.ndof) if i not in self.r]  # condensed
        r, c = self.r, self.c
        self.Krc = K[np.ix_(r, c)]
        self.Kcr = K[np.ix_(c, r)]
        self.Kcc_inv = np.linalg.inv(K[np.ix_(c, c)])
        # Condensed two-node stiffness presented to the global assembly.
        self.k = K[np.ix_(r, r)] - self.Krc @ self.Kcc_inv @ self.Kcr

    # -- loads ---------------------------------------------------------- #
    def _subelement_loads(self, loads) -> List[list]:
        """Restrict the member's loads onto each sub-element (local coordinates)."""
        n, h, L = self.n, self.h, self.L
        per: List[list] = [[] for _ in range(n)]
        for load in loads:
            if isinstance(load, LoadUDL):
                for e in range(n):
                    per[e].append(LoadUDL(0, load.w))
            elif isinstance(load, LoadPL):
                e = min(int(load.a / h), n - 1)
                per[e].append(LoadPL(0, load.P, load.a - e * h))
            elif isinstance(load, LoadPUDL):
                a0, a1 = load.a, load.a + load.c
                for e in range(n):
                    x0, x1 = e * h, (e + 1) * h
                    lo, hi = max(a0, x0), min(a1, x1)
                    if hi - lo > 1e-12 * L:
                        if np.isclose(lo, x0) and np.isclose(hi, x1):
                            per[e].append(LoadUDL(0, load.w))
                        else:
                            per[e].append(LoadPUDL(0, load.w, lo - x0, hi - lo))
            else:
                raise NotImplementedError(
                    f"Load type {type(load).__name__} is not supported on a "
                    "Winkler foundation member; use UDL, point or partial-UDL loads."
                )
        return per

    def _rfull(self, loads):
        """Full-mesh fixed-end reaction vector and the per-sub-element loads/refs."""
        per = self._subelement_loads(loads)
        Rfull = np.zeros(self.ndof)
        ref_e_list = []
        for e in range(self.n):
            ref_e = np.zeros(4)
            for rl in per[e]:
                ref_e = ref_e + np.asarray(rl.get_ref(self.h, 1), dtype=float)
            Rfull[2 * e : 2 * e + 4] += ref_e
            ref_e_list.append(ref_e)
        return Rfull, per, ref_e_list

    def ref(self, loads) -> np.ndarray:
        """Condensed two-node fixed-end reactions (PyCBA ``get_ref`` convention)."""
        Rfull, _, _ = self._rfull(loads)
        return Rfull[self.r] - self.Krc @ self.Kcc_inv @ Rfull[self.c]

    # -- recovery ------------------------------------------------------- #
    def recover(self, dr, loads, npts: int) -> MemberResults:
        """
        Member load-effect arrays along the member given its end displacements
        ``dr = [v_i, θ_i, v_j, θ_j]`` (from the global solve) and its loads.
        """
        dr = np.asarray(dr, dtype=float)
        Rfull, per, ref_e_list = self._rfull(loads)
        # Internal-node displacements: Kcc dc = Fc - Kcr dr with F = -Rfull.
        dc = self.Kcc_inv @ (-Rfull[self.c] - self.Kcr @ dr)
        d_mesh = np.zeros(self.ndof)
        d_mesh[self.r] = dr
        d_mesh[self.c] = dc

        npts_sub = max(2, int(round(npts / self.n)))
        ke_bend = _k_ebeam(self.EI, self.h)
        res_list = []
        for e in range(self.n):
            d_e = d_mesh[2 * e : 2 * e + 4]
            f_e = ke_bend @ d_e + ref_e_list[e]
            res = _eb_member_values(self.h, self.EI, f_e, d_e, per[e], npts_sub)
            res.x = res.x + e * self.h
            res_list.append(res)
        return _concat_results(res_list)
