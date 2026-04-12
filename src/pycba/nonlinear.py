"""
PyCBA Nonlinear Module — Generalized Clough Model

Incremental nonlinear analysis of continuous beams using the Generalized
Clough model for material nonlinearity (concentrated plasticity).

Each plastic hinge is owned by a single element end, so the adjacent element
retains rotational stiffness at the shared node.  This keeps the global
stiffness matrix non-singular during load redistribution.  True collapse is
detected by a separate rank test that zeros both element ends at every hinged
node.

Reference
---------
McCarthy, L.A. (2012). *Probabilistic Analysis of Indeterminate Highway
Bridges Considering Material Nonlinearity*. MPhil Thesis, Dublin Institute
of Technology.

Li, G.Q. & Li, J.J. (2007). *Advanced Analysis and Design of Steel Frames*.
John Wiley & Sons.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class HingeEvent:
    """Record of a yield or plastic hinge formation event."""
    load_factor: float
    location: float
    node_index: int
    event_type: str  # 'initial_yield' or 'plastic_hinge'


@dataclass
class NonlinearResult:
    """Results container for nonlinear beam analysis."""
    collapse_lambda: float
    collapsed: bool
    hinge_events: list[HingeEvent]
    node_coords: np.ndarray
    final_moments: np.ndarray
    final_R: np.ndarray
    lambda_history: list[float] = field(default_factory=list)
    moment_history: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Element stiffness matrices (4×4)
# ---------------------------------------------------------------------------

def _k_FF(EI: float, L: float) -> np.ndarray:
    """Fixed-fixed element stiffness. [k_e in Clough notation]"""
    L2, L3 = L * L, L * L * L
    c = EI / L3
    return c * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L2, -6*L, 2*L2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L2, -6*L, 4*L2],
    ])


def _k_PF(EI: float, L: float) -> np.ndarray:
    """Pinned-fixed (hinge at LEFT). [k_1 in Clough notation]"""
    L2, L3 = L * L, L * L * L
    c = EI / L3
    return c * np.array([
        [3, 0, -3, 3*L],
        [0, 0, 0, 0],
        [-3, 0, 3, -3*L],
        [3*L, 0, -3*L, 3*L2],
    ])


def _k_FP(EI: float, L: float) -> np.ndarray:
    """Fixed-pinned (hinge at RIGHT). [k_2 in Clough notation]"""
    L2, L3 = L * L, L * L * L
    c = EI / L3
    return c * np.array([
        [3, 3*L, -3, 0],
        [3*L, 3*L2, -3*L, 0],
        [-3, -3*L, 3, 0],
        [0, 0, 0, 0],
    ])


# ---------------------------------------------------------------------------
#  Mesh generation
# ---------------------------------------------------------------------------

def _build_mesh(span_L, span_R, mesh_size):
    n_spans = len(span_L)
    nodes = [0.0]
    elem_L_list = []
    n2s = [0]

    for i in range(n_spans):
        L = span_L[i]
        ne = max(1, round(L / mesh_size))
        h = L / ne
        x0 = nodes[-1]
        for j in range(ne):
            elem_L_list.append(h)
            nodes.append(x0 + (j + 1) * h)
            n2s.append(i)

    node_coords = np.array(nodes)
    elem_lengths = np.array(elem_L_list)
    node_to_span = np.array(n2s)

    bc_dofs = []
    support_x = np.concatenate(([0.0], np.cumsum(span_L)))
    for i_sup in range(n_spans + 1):
        mn = int(np.argmin(np.abs(node_coords - support_x[i_sup])))
        if span_R[2 * i_sup] == -1:
            bc_dofs.append(2 * mn)
        if span_R[2 * i_sup + 1] == -1:
            bc_dofs.append(2 * mn + 1)

    return node_coords, elem_lengths, bc_dofs, node_to_span


# ---------------------------------------------------------------------------
#  Main analysis class
# ---------------------------------------------------------------------------

class NonlinearBeamAnalysis:
    """
    Nonlinear continuous beam analysis using the Generalized Clough model.

    Parameters
    ----------
    L : array_like
        Span lengths.
    EI : float or array_like
        Flexural rigidity (per span or scalar).
    R : array_like
        Restraint vector in pycba convention.
    Mp : float or array_like
        Plastic moment capacity (per span or scalar).
    My : float or array_like, optional
        Yield moment capacity. Defaults to Mp / 1.15.
    q : float
        Strain-hardening ratio (0 = elastic-perfectly-plastic).
    mesh_size : float
        Target element length for the internal mesh.
    """

    def __init__(
        self,
        L: Union[list, np.ndarray],
        EI: Union[float, list, np.ndarray],
        R: Union[list, np.ndarray],
        Mp: Union[float, list, np.ndarray],
        My: Optional[Union[float, list, np.ndarray]] = None,
        q: float = 0.0,
        mesh_size: float = 0.5,
    ):
        self.span_L = np.atleast_1d(np.asarray(L, dtype=float))
        self.n_spans = len(self.span_L)
        self.total_length = float(np.sum(self.span_L))
        self.span_R = list(R)
        self.q = q

        _as = lambda v, n: np.full(n, float(v)) if isinstance(v, (int, float)) else np.atleast_1d(np.asarray(v, dtype=float))
        self.span_EI = _as(EI, self.n_spans)
        self.span_Mp = _as(Mp, self.n_spans)
        self.span_My = (self.span_Mp / 1.15) if My is None else _as(My, self.n_spans)

        # Build mesh
        self.node_coords, self.elem_L, self.bc_dofs, self.node_to_span = (
            _build_mesh(self.span_L, self.span_R, mesh_size)
        )
        self.n_nodes = len(self.node_coords)
        self.n_elem = len(self.elem_L)
        self.n_dof = 2 * self.n_nodes

        # Per-element / per-node properties
        self.elem_EI = self.span_EI[self.node_to_span[:self.n_elem]]
        self.node_Mp = self.span_Mp[self.node_to_span]
        self.node_My = self.span_My[self.node_to_span]
        self.node_gamma_y = self.node_My / self.node_Mp

        # ---- Precompute element stiffness matrices (n_elem, 4, 4) ----
        self._ke_all = np.array([_k_FF(self.elem_EI[i], self.elem_L[i]) for i in range(self.n_elem)])
        self._k1_all = np.array([_k_PF(self.elem_EI[i], self.elem_L[i]) for i in range(self.n_elem)])
        self._k2_all = np.array([_k_FP(self.elem_EI[i], self.elem_L[i]) for i in range(self.n_elem)])

        # Precompute elastic global K with BCs applied (for unloading)
        self._K_elastic_bc = self._apply_bc_inplace(self._assemble_elastic())

        # BC mask for fast BC application
        self._bc_set = set(self.bc_dofs)

    # ---- Assembly (vectorized) ----

    def _assemble_elastic(self) -> np.ndarray:
        K = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_elem):
            d = 2 * i
            K[d:d+4, d:d+4] += self._ke_all[i]
        return K

    def _assemble_from_elem_R(self, R_elem: np.ndarray) -> np.ndarray:
        K = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_elem):
            R1, R2 = R_elem[i, 0], R_elem[i, 1]
            if R1 >= R2:
                k = R2 * self._ke_all[i] + (R1 - R2) * self._k2_all[i]
            else:
                k = R1 * self._ke_all[i] + (R2 - R1) * self._k1_all[i]
            d = 2 * i
            K[d:d+4, d:d+4] += k
        return K

    def _apply_bc_inplace(self, K: np.ndarray) -> np.ndarray:
        """Apply BCs to K (modifies and returns K)."""
        for dof in self.bc_dofs:
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
        return K

    def _apply_bc(self, K, F):
        K = K.copy()
        F = F.copy()
        for dof in self.bc_dofs:
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
            F[dof] = 0.0
        return K, F

    # ---- Force vectors ----

    def _build_force_vector(self, LM: list) -> np.ndarray:
        F = np.zeros(self.n_dof)
        span_starts = np.concatenate(([0.0], np.cumsum(self.span_L)))
        for ld in LM:
            i_span = int(ld[0]) - 1
            lt = int(ld[1])
            val = float(ld[2])
            if lt == 1:  # UDL
                xs, xe = span_starts[i_span], span_starts[i_span + 1]
                for ie in range(self.n_elem):
                    if self.node_coords[ie+1] <= xs + 1e-10 or self.node_coords[ie] >= xe - 1e-10:
                        continue
                    h = self.elem_L[ie]
                    F[2*ie] -= val * h / 2
                    F[2*(ie+1)] -= val * h / 2
            elif lt == 2:  # Point load
                self._add_point_load(F, val, span_starts[i_span] + float(ld[3]))
        return F

    def _add_point_load(self, F: np.ndarray, P: float, x: float):
        if x < 0 or x > self.total_length:
            return
        ni = int(np.argmin(np.abs(self.node_coords - x)))
        if abs(self.node_coords[ni] - x) < 1e-10:
            F[2 * ni] -= P
        else:
            nL = ni if self.node_coords[ni] < x else ni - 1
            nR = nL + 1
            h = self.node_coords[nR] - self.node_coords[nL]
            a = x - self.node_coords[nL]
            b = h - a
            F[2*nL] -= P * b**2 * (3*a + b) / h**3
            F[2*nL+1] -= P * a * b**2 / h**2
            F[2*nR] -= P * a**2 * (a + 3*b) / h**3
            F[2*nR+1] += P * a**2 * b / h**2

    def _point_load_vector(self, P: float, x: float) -> np.ndarray:
        F = np.zeros(self.n_dof)
        self._add_point_load(F, P, x)
        return F

    # ---- Moment extraction (vectorized) ----

    def _extract_moments(self, u: np.ndarray, R_elem: np.ndarray) -> np.ndarray:
        moments = np.zeros(self.n_nodes)
        for i in range(self.n_elem):
            d = 2 * i
            R1, R2 = R_elem[i, 0], R_elem[i, 1]
            if R1 >= R2:
                k = R2 * self._ke_all[i] + (R1 - R2) * self._k2_all[i]
            else:
                k = R1 * self._ke_all[i] + (R2 - R1) * self._k1_all[i]
            fe = k @ u[d:d+4]
            if i == 0:
                moments[0] = fe[1]
            moments[i+1] = fe[3]
        return moments

    def _extract_moments_elastic(self, u: np.ndarray) -> np.ndarray:
        moments = np.zeros(self.n_nodes)
        for i in range(self.n_elem):
            d = 2 * i
            fe = self._ke_all[i] @ u[d:d+4]
            if i == 0:
                moments[0] = fe[1]
            moments[i+1] = fe[3]
        return moments

    # ---- R update (vectorized) ----

    def _update_R(self, moments, gamma_max, R_elem, yielded, hinged, hinge_events, pos_label):
        """Update R parameters. Returns True if a new plastic hinge formed."""
        gamma = np.abs(moments) / self.node_Mp
        new_hinge = False

        for j in range(self.n_nodes):
            ie = max(0, j - 1)  # owning element
            end = 1 if j > 0 else 0

            if gamma[j] >= gamma_max[j]:
                gamma_max[j] = gamma[j]
                g, gy = gamma[j], self.node_gamma_y[j]
                if g <= gy:
                    R_elem[ie, end] = 1.0
                elif g >= 1.0:
                    R_elem[ie, end] = self.q
                else:
                    R_elem[ie, end] = 1.0 - (g - gy) / (1.0 - gy) * (1.0 - self.q)
            else:
                R_elem[ie, end] = 1.0

            if gamma[j] >= self.node_gamma_y[j] and j not in yielded:
                yielded.add(j)
                hinge_events.append(HingeEvent(pos_label, self.node_coords[j], j, "initial_yield"))
            if gamma[j] >= 1.0 and j not in hinged:
                hinged.add(j)
                hinge_events.append(HingeEvent(pos_label, self.node_coords[j], j, "plastic_hinge"))
                new_hinge = True
        return new_hinge

    # ---- Mechanism test ----

    def _cluster_hinges(self, hinged_nodes, tol=0.0):
        if not hinged_nodes:
            return []
        if tol <= 0.0:
            tol = 2.0 * float(self.elem_L.min())
        sn = sorted(hinged_nodes)
        clusters = [[sn[0]]]
        for n in sn[1:]:
            if self.node_coords[n] - self.node_coords[clusters[-1][-1]] <= tol:
                clusters[-1].append(n)
            else:
                clusters.append([n])
        return [c[len(c)//2] for c in clusters]

    def _is_mechanism(self, hinged_nodes):
        reps = self._cluster_hinges(hinged_nodes)
        if len(reps) < 2:
            return False
        R_test = np.ones((self.n_elem, 2))
        for j in reps:
            if j > 0:
                R_test[j-1, 1] = 0.0
            if j < self.n_elem:
                R_test[j, 0] = 0.0
        K_test = self._assemble_from_elem_R(R_test)
        self._apply_bc_inplace(K_test)
        return np.linalg.matrix_rank(K_test, tol=1e-6) < self.n_dof

    # ---- Static incremental analysis ----

    def analyze(self, LM, lambda_max=20.0, record_every=0, max_steps=50000):
        F_ref = self._build_force_vector(LM)
        R_elem = np.ones((self.n_elem, 2))
        moments = np.zeros(self.n_nodes)
        gamma_max = np.zeros(self.n_nodes)

        lam = 0.0
        hinge_events, lam_hist, mom_hist = [], [], []
        yielded, hinged = set(), set()
        step = 0

        while lam < lambda_max and step < max_steps:
            Rmin = float(R_elem.min())
            dl = 0.1 if Rmin > 0.5 else (0.01 if Rmin > 0.25 else 0.001)
            dl = min(dl, lambda_max - lam)
            lam += dl
            step += 1

            K = self._assemble_from_elem_R(R_elem)
            K_bc, F_bc = self._apply_bc(K, dl * F_ref)
            try:
                u = np.linalg.solve(K_bc, F_bc)
            except np.linalg.LinAlgError:
                return NonlinearResult(lam - dl, True, hinge_events, self.node_coords, moments, R_elem.copy(), lam_hist, mom_hist)

            moments += self._extract_moments(u, R_elem)
            new_h = self._update_R(moments, gamma_max, R_elem, yielded, hinged, hinge_events, lam)

            if new_h and self._is_mechanism(hinged):
                return NonlinearResult(lam, True, hinge_events, self.node_coords, moments, R_elem.copy(), lam_hist, mom_hist)

            if record_every > 0 and step % record_every == 0:
                lam_hist.append(lam)
                mom_hist.append(moments.copy())

        return NonlinearResult(lam, False, hinge_events, self.node_coords, moments, R_elem.copy(), lam_hist, mom_hist)

    # ---- Vehicle force vector ----

    def _vehicle_force_vector(self, axle_weights, axle_coords, pos: float) -> np.ndarray:
        """
        Build force vector for a multi-axle vehicle at position *pos*.

        Parameters
        ----------
        axle_weights : (n_axles,) array of axle loads (kN).
        axle_coords : (n_axles,) array of axle positions relative to the
            first axle (from :class:`pycba.Vehicle.axle_coords`).
        pos : global coordinate of the first (leading) axle.
        """
        F = np.zeros(self.n_dof)
        for w, ac in zip(axle_weights, axle_coords):
            x = pos - ac  # axle_coords[0]=0, subsequent trail behind
            if 0 <= x <= self.total_length:
                self._add_point_load(F, w, x)
        return F

    # ---- Moving load analysis ----

    def analyze_moving(
        self,
        P=None,
        vehicle=None,
        step=0.5,
        n_sub=5,
        record_every=0,
    ):
        """
        Moving-load nonlinear analysis.

        A vehicle traverses the beam from left to right.  At each position
        step the load is transferred via paired elastic-unload / nonlinear-
        reload sub-increments.

        The vehicle can be specified as:

        * **Single axle** — pass ``P`` (float, kN).
        * **Multi-axle** — pass a :class:`pycba.Vehicle` object.

        Parameters
        ----------
        P : float, optional
            Single point load magnitude.
        vehicle : :class:`pycba.Vehicle`, optional
            A pycba Vehicle object with axle weights and spacings.
        step : float
            Distance the front axle moves per position step.
        n_sub : int
            Sub-increments per position step for the unload/reload transfer.
        record_every : int
            Store moment snapshots every *n* position steps.

        Returns
        -------
        NonlinearResult
            ``collapse_lambda`` is the front-axle position (m) at collapse.
        """
        # Resolve vehicle definition
        if vehicle is not None:
            aw = vehicle.axw
            ax_coords = vehicle.axle_coords
            vehicle_L = vehicle.L
        elif P is not None:
            aw = np.array([float(P)])
            ax_coords = np.array([0.0])
            vehicle_L = 0.0
        else:
            raise ValueError("Specify either P or vehicle")

        # Front axle travels from 0 to total_length + vehicle_L
        # (so the rear axle clears the bridge)
        x_end = self.total_length + vehicle_L
        positions = np.arange(0, x_end + step / 2, step)
        d_frac = 1.0 / n_sub

        R_elem = np.ones((self.n_elem, 2))
        moments = np.zeros(self.n_nodes)
        gamma_max = np.zeros(self.n_nodes)
        hinge_events, lam_hist, mom_hist = [], [], []
        yielded, hinged = set(), set()

        F_prev = np.zeros(self.n_dof)
        K_el_bc = self._K_elastic_bc

        for i_pos, x_front in enumerate(positions):
            F_cur = self._vehicle_force_vector(aw, ax_coords, x_front)

            if i_pos == 0:
                for _ in range(n_sub):
                    self._update_R(moments, gamma_max, R_elem, yielded, hinged, hinge_events, x_front)
                    K = self._assemble_from_elem_R(R_elem)
                    K_bc, F_bc = self._apply_bc(K, d_frac * F_cur)
                    try:
                        u = np.linalg.solve(K_bc, F_bc)
                    except np.linalg.LinAlgError:
                        return self._mr(x_front, True, hinge_events, moments, R_elem, lam_hist, mom_hist)
                    moments += self._extract_moments(u, R_elem)
            else:
                for _ in range(n_sub):
                    # Elastic unload
                    F_ul = -d_frac * F_prev.copy()
                    for dof in self.bc_dofs:
                        F_ul[dof] = 0.0
                    u_ul = np.linalg.solve(K_el_bc, F_ul)
                    moments += self._extract_moments_elastic(u_ul)

                    self._update_R(moments, gamma_max, R_elem, yielded, hinged, hinge_events, x_front)

                    K = self._assemble_from_elem_R(R_elem)
                    K_bc, F_bc = self._apply_bc(K, d_frac * F_cur)
                    try:
                        u = np.linalg.solve(K_bc, F_bc)
                    except np.linalg.LinAlgError:
                        return self._mr(x_front, True, hinge_events, moments, R_elem, lam_hist, mom_hist)
                    moments += self._extract_moments(u, R_elem)

                nh = self._update_R(moments, gamma_max, R_elem, yielded, hinged, hinge_events, x_front)
                if nh and self._is_mechanism(hinged):
                    return self._mr(x_front, True, hinge_events, moments, R_elem, lam_hist, mom_hist)

            F_prev = F_cur
            if record_every > 0 and i_pos % record_every == 0:
                lam_hist.append(x_front)
                mom_hist.append(moments.copy())

        return self._mr(positions[-1], False, hinge_events, moments, R_elem, lam_hist, mom_hist)

    def _mr(self, x, coll, he, mom, R, lh, mh):
        return NonlinearResult(x, coll, he, self.node_coords, mom, R.copy(), lh, mh)
