"""
PyCBA Nonlinear Module — Generalized Clough Model

Incremental nonlinear analysis of continuous beams using the Generalized
Clough model for material nonlinearity (concentrated plasticity).

Tracks the spread of plasticity through a structure, detecting three limit
states: initial yield, first plastic hinge, and collapse mechanism formation.

Each plastic hinge is owned by a single element end, so the adjacent element
retains rotational stiffness at the shared node.  This keeps the global
stiffness matrix non-singular during load redistribution and avoids spurious
mechanism detection.  True collapse is detected by a separate rank test that
temporarily zeros both element ends at every hinged node.

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
#  Element stiffness matrices
# ---------------------------------------------------------------------------

def _k_FF(EI: float, L: float) -> np.ndarray:
    """Fixed-fixed beam element stiffness (4x4). [k_e in Clough notation]"""
    L2 = L * L
    L3 = L2 * L
    c = EI / L3
    return c * np.array(
        [
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L2, -6 * L, 2 * L2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L2, -6 * L, 4 * L2],
        ]
    )


def _k_PF(EI: float, L: float) -> np.ndarray:
    """Pinned-fixed element (hinge at LEFT node). [k_1 in Clough notation]"""
    L2 = L * L
    L3 = L2 * L
    c = EI / L3
    return c * np.array(
        [
            [3, 0, -3, 3 * L],
            [0, 0, 0, 0],
            [-3, 0, 3, -3 * L],
            [3 * L, 0, -3 * L, 3 * L2],
        ]
    )


def _k_FP(EI: float, L: float) -> np.ndarray:
    """Fixed-pinned element (hinge at RIGHT node). [k_2 in Clough notation]"""
    L2 = L * L
    L3 = L2 * L
    c = EI / L3
    return c * np.array(
        [
            [3, 3 * L, -3, 0],
            [3 * L, 3 * L2, -3 * L, 0],
            [-3, -3 * L, 3, 0],
            [0, 0, 0, 0],
        ]
    )


def _clough_stiffness(EI: float, L: float, R1: float, R2: float) -> np.ndarray:
    """
    Generalized Clough model element stiffness.

    Parameters
    ----------
    EI, L : float
        Flexural rigidity and element length.
    R1 : float
        Force recovery parameter at the LEFT end of this element.
    R2 : float
        Force recovery parameter at the RIGHT end of this element.

    Notes
    -----
    Equations 4.12-4.13 from McCarthy (2012), after Li et al. (2007).
    """
    ke = _k_FF(EI, L)
    if R1 >= R2:
        k2 = _k_FP(EI, L)  # hinge at right
        return R2 * ke + (R1 - R2) * k2
    else:
        k1 = _k_PF(EI, L)  # hinge at left
        return R1 * ke + (R2 - R1) * k1


# ---------------------------------------------------------------------------
#  Mesh generation
# ---------------------------------------------------------------------------

def _build_mesh(
    span_L: np.ndarray,
    span_R: list,
    mesh_size: float,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    """
    Build a uniform finite-element mesh from span-level definitions.

    Returns
    -------
    node_coords, elem_lengths, bc_dofs, node_to_span
    """
    n_spans = len(span_L)
    node_coords_list = [0.0]
    elem_lengths_list = []
    node_to_span_list = [0]

    for i_span in range(n_spans):
        L = span_L[i_span]
        n_elem = max(1, round(L / mesh_size))
        h = L / n_elem
        x_start = node_coords_list[-1]
        for j in range(n_elem):
            elem_lengths_list.append(h)
            node_coords_list.append(x_start + (j + 1) * h)
            node_to_span_list.append(i_span)

    node_coords = np.array(node_coords_list)
    elem_lengths = np.array(elem_lengths_list)
    node_to_span = np.array(node_to_span_list)

    bc_dofs = []
    support_x = np.concatenate(([0.0], np.cumsum(span_L)))
    for i_support in range(n_spans + 1):
        x_sup = support_x[i_support]
        mesh_node = int(np.argmin(np.abs(node_coords - x_sup)))
        dof_v = 2 * mesh_node
        dof_r = 2 * mesh_node + 1
        if span_R[2 * i_support] == -1:
            bc_dofs.append(dof_v)
        if span_R[2 * i_support + 1] == -1:
            bc_dofs.append(dof_r)

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

        if isinstance(EI, (int, float)):
            self.span_EI = np.full(self.n_spans, float(EI))
        else:
            self.span_EI = np.atleast_1d(np.asarray(EI, dtype=float))

        if isinstance(Mp, (int, float)):
            self.span_Mp = np.full(self.n_spans, float(Mp))
        else:
            self.span_Mp = np.atleast_1d(np.asarray(Mp, dtype=float))

        if My is None:
            self.span_My = self.span_Mp / 1.15
        elif isinstance(My, (int, float)):
            self.span_My = np.full(self.n_spans, float(My))
        else:
            self.span_My = np.atleast_1d(np.asarray(My, dtype=float))

        # Build internal mesh
        self.node_coords, self.elem_L, self.bc_dofs, self.node_to_span = (
            _build_mesh(self.span_L, self.span_R, mesh_size)
        )
        self.n_nodes = len(self.node_coords)
        self.n_elem = len(self.elem_L)
        self.n_dof = 2 * self.n_nodes

        # Per-element properties
        self.elem_EI = np.array(
            [self.span_EI[self.node_to_span[i]] for i in range(self.n_elem)]
        )
        # Per-node capacity
        self.node_Mp = np.array(
            [self.span_Mp[self.node_to_span[i]] for i in range(self.n_nodes)]
        )
        self.node_My = np.array(
            [self.span_My[self.node_to_span[i]] for i in range(self.n_nodes)]
        )

    # ---- Force recovery parameter ----

    def _force_recovery(self, gamma: float, gamma_y: float) -> float:
        """Eq. 4.8-4.10 of McCarthy (2012)."""
        if gamma <= gamma_y:
            return 1.0
        elif gamma >= 1.0:
            return self.q
        else:
            return 1.0 - (gamma - gamma_y) / (1.0 - gamma_y) * (1.0 - self.q)

    # ---- Hinge ownership ----

    def _owning_elem_end(self, node: int) -> tuple[int, int]:
        """
        Return (element_index, end) that owns a hinge at *node*.

        Convention: assign hinge to the LEFT element's right end.
        For node 0, assign to element 0's left end.

        Returns
        -------
        (elem_idx, end) where end is 0 (left) or 1 (right).
        """
        if node == 0:
            return (0, 0)
        else:
            return (node - 1, 1)

    # ---- Assembly & BCs ----

    def _assemble_from_elem_R(self, R_elem: np.ndarray) -> np.ndarray:
        """
        Assemble global stiffness from per-element-end R values.

        Parameters
        ----------
        R_elem : (n_elem, 2) array — R_left, R_right for each element.
        """
        K = np.zeros((self.n_dof, self.n_dof))
        for i in range(self.n_elem):
            k = _clough_stiffness(
                self.elem_EI[i], self.elem_L[i], R_elem[i, 0], R_elem[i, 1]
            )
            d = 2 * i
            K[d : d + 4, d : d + 4] += k
        return K

    def _apply_bc(
        self, K: np.ndarray, F: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply boundary conditions via direct elimination."""
        K = K.copy()
        F = F.copy()
        for dof in self.bc_dofs:
            F -= K[:, dof] * 0.0
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
            F[dof] = 0.0
        return K, F

    # ---- Force vector from load matrix ----

    def _build_force_vector(self, LM: list) -> np.ndarray:
        """
        Build the reference nodal force vector from a pycba-style load matrix.

        Load types supported:
          1 — UDL (intensity w over full span)
          2 — Point load P at distance *a* from span start
        """
        F = np.zeros(self.n_dof)
        span_starts = np.concatenate(([0.0], np.cumsum(self.span_L)))

        for load_desc in LM:
            i_span = int(load_desc[0]) - 1
            load_type = int(load_desc[1])
            value = float(load_desc[2])

            if load_type == 1:
                w = value
                x_start = span_starts[i_span]
                x_end = span_starts[i_span + 1]
                for ie in range(self.n_elem):
                    x_L = self.node_coords[ie]
                    x_R = self.node_coords[ie + 1]
                    if x_R <= x_start + 1e-10 or x_L >= x_end - 1e-10:
                        continue
                    h = self.elem_L[ie]
                    F[2 * ie] -= w * h / 2
                    F[2 * (ie + 1)] -= w * h / 2

            elif load_type == 2:
                P = value
                a = float(load_desc[3])
                x_load = span_starts[i_span] + a
                node_idx = int(np.argmin(np.abs(self.node_coords - x_load)))
                dist = abs(self.node_coords[node_idx] - x_load)

                if dist < 1e-10:
                    F[2 * node_idx] -= P
                else:
                    if self.node_coords[node_idx] < x_load:
                        n_left, n_right = node_idx, node_idx + 1
                    else:
                        n_left, n_right = node_idx - 1, node_idx
                    h = self.node_coords[n_right] - self.node_coords[n_left]
                    a_loc = x_load - self.node_coords[n_left]
                    b_loc = h - a_loc
                    F[2 * n_left] -= P * b_loc**2 * (3 * a_loc + b_loc) / h**3
                    F[2 * n_left + 1] -= P * a_loc * b_loc**2 / h**2
                    F[2 * n_right] -= P * a_loc**2 * (a_loc + 3 * b_loc) / h**3
                    F[2 * n_right + 1] += P * a_loc**2 * b_loc / h**2

        return F

    # ---- Moment extraction ----

    def _extract_moments(self, u: np.ndarray, R_elem: np.ndarray) -> np.ndarray:
        """Extract bending-moment increments at each node from displacements."""
        moments = np.zeros(self.n_nodes)
        for i in range(self.n_elem):
            d = 2 * i
            u_e = u[d : d + 4]
            k = _clough_stiffness(
                self.elem_EI[i], self.elem_L[i], R_elem[i, 0], R_elem[i, 1]
            )
            f_e = k @ u_e
            if i == 0:
                moments[0] = f_e[1]
            moments[i + 1] = f_e[3]
        return moments

    # ---- Point load at arbitrary global position ----

    def _point_load_vector(self, P: float, x_pos: float) -> np.ndarray:
        """Build force vector for a single point load P at global coordinate x_pos."""
        F = np.zeros(self.n_dof)
        if x_pos < 0 or x_pos > self.total_length:
            return F

        node_idx = int(np.argmin(np.abs(self.node_coords - x_pos)))
        dist = abs(self.node_coords[node_idx] - x_pos)

        if dist < 1e-10:
            F[2 * node_idx] -= P
        else:
            if self.node_coords[node_idx] < x_pos:
                n_left, n_right = node_idx, node_idx + 1
            else:
                n_left, n_right = node_idx - 1, node_idx
            h = self.node_coords[n_right] - self.node_coords[n_left]
            a = x_pos - self.node_coords[n_left]
            b = h - a
            F[2 * n_left] -= P * b**2 * (3 * a + b) / h**3
            F[2 * n_left + 1] -= P * a * b**2 / h**2
            F[2 * n_right] -= P * a**2 * (a + 3 * b) / h**3
            F[2 * n_right + 1] += P * a**2 * b / h**2
        return F

    # ---- R update helper ----

    def _update_R_state(
        self,
        moments: np.ndarray,
        gamma_max: np.ndarray,
        R_elem: np.ndarray,
        yielded: set,
        hinged: set,
        hinge_events: list,
        load_position: float,
    ) -> bool:
        """
        Update force recovery parameters and record hinge events.

        Returns True if a new plastic hinge formed this step.
        """
        new_hinge = False
        for j in range(self.n_nodes):
            Mp_j = self.node_Mp[j]
            My_j = self.node_My[j]
            gamma_y = My_j / Mp_j
            gamma = abs(moments[j]) / Mp_j

            ie, end = self._owning_elem_end(j)

            if gamma >= gamma_max[j]:
                gamma_max[j] = gamma
                R_elem[ie, end] = self._force_recovery(gamma, gamma_y)
            else:
                R_elem[ie, end] = 1.0

            if gamma >= gamma_y and j not in yielded:
                yielded.add(j)
                hinge_events.append(
                    HingeEvent(load_position, self.node_coords[j], j, "initial_yield")
                )
            if gamma >= 1.0 and j not in hinged:
                hinged.add(j)
                hinge_events.append(
                    HingeEvent(load_position, self.node_coords[j], j, "plastic_hinge")
                )
                new_hinge = True
        return new_hinge

    # ---- Mechanism test ----

    def _cluster_hinges(self, hinged_nodes: set, tol: float = 0.0) -> list[int]:
        """
        Group closely-spaced hinged nodes into distinct hinge locations.

        Nodes within *tol* metres of each other are treated as a single
        plastic zone (one mechanism hinge).  Returns the representative
        node (centre of each cluster).

        If *tol* is 0, defaults to twice the minimum element length in
        the mesh.
        """
        if not hinged_nodes:
            return []
        if tol <= 0.0:
            tol = 2.0 * float(self.elem_L.min())

        sorted_nodes = sorted(hinged_nodes)
        clusters: list[list[int]] = [[sorted_nodes[0]]]
        for n in sorted_nodes[1:]:
            if self.node_coords[n] - self.node_coords[clusters[-1][-1]] <= tol:
                clusters[-1].append(n)
            else:
                clusters.append([n])

        # Representative: middle node of each cluster
        return [c[len(c) // 2] for c in clusters]

    def _is_mechanism(self, hinged_nodes: set) -> bool:
        """
        Test whether the current set of plastic hinges forms a mechanism.

        Closely-spaced hinges (plastic zones) are first clustered into
        distinct hinge locations.  Then K is assembled with R = 0 at one
        representative node per cluster and checked for rank deficiency.
        """
        reps = self._cluster_hinges(hinged_nodes)
        if len(reps) < 2:
            return False

        R_test = np.ones((self.n_elem, 2))
        for j in reps:
            if j > 0:
                R_test[j - 1, 1] = 0.0
            if j < self.n_elem:
                R_test[j, 0] = 0.0

        K_test = self._assemble_from_elem_R(R_test)
        K_test, _ = self._apply_bc(K_test, np.zeros(self.n_dof))
        rank = np.linalg.matrix_rank(K_test, tol=1e-6)
        return rank < self.n_dof

    # ---- Main analysis ----

    def analyze(
        self,
        LM: list,
        lambda_max: float = 20.0,
        record_every: int = 0,
    ) -> NonlinearResult:
        """
        Incremental nonlinear analysis with adaptive step size.

        Parameters
        ----------
        LM : list of list
            Load matrix in pycba format (1-based spans).
        lambda_max : float
            Maximum load factor to attempt.
        record_every : int
            If > 0, store moment snapshots every *n* increments.

        Returns
        -------
        NonlinearResult
        """
        F_ref = self._build_force_vector(LM)

        # State: per-element-end R values
        R_elem = np.ones((self.n_elem, 2))

        # State: per-node moments and yield tracking
        moments = np.zeros(self.n_nodes)
        gamma_max = np.zeros(self.n_nodes)

        lambda_T = 0.0
        hinge_events: list[HingeEvent] = []
        lambda_history: list[float] = []
        moment_history: list[np.ndarray] = []
        yielded: set[int] = set()
        hinged: set[int] = set()
        step = 0

        while lambda_T < lambda_max:
            # --- Adaptive increment (McCarthy Fig. 4.5) ---
            R_min = float(R_elem.min())
            if R_min > 0.5:
                d_lambda = 0.1
            elif R_min > 0.25:
                d_lambda = 0.01
            else:
                d_lambda = 0.001

            if lambda_T + d_lambda > lambda_max:
                d_lambda = lambda_max - lambda_T

            lambda_T += d_lambda
            step += 1

            # --- Assemble & solve ---
            K = self._assemble_from_elem_R(R_elem)
            F_inc = d_lambda * F_ref
            K_bc, F_bc = self._apply_bc(K, F_inc)

            try:
                u_inc = np.linalg.solve(K_bc, F_bc)
            except np.linalg.LinAlgError:
                return NonlinearResult(
                    collapse_lambda=lambda_T - d_lambda,
                    collapsed=True,
                    hinge_events=hinge_events,
                    node_coords=self.node_coords,
                    final_moments=moments,
                    final_R=R_elem.copy(),
                    lambda_history=lambda_history,
                    moment_history=moment_history,
                )

            # --- Extract increment moments & accumulate ---
            m_inc = self._extract_moments(u_inc, R_elem)
            moments += m_inc

            # --- Update yield functions & R (element-end ownership) ---
            new_hinge = False
            for j in range(self.n_nodes):
                Mp_j = self.node_Mp[j]
                My_j = self.node_My[j]
                gamma_y = My_j / Mp_j
                gamma = abs(moments[j]) / Mp_j

                # Determine owning element end for this node
                ie, end = self._owning_elem_end(j)

                if gamma >= gamma_max[j]:
                    # Loading — update owning element end only
                    gamma_max[j] = gamma
                    R_elem[ie, end] = self._force_recovery(gamma, gamma_y)
                else:
                    # Unloading — elastic recovery at owning end
                    R_elem[ie, end] = 1.0

                # Record events
                if gamma >= gamma_y and j not in yielded:
                    yielded.add(j)
                    hinge_events.append(
                        HingeEvent(lambda_T, self.node_coords[j], j, "initial_yield")
                    )
                if gamma >= 1.0 and j not in hinged:
                    hinged.add(j)
                    hinge_events.append(
                        HingeEvent(lambda_T, self.node_coords[j], j, "plastic_hinge")
                    )
                    new_hinge = True

            # --- Mechanism check (physical: both ends zeroed) ---
            if new_hinge and self._is_mechanism(hinged):
                return NonlinearResult(
                    collapse_lambda=lambda_T,
                    collapsed=True,
                    hinge_events=hinge_events,
                    node_coords=self.node_coords,
                    final_moments=moments,
                    final_R=R_elem.copy(),
                    lambda_history=lambda_history,
                    moment_history=moment_history,
                )

            # --- Record history ---
            if record_every > 0 and step % record_every == 0:
                lambda_history.append(lambda_T)
                moment_history.append(moments.copy())

        # Reached lambda_max without collapse
        return NonlinearResult(
            collapse_lambda=lambda_T,
            collapsed=False,
            hinge_events=hinge_events,
            node_coords=self.node_coords,
            final_moments=moments,
            final_R=R_elem.copy(),
            lambda_history=lambda_history,
            moment_history=moment_history,
        )

    # ---- Moving load analysis ----

    def analyze_moving(
        self,
        P: float,
        step: float = 0.1,
        n_sub: int = 10,
        record_every: int = 0,
    ) -> NonlinearResult:
        """
        Moving-load nonlinear analysis (McCarthy Section 7.3).

        A single point load of magnitude *P* traverses the beam from left
        to right.  At each position step the load is transferred from the
        old position to the new one via *n_sub* paired unload/reload
        sub-increments.  Unloading uses elastic stiffness (R = 1);
        reloading uses the current nonlinear stiffness.

        Parameters
        ----------
        P : float
            Point-load magnitude.
        step : float
            Distance the load moves per position step.
        n_sub : int
            Number of sub-increments per position step for the
            unload/reload transfer.
        record_every : int
            If > 0, append a moment snapshot every *n* position steps.

        Returns
        -------
        NonlinearResult
            ``collapse_lambda`` is the load position (in metres) at
            collapse.  ``hinge_events`` record the load position (not
            a load factor) for each yield / plastic-hinge event.
        """
        positions = np.arange(0, self.total_length + step / 2, step)
        d_frac = 1.0 / n_sub  # fraction of P per sub-increment

        # Persistent state
        R_elem = np.ones((self.n_elem, 2))
        moments = np.zeros(self.n_nodes)
        gamma_max = np.zeros(self.n_nodes)

        hinge_events: list[HingeEvent] = []
        lambda_history: list[float] = []
        moment_history: list[np.ndarray] = []
        yielded: set[int] = set()
        hinged: set[int] = set()

        F_prev = np.zeros(self.n_dof)

        for i_pos, x_pos in enumerate(positions):
            F_cur = self._point_load_vector(P, x_pos)

            if i_pos == 0:
                # --- Initial loading: apply P at first position ---
                for _ in range(n_sub):
                    self._update_R_state(
                        moments, gamma_max, R_elem,
                        yielded, hinged, hinge_events, x_pos,
                    )
                    K = self._assemble_from_elem_R(R_elem)
                    F_inc = d_frac * F_cur
                    K_bc, F_bc = self._apply_bc(K, F_inc)
                    try:
                        u_inc = np.linalg.solve(K_bc, F_bc)
                    except np.linalg.LinAlgError:
                        return self._moving_result(
                            x_pos, True, hinge_events, moments, R_elem,
                            lambda_history, moment_history,
                        )
                    moments += self._extract_moments(u_inc, R_elem)
            else:
                # --- Transfer: unload old, load new in paired sub-steps ---
                for _ in range(n_sub):
                    # 1. Elastic unload at previous position
                    R_elastic = np.ones((self.n_elem, 2))
                    K_el = self._assemble_from_elem_R(R_elastic)
                    F_unload = -d_frac * F_prev
                    K_bc, F_bc = self._apply_bc(K_el, F_unload)
                    u_unload = np.linalg.solve(K_bc, F_bc)
                    moments += self._extract_moments(u_unload, R_elastic)

                    # 2. Update R after unloading
                    new_hinge = self._update_R_state(
                        moments, gamma_max, R_elem,
                        yielded, hinged, hinge_events, x_pos,
                    )

                    # 3. Nonlinear reload at current position
                    K = self._assemble_from_elem_R(R_elem)
                    F_load = d_frac * F_cur
                    K_bc, F_bc = self._apply_bc(K, F_load)
                    try:
                        u_load = np.linalg.solve(K_bc, F_bc)
                    except np.linalg.LinAlgError:
                        return self._moving_result(
                            x_pos, True, hinge_events, moments, R_elem,
                            lambda_history, moment_history,
                        )
                    moments += self._extract_moments(u_load, R_elem)

                # Check mechanism after all sub-steps at this position
                new_hinge = self._update_R_state(
                    moments, gamma_max, R_elem,
                    yielded, hinged, hinge_events, x_pos,
                )
                if new_hinge and self._is_mechanism(hinged):
                    return self._moving_result(
                        x_pos, True, hinge_events, moments, R_elem,
                        lambda_history, moment_history,
                    )

            F_prev = F_cur.copy()

            # Record history
            if record_every > 0 and i_pos % record_every == 0:
                lambda_history.append(x_pos)
                moment_history.append(moments.copy())

        # Full traverse without collapse
        return self._moving_result(
            positions[-1], False, hinge_events, moments, R_elem,
            lambda_history, moment_history,
        )

    def _moving_result(self, x, collapsed, hinge_events, moments, R_elem,
                       lambda_history, moment_history) -> NonlinearResult:
        return NonlinearResult(
            collapse_lambda=x,
            collapsed=collapsed,
            hinge_events=hinge_events,
            node_coords=self.node_coords,
            final_moments=moments,
            final_R=R_elem.copy(),
            lambda_history=lambda_history,
            moment_history=moment_history,
        )
