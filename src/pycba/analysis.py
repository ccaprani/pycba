"""
PyCBA - Continuous Beam Analysis

Implements the direct stiffness method for linear-elastic continuous beam
analysis. The primary entry point is :class:`BeamAnalysis`, which assembles
the global stiffness matrix, applies boundary conditions (including prescribed
displacements / support settlements), solves for nodal displacements, and
recovers reactions and member load effects.

**PyCBA is unit-agnostic.** No conversions are performed; any internally
consistent set of units (e.g. kN/m/kNm, or N/mm/Nmm) may be used as long as
all inputs share the same system.  The only exception is :meth:`BeamAnalysis.plot_results`,
which scales deflections by 1×10³ and labels the axis "mm" — that label is
only correct when the length unit is metres.

Sign conventions
----------------
* Vertical displacements and forces: **positive upward**.
* Rotations and moments: **positive counter-clockwise**.
* Settlement (prescribed displacement): negative value = downward.
* UDL / point loads: positive value = downward acting load.
* Moment loads: positive value = counter-clockwise.

Restraint vector ``R``
~~~~~~~~~~~~~~~~~~~~~~
One entry per nodal DOF (vertical then rotational, node by node):

* ``-1`` — fully fixed (zero displacement unless overridden by ``D``).
* ``0``  — free.
* ``+k`` — elastic spring with stiffness *k* in consistent
  force/length (translational) or force·length/angle (rotational) units.

An OO Python adaptation of CBA, originally written for MATLAB:
http://www.colincaprani.com/programming/matlab/
"""

from typing import Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from .beam import Beam, LoadMatrix
from .results import BeamResults
from .load import add_LM


class BeamAnalysis:
    """
    Direct-stiffness continuous beam analyser.

    Assembles the global stiffness matrix from individual span stiffness
    matrices, applies support boundary conditions (including elastic springs
    and prescribed displacements), solves the linear system for nodal
    displacements, and recovers support reactions and distributed load effects
    along each member.

    After calling :meth:`analyze`, results are available through
    :attr:`beam_results`:

    * ``beam_results.R``  — reactions at fully-fixed DOFs.
    * ``beam_results.Rs`` — spring forces ``k_s × u_i`` at spring DOFs.
    * ``beam_results.D``  — global nodal displacement vector.
    * ``beam_results.results`` — concatenated member load-effect arrays
      (``x``, ``M``, ``V``, ``D``, ``R``).
    """

    def __init__(
        self,
        L: np.ndarray,
        EI: Union[float, np.ndarray],
        R: np.ndarray,
        LM: Optional[LoadMatrix] = None,
        eletype: Optional[np.ndarray] = None,
        D: Optional[list] = None,
    ):
        """
        Construct a beam analysis object.

        Parameters
        ----------
        L : array_like of float
            Span lengths.  Length ``N`` for an ``N``-span beam.
        EI : float or array_like of float
            Flexural rigidity of each span.  A scalar value is applied to all
            spans; otherwise one value per span is required.
        R : array_like of int or float
            Nodal restraint vector, length ``2(N+1)``.  Two entries per node
            (vertical DOF then rotational DOF), ordered left to right:

            * ``-1`` — fully restrained (zero displacement unless overridden
              by ``D``).
            * ``0``  — free.
            * ``+k`` — elastic spring stiffness in consistent units.

        LM : list of list, optional
            Load matrix: a list of load descriptors, each of the form
            ``[span, load_type, value, a, c]``.  Load types:

            1. UDL — ``value`` is load intensity; set ``a = c = 0``.
            2. Point load — ``value`` at distance ``a`` from the span start.
            3. Partial UDL — ``value`` intensity from ``a`` for length ``c``.
            4. Moment load — ``value`` at distance ``a`` from the span start.

        eletype : array_like of int, optional
            Element type for each span, controlling which end(s) carry moment:

            1. Fixed–fixed (default).
            2. Fixed–pinned (moment release at right end).
            3. Pinned–fixed (moment release at left end).
            4. Pinned–pinned (moment releases at both ends).

            At an internal hinge, only one of the two members meeting at that
            node should have a pinned end.
        D : list, optional
            Prescribed-displacement vector, length ``2(N+1)`` (same as ``R``).
            Use ``None`` for DOFs whose displacement is unknown (the default).
            Provide a float for DOFs with a known displacement (e.g. a support
            settlement — negative = downward).  Fixed supports (``R = -1``)
            default to zero displacement unless ``D`` provides an explicit
            override.

        Raises
        ------
        ValueError
            If ``R`` and ``D`` have different lengths, or ``EI`` is not
            scalar and its length differs from ``len(L)``.
        """
        self.npts = 100
        self._beam_results = None

        if eletype is None:
            self.eletype = np.ones((len(L), 1))
        else:
            self.eletype = eletype
        # Create the beam
        self._beam = Beam(L=L, EI=EI, R=R, LM=LM, eletype=self.eletype, D=D)

        self._n = self._beam.no_spans
        self._no_nodes = self._n + 1
        self._nDOF = 2 * self._no_nodes

    @property
    def beam_results(self) -> BeamResults:
        """
        BeamResults : Post-analysis results object.

        ``None`` until :meth:`analyze` has been called successfully.
        Provides nodal displacements (``D``), reactions at fixed supports
        (``R``), spring forces (``Rs``), and per-member load-effect arrays
        (``vRes``, ``results``).
        """
        return self._beam_results

    @property
    def beam(self) -> Beam:
        """
        Beam : The underlying :class:`~pycba.beam.Beam` object.

        Provides direct access to span geometry, stiffness matrices,
        restraints, and prescribed displacements.
        """
        return self._beam

    def set_loads(self, LM: LoadMatrix):
        """
        Replace the current load matrix with a new one.

        Any loads previously added via :meth:`add_udl`, :meth:`add_pl`,
        :meth:`add_pudl`, or :meth:`add_ml` are discarded.

        Parameters
        ----------
        LM : list of list
            New load matrix in the same format as the ``LM`` argument of
            :meth:`__init__`.
        """
        self._beam.loads = LM

    def add_udl(self, i_span: int, w: float):
        """
        Append a full-span uniformly-distributed load.

        Parameters
        ----------
        i_span : int
            1-based span index.
        w : float
            Load intensity.  Positive values act downward.
        """
        load = [i_span, 1, w]
        self._beam.add_load(load)

    def add_pl(self, i_span: int, p: float, a: float):
        """
        Append a point load.

        Parameters
        ----------
        i_span : int
            1-based span index.
        p : float
            Load magnitude.  Positive values act downward.
        a : float
            Distance from the left end of the span to the load.
        """
        load = [i_span, 2, p, a]
        self._beam.add_load(load)

    def add_pudl(self, i_span: int, w: float, a: float, c: float):
        """
        Append a partial uniformly-distributed load.

        Any portion of the load that extends beyond the end of the span is
        silently ignored.

        Parameters
        ----------
        i_span : int
            1-based span index.
        w : float
            Load intensity.  Positive values act downward.
        a : float
            Distance from the left end of the span to the start of the load.
        c : float
            Length (cover) of the partial UDL.
        """
        load = [i_span, 3, w, a, c]
        self._beam.add_load(load)

    def add_ml(self, i_span: int, m: float, a: float):
        """
        Append a concentrated moment load.

        Parameters
        ----------
        i_span : int
            1-based span index.
        m : float
            Moment magnitude.  Positive values are counter-clockwise.
        a : float
            Distance from the left end of the span to the load.
        """
        load = [i_span, 4, m, a]
        self._beam.add_load(load)

    def analyze(self, npts: Optional[int] = None) -> int:
        """
        Execute the direct-stiffness analysis.

        Assembles the unrestricted global stiffness matrix, validates the
        model, applies boundary conditions (including spring supports and
        prescribed displacements), solves for nodal displacements, and
        recovers support reactions.  Results are stored in
        :attr:`beam_results`.

        Parameters
        ----------
        npts : int, optional
            Number of evaluation points along each member for computing
            distributed load effects (bending moment, shear, deflection).
            Must be greater than 3; defaults to 100 if omitted or ``≤ 3``.

        Returns
        -------
        int
            ``0`` on successful completion.

        Raises
        ------
        ValueError
            If the model is invalid (see :meth:`_validate`) or if the
            structure is geometrically unstable (see :meth:`_solver`).
        """
        if npts and npts > 3:
            self.npts = npts

        restraints = self._beam.restraints
        d_presc = self._beam.prescribed_displacements
        fU = self._forces()

        self._validate(restraints, d_presc, fU)

        f = np.copy(fU)
        ksysU = self._assemble()
        ksys = np.copy(ksysU)
        ksys, f = self._apply_bc(ksys, f)
        d = self._solver(ksys, f)
        r, rs = self._reactions(ksysU, d, fU)

        self._beam_results = BeamResults(self._beam, d, r, self.npts, rs)
        return 0

    def _forces(self) -> np.ndarray:
        """
        Build the unrestricted global nodal force vector from the load matrix.

        Iterates over spans, retrieves each span's released end forces
        (consistent nodal loads adjusted for element type / moment releases),
        and accumulates them in the global vector with the sign reversal
        required by the direct stiffness method (loads oppose the reactions
        they generate).

        Returns
        -------
        f : np.ndarray, shape (nDOF,)
            Unrestricted global nodal force vector.
        """
        self._beam._set_loads()

        f = np.zeros(self._nDOF)

        for i in range(self._n):
            dof_i = 2 * i
            fmbr = self._beam.get_ref(i)
            # Cumulatively apply forces in opposite direction
            f[dof_i : dof_i + 4] -= fmbr
        return f

    def _validate(
        self,
        restraints: list,
        d_presc: list,
        fU: np.ndarray,
    ) -> None:
        """
        Pre-analysis model validity check.

        Currently checks for the one combination that is physically
        inconsistent with the direct elimination method: a DOF that
        simultaneously has a spring support, a prescribed displacement, *and*
        a non-zero consistent nodal load.  In that case the BC enforcement
        sets ``f[i] = d_i``, silently overwriting the load — no warning is
        possible after the fact, so this must be caught before the solve.

        Parameters
        ----------
        restraints : list
            Beam restraint vector (same as ``R``).
        d_presc : list
            Prescribed-displacement vector (``None`` entries = free DOFs).
        fU : np.ndarray, shape (nDOF,)
            Unrestricted nodal force vector from :meth:`_forces`.

        Raises
        ------
        ValueError
            If any DOF simultaneously carries a spring restraint, a prescribed
            displacement, and a non-zero consistent nodal load.
        """
        for i in range(self._nDOF):
            if restraints[i] > 0 and d_presc[i] is not None and fU[i] != 0.0:
                raise ValueError(
                    f"Invalid model at DOF {i}: a spring support, a prescribed "
                    f"displacement, and a non-zero external nodal load "
                    f"(fU[{i}] = {fU[i]}) cannot coexist. The elimination "
                    "method would silently discard the nodal load. "
                    "Remove the prescribed displacement, the spring, or the "
                    "external load at this DOF."
                )

    def _assemble(self) -> np.ndarray:
        """
        Assemble the unrestricted global stiffness matrix.

        Loops over spans and overlaps each span's 4×4 stiffness matrix into
        the ``2(N+1) × 2(N+1)`` global matrix using the standard connectivity
        pattern.  Spring stiffnesses (``R > 0``) are added to the diagonal
        here so that the returned matrix is the *complete* unrestricted system,
        which is required by :meth:`_reactions` to recover spring forces
        correctly.

        Returns
        -------
        ksys : np.ndarray, shape (nDOF, nDOF)
            Unrestricted global stiffness matrix including spring contributions.
        """
        ksys = np.zeros((self._nDOF, self._nDOF))

        for i in range(self._n):
            kb = self._beam.get_span_k(i)
            dof_i = 2 * i
            ksys[dof_i : dof_i + 4, dof_i : dof_i + 4] += kb

        # Add spring stiffness before copying so ksys (used for reactions)
        # includes the spring contribution and spring forces are not lost.
        r_vec = self._beam.restraints
        for i in range(self._nDOF):
            if r_vec[i] > 0:
                ksys[i][i] += r_vec[i]

        return ksys

    def _apply_bc(self, k: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Impose boundary conditions using the direct elimination method.

        For each DOF with a known displacement ``d_i`` (either explicitly
        prescribed via ``D`` or implicitly zero for fully-fixed supports),
        the corresponding row and column are zeroed, the diagonal is set to
        one, and the right-hand side is updated by subtracting
        ``k[:, i] * d_i`` before overwriting ``f[i] = d_i``.  This preserves
        the displacement field exactly and transfers the constraint contribution
        to the remaining free DOFs.

        Spring DOFs (``R > 0``) without a prescribed displacement are *not*
        eliminated — their stiffness is already on the diagonal of ``k`` from
        :meth:`_assemble` and they remain as free unknowns.

        Parameters
        ----------
        k : np.ndarray, shape (nDOF, nDOF)
            Unrestricted global stiffness matrix (modified in-place).
        f : np.ndarray, shape (nDOF,)
            Global nodal force vector (modified in-place).

        Returns
        -------
        k : np.ndarray, shape (nDOF, nDOF)
            Restricted stiffness matrix with constrained rows/columns zeroed.
        f : np.ndarray, shape (nDOF,)
            Modified force vector incorporating prescribed displacement values.
        """
        r = self._beam.restraints
        d = self._beam.prescribed_displacements
        for i in range(self._nDOF):
            # Determine prescribed value: explicit settlement, or fixed support (= 0)
            if d[i] is not None:
                di = d[i]
            elif r[i] < 0:
                di = 0.0
            else:
                continue  # free or spring-only DOF: nothing to eliminate

            # Subtract full column i from RHS (including diagonal) before zeroing
            f -= k[:, i] * di
            k[i, :] = 0.0
            k[:, i] = 0.0
            k[i, i] = 1.0
            f[i] = di
        return k, f

    def _reactions(self, k: np.ndarray, d: np.ndarray, f: np.ndarray) -> tuple:
        """
        Recover support reactions and spring forces from the solved displacement field.

        Uses the unrestricted global stiffness matrix ``k`` (i.e. ``ksysU``
        from :meth:`analyze`) so that spring contributions are included in the
        residual calculation.

        For fully-fixed DOFs (``restraints[i] == -1``) the nodal residual
        ``(k @ d - f)[i]`` equals the support reaction directly, because those
        DOFs have zero (or prescribed) displacement and no ambiguity with
        spring terms.

        For spring DOFs (``restraints[i] > 0``) the residual also contains
        structural coupling and any applied nodal load, so it does *not* in
        general equal the spring force alone.  The spring force is therefore
        computed explicitly as ``k_s * u_i``.

        Parameters
        ----------
        k : np.ndarray, shape (nDOF, nDOF)
            Unrestricted global stiffness matrix (including spring terms).
        d : np.ndarray, shape (nDOF,)
            Solved global nodal displacement vector (m, rad).
        f : np.ndarray, shape (nDOF,)
            Unrestricted nodal force vector ``fU`` from :meth:`_forces`.

        Returns
        -------
        r : np.ndarray
            Reactions at fully-fixed DOFs (``restraints[i] == -1``), in DOF
            order.
        rs : np.ndarray
            Spring forces ``-k_s * u_i`` (upward positive) at spring DOFs
            (``restraints[i] > 0``), in DOF order.
        """
        residual = k @ d - f
        restraints = self._beam.restraints
        # For fixed DOFs the full nodal residual equals the support reaction,
        # because d[i] = 0 leaves no ambiguity about which term dominates.
        r = np.array([residual[i] for i in range(self._nDOF) if restraints[i] < 0])
        # For spring DOFs, residual[i] = k_s*u_i + structural coupling - f_applied[i],
        # so it is NOT purely the spring force when external nodal loads are present.
        # Use -k_s*u_i explicitly: negative because the spring reaction is upward
        # (positive) when the displacement is downward (negative).
        rs = np.array(
            [-restraints[i] * d[i] for i in range(self._nDOF) if restraints[i] > 0]
        )
        return r, rs

    def _solver(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the restricted linear system ``A x = b`` for nodal displacements.

        Parameters
        ----------
        A : np.ndarray, shape (nDOF, nDOF)
            Restricted global stiffness matrix from :meth:`_apply_bc`.
        b : np.ndarray, shape (nDOF,)
            Restricted force vector from :meth:`_apply_bc`.

        Returns
        -------
        x : np.ndarray, shape (nDOF,)
            Nodal displacement vector.

        Raises
        ------
        ValueError
            If ``A`` is singular, indicating a geometrically unstable
            structure (insufficient support restraints).
        """
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Structure is geometrically unstable: the stiffness matrix is "
                "singular. Check that sufficient support restraints are defined."
            ) from exc
        return x

    def plot_results(self):
        """
        Plot bending moment, shear force, and deflection diagrams.

        Produces a three-panel figure of bending moment, shear force, and
        deflection along the beam.  Bending moment is plotted with the
        sagging-positive convention (y-axis inverted so sagging appears below
        the beam line).

        .. note::
            The axis labels ("kNm", "kN", "mm") and the deflection scaling
            (×1000) assume a kN / m unit system.  If a different consistent
            unit system is used the plots will still be correct in shape but
            the labels and deflection axis values will need interpretation.

        Has no effect and prints a warning if :meth:`analyze` has not been
        called yet.
        """
        if self._beam_results is None:
            print("Nothing to plot - run analysis first")
            return
        res = self._beam_results.results
        L = self._beam.length

        fig, axs = plt.subplots(3, 1)

        ax = axs[0]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, res.M, "r")
        ax.invert_yaxis()
        ax.grid()
        ax.set_ylabel("Bending Moment (kNm)")

        ax = axs[1]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, res.V, "r")
        ax.grid()
        ax.set_ylabel("Shear Force (kN)")

        ax = axs[2]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, res.D * 1e3, "r")
        ax.grid()
        ax.set_ylabel("Deflection (mm)")
        ax.set_xlabel("Distance along beam (m)")

        plt.show()
