"""
PyCBA - Continuous Beam Analysis

Implements the direct stiffness method for linear-elastic continuous beam
analysis. The primary entry point is :class:`BeamAnalysis`, which assembles
the global stiffness matrix, applies boundary conditions (including prescribed
displacements / support settlements), solves for nodal displacements, and
recovers reactions and member load effects.

**PyCBA is unit-agnostic.** No conversions are performed; any internally
consistent set of units (e.g. kN/m/kNm, or N/mm/Nmm) may be used as long as
all inputs share the same system.  Units appear only when results are
*plotted* — in the axis labels and the deflection display scale — governed by
a display unit system (see :mod:`pycba.units` and :func:`pycba.set_units`).
The default is SI with kN and m; pass ``units=`` to any plotting method, or
call :func:`pycba.set_units`, to label a different system (the analysis itself
is unaffected).

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

from typing import Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
from .beam import Beam, LoadMatrix
from .results import BeamResults
from .load import add_LM
from .section import SectionEI


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
        R: Optional[np.ndarray] = None,
        LM: Optional[LoadMatrix] = None,
        eletype: Optional[np.ndarray] = None,
        D: Optional[list] = None,
        supports: Optional[Sequence] = None,
        GAv: Optional[Union[float, SectionEI, Sequence]] = None,
        kf: Optional[Union[float, Sequence]] = None,
    ):
        """
        Construct a beam analysis object.

        Parameters
        ----------
        L : array_like of float
            Span lengths.  Length ``N`` for an ``N``-span beam.
        EI : float, pycba.section.SectionEI, or array_like
            Flexural rigidity of each span.  A single scalar (or a single
            :class:`~pycba.section.SectionEI`) is applied to all spans;
            otherwise one entry per span is required.  A span whose rigidity is
            given as a :class:`~pycba.section.SectionEI` is treated as
            **non-prismatic** (variable ``EI``) and analysed by flexibility
            integration; scalar entries use the closed-form prismatic element.
        R : array_like of int or float, optional
            Nodal restraint vector, length ``2(N+1)``.  Two entries per node
            (vertical DOF then rotational DOF), ordered left to right:

            * ``-1`` — fully restrained (zero displacement unless overridden
              by ``D``).
            * ``0``  — free.
            * ``+k`` — elastic spring stiffness in consistent units.

            Provide either ``R`` or the friendlier ``supports``, not both.

        LM : list of list, optional
            Load matrix: a list of load descriptors.  The number of columns
            per entry depends on the load type:

            1. UDL — ``[span, 1, w]``.
            2. Point load — ``[span, 2, P, a]``.
            3. Partial UDL — ``[span, 3, w, a, c]``.
            4. Moment load — ``[span, 4, M, a]``.
            5. Trapezoidal — ``[span, 5, w1, w2]`` (full span) or
               ``[span, 5, w1, w2, a, c]`` (partial).
            6. Imposed curvature — ``[span, 6, k0, k1, ...]`` where the free
               curvature field is ``κ(x) = k0 + k1·x + …`` (e.g. creep,
               shrinkage or thermal curvature).

        eletype : array_like, optional
            Element type for each span, controlling which end(s) carry moment.
            Each entry may be an integer code, a :class:`~pycba.MemberType`,
            or its name string (e.g. ``"FP"``):

            1. ``FF`` Fixed–fixed (default).
            2. ``FP`` Fixed–pinned (moment release at right end).
            3. ``PF`` Pinned–fixed (moment release at left end).
            4. ``PP`` Pinned–pinned (moment releases at both ends).

            At an internal hinge, only one of the two members meeting at that
            node should have a pinned end.
        D : list, optional
            Prescribed-displacement vector, length ``2(N+1)`` (same as ``R``).
            Use ``None`` for DOFs whose displacement is unknown (the default).
            Provide a float for DOFs with a known displacement (e.g. a support
            settlement — negative = downward).  Fixed supports (``R = -1``)
            default to zero displacement unless ``D`` provides an explicit
            override.
        supports : sequence of (str or [float, float]), optional
            A friendlier alternative to ``R``: one entry per node (left to
            right), each a support name or a raw ``[vertical, rotation]`` DOF
            pair.  Recognised names (case-insensitive):

            * ``"p"`` / ``"pin"`` / ``"pinned"`` and ``"r"`` / ``"roller"`` —
              vertical held, rotation free.
            * ``"e"`` / ``"encastre"`` / ``"fixed"`` / ``"clamped"`` — fully
              fixed.
            * ``"f"`` / ``"free"`` — unrestrained (e.g. a cantilever tip).

            Elastic springs are given as a raw pair, e.g. ``[5e4, 0]`` for a
            vertical spring.  Lowered to ``R`` via
            :func:`~pycba.supports_to_R`; mutually exclusive with ``R``.
        GAv : float, pycba.section.SectionEI, or array_like, optional
            Transverse shear rigidity ``G·A_v`` of each span.  A span given a
            finite ``GAv`` is analysed as a shear-deformable **Timoshenko**
            element; ``None`` (the default) keeps the exact Euler–Bernoulli
            element.  Broadcasts like ``EI``: a single scalar (or one
            :class:`~pycba.section.SectionEI` for a variable ``GAv(x)``) applies
            to all spans, otherwise one entry per span.
        kf : float or array_like, optional
            Winkler foundation modulus (modulus of subgrade reaction per unit
            beam length).  A span with a finite ``kf`` rests on an elastic
            (Winkler) foundation, modelled as a statically-condensed
            beam-on-elastic-foundation super-element.  Like ``EI``: a scalar
            applies to all spans, otherwise one entry per span (each ``None`` or
            a modulus).  Supported for prismatic, fixed-fixed spans without
            ``GAv``, carrying UDL / point / partial-UDL loads.

        Raises
        ------
        ValueError
            If both ``R`` and ``supports`` (or neither) are given, ``R`` and
            ``D`` have different lengths, or ``EI`` is not scalar and its length
            differs from ``len(L)``.
        """
        self.npts = 100
        # Optional per-member "shear point" sections (0-based member index ->
        # member-local coordinates) spliced into the evaluation grid so the
        # shear is recovered exactly either side of each section.  ``None``
        # leaves the uniform grid unchanged; set by a moving-load analysis.
        self.shear_points = None
        self._beam_results = None

        if eletype is None:
            self.eletype = np.ones((len(L), 1))
        else:
            self.eletype = eletype
        # Create the beam
        self._beam = Beam(
            L=L,
            EI=EI,
            R=R,
            LM=LM,
            eletype=self.eletype,
            D=D,
            supports=supports,
            GAv=GAv,
            kf=kf,
        )

        self._n = self._beam.no_spans
        self._no_nodes = self._n + 1
        self._nDOF = 2 * self._no_nodes
        # Beam structure_version at the last successful stability check, so the
        # check is not repeated across looped analyses (e.g. a moving load)
        # unless the structure itself changes.  ``None`` = never checked.
        self._checked_version = None

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

    def add_ic(self, i_span: int, kappa):
        r"""
        Append an imposed-curvature (initial-strain) member load.

        The free curvature field ``κ(x) = k0 + k1·x + k2·x² + …`` is imposed
        over the member.  On a simply-supported span it produces no internal
        forces (only a free deflected shape); on a restrained or continuous
        structure its restraint generates real moments and reactions.  This is
        the mechanism for applying creep, shrinkage and thermal curvatures to a
        continuous beam (see :class:`pycba.load.LoadIC`).

        Parameters
        ----------
        i_span : int
            1-based span index.
        kappa : float or array_like of float
            Imposed-curvature polynomial coefficients in increasing powers of
            ``x``: ``[k0, k1, k2, ...]``.  A scalar is a uniform curvature.
        """
        coeffs = np.atleast_1d(np.asarray(kappa, dtype=float)).tolist()
        load = [i_span, 6] + coeffs
        self._beam.add_load(load)

    def add_trap(
        self,
        i_span: int,
        w1: float,
        w2: float,
        a: Optional[float] = None,
        c: Optional[float] = None,
    ):
        """
        Append a trapezoidal (linearly varying) distributed load.

        When *a* and *c* are omitted the load covers the full span, varying
        from *w1* at the left end to *w2* at the right end.  When *a* and *c*
        are given the load covers the region from *a* to *a + c*, varying from
        *w1* to *w2* over that length.

        Parameters
        ----------
        i_span : int
            1-based span index.
        w1 : float
            Load intensity at the start of the load.  Positive values act downward.
        w2 : float
            Load intensity at the end of the load.  Positive values act downward.
        a : float, optional
            Distance from the left end of the span to the start of the load.
            If given, *c* must also be provided.
        c : float, optional
            Length (cover) of the load.  Required when *a* is provided.
        """
        if a is not None and c is None:
            raise ValueError("If 'a' is specified, 'c' must also be provided")
        if a is not None:
            load = [i_span, 5, w1, w2, a, c]
        else:
            load = [i_span, 5, w1, w2]
        self._beam.add_load(load)

    def analyze(self, npts: Optional[int] = None, check_stability: bool = True) -> int:
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
        check_stability : bool, optional
            If ``True`` (default), check the assembled stiffness for a
            mechanism before solving and raise a clear error (see
            :meth:`is_stable` / :meth:`_check_stability`).  Set ``False`` to
            skip the check for an unusual but intentionally near-singular
            model.  The check runs at most once per structure: its result is
            cached and only re-evaluated if the beam structure changes, so it
            adds no cost to looped analyses (e.g. a moving load) that vary
            only the loads.

        Returns
        -------
        int
            ``0`` on successful completion.

        Raises
        ------
        ValueError
            If the model is invalid (see :meth:`_validate`) or if the
            structure is geometrically unstable (see :meth:`_check_stability`
            and :meth:`_solver`).
        """
        if npts and npts > 3:
            self.npts = npts

        restraints = self._beam.restraints
        d_presc = self._beam.prescribed_displacements
        fU = self._forces()

        self._validate(restraints, d_presc, fU)

        f = np.copy(fU)
        ksysU = self._assemble()
        if check_stability and self._checked_version != self._beam.structure_version:
            self._check_stability(ksysU, restraints, d_presc)
            self._checked_version = self._beam.structure_version
        ksys = np.copy(ksysU)
        ksys, f = self._apply_bc(ksys, f)
        d = self._solver(ksys, f)
        r, rs = self._reactions(ksysU, d, fU)

        self._beam_results = BeamResults(
            self._beam, d, r, self.npts, rs, self.shear_points
        )
        return 0

    def modal(self, mass, n_modes: int = 10, nseg: int = 12):
        """
        Free-vibration (modal) analysis: natural frequencies and mode shapes.

        Assembles a consistent mass matrix alongside the stiffness matrix on a
        refined mesh (each span split into ``nseg`` Euler-Bernoulli
        sub-elements) and solves the generalized eigenproblem
        ``K φ = ω² M φ``.  Supports, including elastic springs, are applied at
        the original span nodes; the analysis is independent of any applied
        loads.

        Parameters
        ----------
        mass : float or array_like
            Mass per unit length, a scalar for every span or one value per span
            (consistent units, e.g. kg/m if EI is in N·m²).
        n_modes : int, optional
            Number of lowest modes to return. The default is 10.
        nseg : int, optional
            Sub-elements per span for the refined mesh. The default is 12.

        Returns
        -------
        pycba.modal.ModalResults
            The natural frequencies (``omega`` rad/s, ``f`` Hz) and mode shapes.

        Notes
        -----
        Supported for prismatic, fixed-fixed spans without shear flexibility
        (``GAv``); other combinations raise a clear ``NotImplementedError``.
        """
        from .modal import solve_modal

        return solve_modal(self._beam, mass, n_modes=n_modes, nseg=nseg)

    # Reciprocal-condition-number floor for the free-DOF stiffness partition.
    # True mechanisms sit near machine epsilon (~1e-16); legitimately flexible
    # structures stay well above this, so 1e-12 separates them with margin.
    _STABILITY_RCOND = 1e-12

    def _check_stability(
        self, ksysU: np.ndarray, restraints: list, d_presc: list
    ) -> None:
        """
        Detect a mechanism (near-singular structure) before solving.

        Direct elimination puts ``1.0`` on the diagonal of constrained DOFs,
        which pollutes the condition number of the reduced system, so this
        works instead on the **free-DOF partition** of the *unrestricted*
        stiffness matrix - the block that actually governs the unknown
        displacements.  Excluded DOFs are those that are fully fixed
        (``restraints < 0``) or carry a prescribed displacement; spring DOFs
        remain free and contribute their stiffness.

        For a stable linear-elastic structure this partition is symmetric
        positive-definite.  A mechanism (insufficient restraint, or an
        over-released internal hinge) makes it singular: its smallest
        eigenvalue collapses to zero relative to the largest.  The reciprocal
        condition number ``min|λ| / max|λ|`` is therefore compared against
        :attr:`_STABILITY_RCOND`; a value below it indicates a mechanism.

        Parameters
        ----------
        ksysU : np.ndarray, shape (nDOF, nDOF)
            Unrestricted global stiffness matrix (including spring terms).
        restraints : list
            Beam restraint vector (same as ``R``).
        d_presc : list
            Prescribed-displacement vector (``None`` entries = free DOFs).

        Raises
        ------
        ValueError
            If the free-DOF stiffness partition is singular to within
            :attr:`_STABILITY_RCOND`, i.e. the structure is a mechanism.
        """
        free = [
            i
            for i in range(self._nDOF)
            if not (restraints[i] < 0 or d_presc[i] is not None)
        ]
        if not free:
            return  # fully constrained: nothing to solve, trivially stable

        kff = ksysU[np.ix_(free, free)]
        ev = np.abs(np.linalg.eigvalsh(kff))
        ev_max = ev.max()
        if ev_max == 0.0 or (ev.min() / ev_max) < self._STABILITY_RCOND:
            raise ValueError(
                "Structure is geometrically unstable: the free-DOF stiffness "
                "is singular, indicating a mechanism (e.g. insufficient support "
                "restraints or an over-released internal hinge). Add restraint, "
                "or pass analyze(check_stability=False) to override this check."
            )

    def is_stable(self) -> bool:
        """
        Return whether the structure is stable (not a mechanism).

        Runs the same free-DOF stability check as :meth:`analyze` but returns a
        boolean instead of raising, so the model can be validated up front
        without solving.  A ``True`` result is cached (keyed on the beam's
        :attr:`~pycba.beam.Beam.structure_version`), so a subsequent
        ``analyze()`` does not repeat the check unless the structure changes.

        Returns
        -------
        bool
            ``True`` if the free-DOF stiffness partition is non-singular to
            within :attr:`_STABILITY_RCOND`, otherwise ``False``.
        """
        restraints = self._beam.restraints
        d_presc = self._beam.prescribed_displacements
        try:
            self._check_stability(self._assemble(), restraints, d_presc)
        except ValueError:
            return False
        self._checked_version = self._beam.structure_version
        return True

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

    def plot_beam(
        self,
        loads=None,
        *,
        tikz=None,
        ax=None,
        save=None,
        compile=False,
        load_cases=None,
        **kwargs,
    ):
        """
        Draw a structural schematic of the beam.

        Convenience wrapper around :meth:`pycba.beam.Beam.plot` so the model
        can be drawn directly from the analysis object (mirroring
        :meth:`plot_results`) without reaching through :attr:`beam`.  Renders
        with matplotlib by default; saving to a ``.tex`` path (or passing
        ``tikz=True``) produces TikZ/``stanli`` output instead.

        The beam structure (geometry, supports, internal hinges) is always
        drawn; the loads layer is optional and its source is selected with
        ``loads``:

        * ``None`` (default) - the beam's own load matrix.
        * ``[]`` - draw the bare structure only.
        * a PyCBA load matrix, a :class:`~pycba.load_cases.LoadCase`, or a
          :class:`~pycba.load_cases.LoadCombination` (supply its
          :class:`~pycba.load_cases.LoadCases` via ``load_cases``).

        Parameters
        ----------
        loads : list | LoadCase | LoadCombination, optional
            The load source to draw.
        tikz : bool, optional
            Backend selector.  ``None`` (default) infers it from ``save`` (a
            ``.tex`` target renders TikZ, anything else uses matplotlib); pass
            ``True``/``False`` to force the backend.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into (matplotlib backend only); a new figure is
            created if omitted.
        save : str or pathlib.Path, optional
            If given, also write the visualisation to this path.  A ``.tex``
            target writes the TikZ source (and selects the TikZ backend); any
            other extension is saved by matplotlib.
        compile : bool
            Under the TikZ backend with ``save`` set, also run ``pdflatex`` to
            produce a PDF (a ``.pdf`` save target enables this automatically).
        load_cases : pycba.load_cases.LoadCases, optional
            Required only when ``loads`` is a ``LoadCombination``.
        **kwargs
            Forwarded to the backend renderer (``dimensions``, ``labels``,
            ``load_values``, ``color`` for matplotlib; ``standalone``,
            ``scale``, ``dimensions``, ``labels``, ``load_values`` for TikZ).
            ``units`` selects the display unit system (see
            :func:`pycba.set_units`).

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn into (default matplotlib backend).
        str
            The LaTeX source, when ``tikz=True`` and ``save`` is not given.
        pathlib.Path
            The written file, when ``tikz=True`` and ``save`` is given.
        """
        return self._beam.plot(
            loads,
            tikz=tikz,
            ax=ax,
            save=save,
            compile=compile,
            load_cases=load_cases,
            **kwargs,
        )

    def plot_results(
        self,
        show_beam: bool = True,
        show: bool = True,
        units=None,
        figsize=None,
        backend=None,
    ):
        """
        Plot bending moment, shear force, and deflection diagrams.

        Produces a figure of bending moment, shear force, and deflection along
        the beam.  Bending moment is plotted with the sagging-positive
        convention (y-axis inverted so sagging appears below the beam line).
        By default the loaded-beam schematic is drawn as a top panel sharing
        the x-axis, so the model and its load effects can be read together.

        Parameters
        ----------
        show_beam : bool
            Draw the beam/loading schematic as a top panel above the result
            diagrams (default ``True``).  Set ``False`` for the bare
            three-panel moment/shear/deflection figure.
        show : bool
            Call ``matplotlib.pyplot.show()`` before returning (default
            ``True``).  Set ``False`` to obtain the figure handles without
            displaying — e.g. to ``savefig`` or restyle the figure first.
        units : str or pycba.units.UnitSystem, optional
            Display unit system for the axis labels and the deflection scale
            (e.g. ``"SI"``, ``"US-ft"``, ``"N-mm"``, ``"none"``).  Defaults to
            the global default (see :func:`pycba.set_units`); the analysis
            itself is unit-agnostic and unaffected.
        figsize : tuple(float, float), optional
            Figure size in inches. Defaults to 10 wide and ~3 in per subplot
            row (so the diagrams are not squashed), consistent with the other
            PyCBA result plots; pass an explicit tuple to override.
        backend : {"matplotlib", "plotly"}, optional
            Plotting backend; defaults to the global default (see
            :func:`pycba.set_backend`).  With ``"plotly"`` an interactive,
            hover-to-read figure of the three diagrams (sharing the x-axis) is
            returned; the ``show_beam``, ``show`` and ``figsize`` arguments do
            not apply.

        Returns
        -------
        matplotlib.figure.Figure
            The figure, or ``None`` if :meth:`analyze` has not been called.
            With ``backend="plotly"`` a single ``plotly.graph_objects.Figure``
            is returned instead of the ``(figure, axes)`` pair.
        numpy.ndarray of matplotlib.axes.Axes
            The panel axes (length 4 when ``show_beam`` is ``True``, else 3).

        Notes
        -----
        Has no effect and prints a warning if :meth:`analyze` has not been
        called yet.
        """
        from .units import resolve
        from .plotting import resolve_backend

        if self._beam_results is None:
            print("Nothing to plot - run analysis first")
            return None
        if resolve_backend(backend) == "plotly":
            from .plotting import results_figure

            return results_figure(self._beam_results.results, units=units)
        us = resolve(units)
        res = self._beam_results.results
        L = self._beam.length

        if show_beam:
            # ~3 in per diagram row, plus a slim strip for the schematic.
            ratios = [0.6, 1.0, 1.0, 1.0]
            fig, axs = plt.subplots(
                4,
                1,
                sharex=True,
                figsize=figsize or (10, 3.0 * sum(ratios)),
                gridspec_kw={"height_ratios": ratios},
            )
            # The schematic stretches to fill its panel (equal_aspect=False) so
            # it stays aligned in x with the diagrams below.
            self._beam.plot(ax=axs[0], dimensions=False, equal_aspect=False, units=us)
            axs[0].set_xlabel("")
            diag = axs[1:]
        else:
            fig, axs = plt.subplots(3, 1, sharex=True, figsize=figsize or (10, 9.0))
            diag = axs

        ax = diag[0]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, res.M, "r")
        ax.invert_yaxis()
        ax.grid()
        ax.set_ylabel(us.moment_axis)

        ax = diag[1]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, res.V, "r")
        ax.grid()
        ax.set_ylabel(us.shear_axis)

        ax = diag[2]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, res.D * us.disp_scale, "r")
        ax.grid()
        ax.set_ylabel(us.deflection_axis)
        ax.set_xlabel(us.distance_axis)

        if show:
            plt.show()
        return fig, axs

    def _plot_diagram(
        self, kind, ax=None, units=None, figsize=None, backend=None, **kwargs
    ):
        """
        Draw a single result diagram (bending moment, shear, or deflection).

        Shared backend for :meth:`plot_bmd`, :meth:`plot_sfd` and
        :meth:`plot_dsd`.  When ``ax`` is ``None`` a new figure/axes is
        created and fully set up (zero baseline, axis labels, grid, and -- for
        the bending moment -- the inverted, sagging-positive y-axis); when an
        existing ``ax`` is passed only the curve is added, so a second analysis
        can be overlaid for comparison without disturbing the axis.

        Parameters
        ----------
        kind : {"M", "V", "D"}
            Which load effect to draw.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created if omitted.
        units : str or pycba.units.UnitSystem, optional
            Display unit system (see :func:`pycba.set_units`).
        figsize : tuple(float, float), optional
            Figure size when a new axes is created.
        **kwargs
            Forwarded to :meth:`matplotlib.axes.Axes.plot` for the curve
            (e.g. ``color``, ``ls``, ``lw``, ``label``).

        Returns
        -------
        matplotlib.axes.Axes or None
            The axes drawn into, or ``None`` if :meth:`analyze` has not run.
        """
        from .units import resolve
        from .plotting import resolve_backend

        if self._beam_results is None:
            print("Nothing to plot - run analysis first")
            return None
        if resolve_backend(backend) == "plotly":
            from .plotting import diagram_figure

            return diagram_figure(self._beam_results.results, kind, units=units)
        us = resolve(units)
        res = self._beam_results.results
        L = self._beam.length
        y, ylabel, invert = {
            "M": (res.M, us.moment_axis, True),
            "V": (res.V, us.shear_axis, False),
            "D": (res.D * us.disp_scale, us.deflection_axis, False),
        }[kind]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize or (8, 3.2))
            ax.plot([0, L], [0, 0], "k", lw=2)
            ax.grid()
            ax.set_ylabel(ylabel)
            ax.set_xlabel(us.distance_axis)
            if invert:
                ax.invert_yaxis()
        kwargs.setdefault("color", "r")
        ax.plot(res.x, y, **kwargs)
        return ax

    def plot_bmd(self, ax=None, units=None, backend=None, **kwargs):
        """
        Plot the bending-moment diagram.

        Uses the sagging-positive convention (the y-axis is inverted so sagging
        appears below the beam line), matching :meth:`plot_results`.  Pass an
        existing ``ax`` to overlay a second analysis for comparison; the axis is
        set up only on the first call, so the overlay is not re-inverted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created if omitted.
        units : str or pycba.units.UnitSystem, optional
            Display unit system (see :func:`pycba.set_units`).
        backend : {"matplotlib", "plotly"}, optional
            Plotting backend; defaults to the global default (see
            :func:`pycba.set_backend`).  With ``"plotly"`` an interactive figure
            is returned and ``ax``/``**kwargs`` do not apply.
        **kwargs
            Forwarded to the curve plot (``color``, ``ls``, ``lw``, ``label``).

        Returns
        -------
        matplotlib.axes.Axes or None
            The axes drawn into (``None`` before :meth:`analyze`), or a
            ``plotly.graph_objects.Figure`` with ``backend="plotly"``.
        """
        return self._plot_diagram("M", ax=ax, units=units, backend=backend, **kwargs)

    def plot_sfd(self, ax=None, units=None, backend=None, **kwargs):
        """
        Plot the shear-force diagram.

        See :meth:`plot_bmd` for the parameters (overlay via ``ax``, ``units``,
        ``backend`` and matplotlib ``**kwargs``).

        Returns
        -------
        matplotlib.axes.Axes or None
            The axes drawn into (``None`` before :meth:`analyze`), or a
            ``plotly.graph_objects.Figure`` with ``backend="plotly"``.
        """
        return self._plot_diagram("V", ax=ax, units=units, backend=backend, **kwargs)

    def plot_dsd(self, ax=None, units=None, backend=None, **kwargs):
        """
        Plot the deflected-shape diagram.

        See :meth:`plot_bmd` for the parameters (overlay via ``ax``, ``units``,
        ``backend`` and matplotlib ``**kwargs``).

        Returns
        -------
        matplotlib.axes.Axes or None
            The axes drawn into (``None`` before :meth:`analyze`), or a
            ``plotly.graph_objects.Figure`` with ``backend="plotly"``.
        """
        return self._plot_diagram("D", ax=ax, units=units, backend=backend, **kwargs)

    def __repr__(self) -> str:
        b = self._beam
        n_sup = b._n_supports()
        n_loads = len(b.LM)
        state = "analysed" if self._beam_results is not None else "not analysed"
        return (
            f"BeamAnalysis({b.no_spans} span{'' if b.no_spans == 1 else 's'}, "
            f"{n_sup} support{'' if n_sup == 1 else 's'}, "
            f"{n_loads} load{'' if n_loads == 1 else 's'}, {state})"
        )
