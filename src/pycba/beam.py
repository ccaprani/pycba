"""
PyCBA - Beam Class definition
"""
from typing import Optional, Union
import numpy as np
from scipy import integrate
from .load import parse_LM, LoadType, LoadMatrix, LoadCNL, LoadIC
from .section import SectionEI


class Beam:
    """
    Class definition
    """

    def __init__(
        self,
        L: Optional[np.ndarray] = None,
        EI: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        LM: Optional[LoadMatrix] = None,
        eletype: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ):
        """
        Constructs a beam object

        Parameters
        ----------
        L : np.ndarray
            A vector of span lengths.
        EI : np.ndarray
            A vector of member flexural rigidities.
        R : np.ndarray
            A vector describing the support conditions at each member end.
        LM : Optional[list[list[Union[int, float]]]]
            The load matrix: a list of loads on the beam; each load with several
            parameters.
        eletype : Optional[np.ndarray]
            A vector of the member types. Defaults to a fixed-fixed element.
        D : Optional[np.ndarray]
            A vector of prescribed displacements. Must have same length as R.
            Use None for DOFs without prescribed displacement.


        Returns
        -------
        None.
        """
        self._no_spans = 0
        self._no_restraints = 0
        self._length = 0
        self.mbr_lengths = []
        self.mbr_EIs = []
        self.mbr_eletype = []
        self._restraints = []
        self._prescribed_displacements = []
        self._loads = []
        self.LM = []
        self._terminal_coords = [0.0]
        # Bumped on any change to the *structure* (geometry, rigidity,
        # restraints, prescribed displacements) so a cached stability check can
        # be invalidated.  Load changes deliberately do not bump it.
        self._structure_version = 0

        if L is not None and eletype is not None:
            # A single rigidity (scalar EI, or one SectionEI) applies to all
            # spans; otherwise one rigidity per span is required.
            if isinstance(EI, (float, int, SectionEI)):
                for l, et in zip(L, eletype):
                    self.add_span(l, EI, et)
            else:
                if len(L) == len(EI):
                    for l, ei, et in zip(L, EI, eletype):
                        self.add_span(l, ei, et)
                else:
                    raise ValueError("Define EI for each span")
            if len(R) == 2 * len(L) + 2:
                self._restraints = R
            else:
                raise ValueError("Insufficient restraints defined")
            if D is not None:
                if len(D) != len(R):
                    raise ValueError(
                        f"D must have same length as R ({len(R)}), got {len(D)}"
                    )
                self._prescribed_displacements = D
            else:
                self._prescribed_displacements = [None] * len(R)
        if LM is not None:
            self.LM = LM

    def add_span(self, L: float, EI: float, eletype: int):
        """
        Add a span to the continuous beam

        Parameters
        ----------
        L : float
            The length of the member.
        EI : np.ndarray
            The flexural rigidity of the member.
        eletype : int
            The element type for the member

        Returns
        -------
        None.

        """
        if isinstance(EI, SectionEI):
            # A non-prismatic section must span the full member length so its
            # piecewise EI(x) covers the element exactly.
            EI.validate_length(L)
        self.mbr_lengths.append(L)
        self.mbr_EIs.append(EI)
        self.mbr_eletype.append(eletype)
        self._no_spans = len(self.mbr_lengths)
        self._length += L
        self._terminal_coords.append(self._terminal_coords[-1] + L)
        self._structure_version += 1

    @property
    def loads(self) -> LoadMatrix:
        """
        Returns the load matrix for the beam

        Returns
        -------
        LM : LoadMatrix
            The load matrix for the beam

        """
        return self.LM

    @loads.setter
    def loads(self, LM):
        """
        Sets the load matrix for the beam

        Parameters
        -------
        LM : LoadMatrix
            The load matrix for the beam

        Returns
        -------
        None

        """
        self.LM = LM
        self.no_loads = len(self.LM)

    def add_load(self, load: LoadType):
        """
        Adds a new load to the beam's load matrix

        Parameters
        ----------
        load : List[Union[int,float]]
            A list describing the load to be added

        Returns
        -------
        None
        """

        self.LM.append(load)
        self.no_loads = len(self.LM)

    def _set_loads(self):
        """
        Explicit internal setter for loads

        Parameters
        -------
        LM : LoadMatrix
            The load matrix for the beam

        Returns
        -------
        None

        """
        self._loads = parse_LM(self.LM)
        # Imposed-curvature loads need the member rigidity to evaluate their
        # fixed-end forces; supply it from the span definition.
        for load in self._loads:
            if isinstance(load, LoadIC):
                load.EI = self.mbr_EIs[load.i_span]

    @property
    def restraints(self) -> np.ndarray:
        """
        Returns the restraints vector for the beam

        Returns
        -------
        _restraints : np.ndarray
            The restraints vector for the beam

        """
        return self._restraints

    @restraints.setter
    def restraints(self, r):
        """
        Stores support conditions

        Parameters
        -------
        r : np.ndarray
            The restraint vector for the beam

        Returns
        -------
        None

        """
        self._restraints = r
        self._structure_version += 1

    @property
    def structure_version(self) -> int:
        """
        int : A counter that increments whenever the beam *structure* changes.

        Bumped by changes to geometry, rigidity, restraints or prescribed
        displacements (but not loads).  Used to invalidate a cached stability
        check (see :meth:`pycba.analysis.BeamAnalysis.is_stable`).
        """
        return self._structure_version

    @property
    def prescribed_displacements(self) -> list:
        """
        Returns the prescribed displacements vector for the beam

        Returns
        -------
        _prescribed_displacements : list
            The prescribed displacements vector for the beam

        """
        return self._prescribed_displacements

    @prescribed_displacements.setter
    def prescribed_displacements(self, d):
        """
        Stores prescribed displacements

        Parameters
        -------
        d : list
            The prescribed displacement vector for the beam

        Returns
        -------
        None

        """
        if len(d) != len(self._restraints):
            raise ValueError(
                f"D must have same length as R ({len(self._restraints)}), got {len(d)}"
            )
        self._prescribed_displacements = d
        self._structure_version += 1

    def _set_element_type(self, i_span):
        """
        Stores element type for a span based on support conditions
        """
        raise NotImplementedError("Changing element type not supported")

    @property
    def no_spans(self):
        """
        Returns the no. of spans in the beam

        Returns
        -------
        no_spans : int
            The number of spans in the beam
        """
        return self._no_spans

    @property
    def no_restraints(self):
        """
        Returns the number of restraints of the beam

        Returns
        -------
        no_restraints : int
            The number of restraints in the beam
        """
        return len(self._restraints)

    @property
    def no_fixed_restraints(self):
        """
        Returns the number of fixed restraints of the beam (fully-supported DOFs)

        Returns
        -------
        no_fixed_restraints : int
            The number of fixed restraints in the beam
        """
        return len(np.where(np.array(self._restraints) == -1)[0])

    def _n_supports(self) -> int:
        """Number of nodes carrying any restraint (fixed or spring)."""
        R = list(self._restraints)
        return sum(
            1
            for i in range(self._no_spans + 1)
            if 2 * i + 1 < len(R) and (R[2 * i] != 0 or R[2 * i + 1] != 0)
        )

    def __repr__(self) -> str:
        n_sup = self._n_supports()
        n_loads = len(self.LM)
        return (
            f"Beam({self._no_spans} span{'' if self._no_spans == 1 else 's'}, "
            f"L={self._length:g}, "
            f"{n_sup} support{'' if n_sup == 1 else 's'}, "
            f"{n_loads} load{'' if n_loads == 1 else 's'})"
        )

    @property
    def length(self):
        """
        Returns
        -------
        length : float
            The total length of the beam
        """
        return self._length

    def get_local_span_coords(self, pos: float) -> (int, float):
        """
        Returns the span index and position in span for a position given in global
        coordinates on the beam

        Parameters
        ----------
        pos : float
            The position of interest in global coordinates along the length of the beam

        Returns
        -------
        ispan : int
            The index (1-based) of the span in which the point of interest falls
        pos_in_span : float
            The local coordinate along the member of the point of interest

        """
        if pos < 0.0 or pos > self.length:
            return -1, 0
        try:
            ispan = next(i - 1 for i, x in enumerate(self._terminal_coords) if x > pos)
        except StopIteration:  # at the end of the beam
            ispan = self._no_spans - 1
        pos_in_span = pos - self._terminal_coords[ispan]

        return ispan, pos_in_span

    def plot(
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

        By default this renders with matplotlib; saving to a ``.tex`` path (or
        passing ``tikz=True``) produces publication-quality TikZ/``stanli``
        output instead.  The beam structure (geometry, supports, internal
        hinges) is always drawn; the loads layer is optional and its source is
        selected with ``loads``:

        * ``None`` (default) - the beam's own load matrix ``self.LM``.
        * ``[]`` - draw the bare structure only.
        * a PyCBA load matrix, a :class:`~pycba.load_cases.LoadCase`, or a
          :class:`~pycba.load_cases.LoadCombination` (supply its
          :class:`~pycba.load_cases.LoadCases` via ``load_cases``).

        Parameters
        ----------
        loads : list | LoadCase | LoadCombination, optional
            The load source to draw.
        tikz : bool, optional
            Backend selector.  ``None`` (default) infers it from ``save``: a
            ``.tex`` target renders TikZ/``stanli``; anything else (or no
            ``save``) renders with matplotlib.  Pass ``True``/``False`` to
            force the backend - e.g. ``tikz=True`` to return the LaTeX source,
            or to render an otherwise-ambiguous ``.pdf`` target via TikZ.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into (matplotlib backend only); a new figure is
            created if omitted.
        save : str or pathlib.Path, optional
            If given, also write the visualisation to this path; the file
            extension selects the backend when ``tikz`` is ``None``.  A
            ``.tex`` target writes the TikZ/``stanli`` source; any other
            extension is saved by matplotlib (format inferred from the
            extension, e.g. ``.png``/``.pdf``/``.svg``).  Under the TikZ
            backend a ``.pdf`` target is compiled with ``pdflatex``.
        compile : bool
            Under the TikZ backend with ``save`` set, run ``pdflatex`` to also
            produce a PDF (requires a LaTeX install with ``stanli``).  A
            ``.pdf`` save target enables this automatically.  Ignored for the
            matplotlib backend.
        load_cases : pycba.load_cases.LoadCases, optional
            Required only when ``loads`` is a ``LoadCombination``.
        **kwargs
            Forwarded to the backend renderer:
            :meth:`pycba.render.BeamPlotter.render_mpl`
            (``dimensions``, ``labels``, ``load_values``, ``color``,
            ``units``) or, when ``tikz=True``,
            :meth:`pycba.render.BeamPlotter.render_tikz` (``standalone``,
            ``scale``, ``dimensions``, ``labels``, ``load_values``,
            ``units``).  ``units`` selects the display unit system (see
            :func:`pycba.set_units`).

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn into (matplotlib backend).
        str
            The LaTeX source, when the TikZ backend is selected and ``save`` is
            not given.
        pathlib.Path
            The written file, when the TikZ backend is selected with ``save``
            (the ``.tex`` path, or the ``.pdf`` when compiled).
        """
        from pathlib import Path
        from .render import BeamPlotter

        use_tikz = tikz
        if use_tikz is None:
            # Infer the backend from the output file: a ``.tex`` target means
            # TikZ, anything else (or no ``save``) means matplotlib.
            use_tikz = save is not None and Path(save).suffix.lower() == ".tex"

        plotter = BeamPlotter(self, loads, load_cases=load_cases)
        if use_tikz:
            if save is not None:
                # A .pdf target implies the compiled figure is wanted.
                if Path(save).suffix.lower() == ".pdf":
                    compile = True
                return plotter.save_tikz(save, compile=compile, **kwargs)
            return plotter.render_tikz(**kwargs)

        ax = plotter.render_mpl(ax=ax, **kwargs)
        if save is not None:
            ax.figure.savefig(save, bbox_inches="tight")
        return ax

    def to_tikz(self, loads=None, *, load_cases=None, **kwargs) -> str:
        """
        Generate a TikZ/``stanli`` representation of the beam.

        Loads are selected with ``loads`` exactly as for :meth:`plot`.

        Parameters
        ----------
        loads : list | LoadCase | LoadCombination, optional
            The load source to draw.
        load_cases : pycba.load_cases.LoadCases, optional
            Required only when ``loads`` is a ``LoadCombination``.
        **kwargs
            Forwarded to :meth:`pycba.render.BeamPlotter.render_tikz`
            (``standalone``, ``scale``, ``dimensions``, ``labels``,
            ``load_values``).

        Returns
        -------
        str
            The LaTeX source.
        """
        from .render import BeamPlotter

        return BeamPlotter(self, loads, load_cases=load_cases).render_tikz(**kwargs)

    def save_tikz(self, path, loads=None, *, compile=False, load_cases=None, **kwargs):
        """
        Write the TikZ/``stanli`` source to ``path`` (and optionally compile it).

        Loads are selected with ``loads`` exactly as for :meth:`plot`.

        Parameters
        ----------
        path : str or pathlib.Path
            Output ``.tex`` path.
        loads : list | LoadCase | LoadCombination, optional
            The load source to draw.
        compile : bool
            If ``True``, run ``pdflatex`` to also produce a PDF (requires a
            LaTeX install with the ``stanli`` package).
        load_cases : pycba.load_cases.LoadCases, optional
            Required only when ``loads`` is a ``LoadCombination``.

        Returns
        -------
        pathlib.Path
            The ``.tex`` path, or the produced ``.pdf`` when ``compile=True``.
        """
        from .render import BeamPlotter

        return BeamPlotter(self, loads, load_cases=load_cases).save_tikz(
            path, compile=compile, **kwargs
        )

    def get_ref(self, i_span: int) -> LoadCNL:
        """
        Returns Released End Forces for the member; that is, the Consistent Nodal Loads
        modified for the element type (i.e. releases)

        Parameters
        ----------
        ispan : int
            The index (1-based) of the span in which the point of interest falls

        Returns
        -------
        ref : LoadCNL
            The totalled CNL object for the member, considering all loads.

        """
        ref = np.zeros(4)
        L = self.mbr_lengths[i_span]
        eType = self.mbr_eletype[i_span]
        EI = self.mbr_EIs[i_span]

        for load in self._loads:
            if load.i_span != i_span:
                continue
            if isinstance(EI, SectionEI):
                ref += self._ref_nonprismatic(load, EI, L, eType)
            else:
                ref += load.get_ref(L, eType)
        return ref

    def _ref_nonprismatic(
        self, load, EI: SectionEI, L: float, eType: int
    ) -> np.ndarray:
        """
        Released end forces of a single load on a non-prismatic member.

        The fixed-end moments are obtained by the same flexibility integration
        used for the element stiffness.  On the simply-supported released
        element the load produces the moment diagram ``M(x)`` (taken from the
        load's :meth:`~pycba.load.Load.get_mbr_results`), and the primary end
        rotations are

        .. math::
            θ_{0} = \\Big[\\int_0^L \\frac{m_i(x)\\,M(x)}{EI(x)}\\,dx,\\;
                          \\int_0^L \\frac{m_j(x)\\,M(x)}{EI(x)}\\,dx\\Big]

        with ``m_i = 1 − x/L`` and ``m_j = −x/L`` (the same unit-moment
        diagrams used for the element flexibility).  The fixed-end moments
        follow directly as ``[M_a, M_b] = K_θ θ_0`` in PyCBA's nodal-moment
        sign convention, and the corresponding end shears are the
        simply-supported reactions plus the ``(M_a + M_b)/L`` couple.  Moment
        releases are then imposed by static condensation, mirroring
        :meth:`get_ref`.

        For a constant ``EI(x)`` this reproduces the prismatic
        :meth:`pycba.load.Load.get_ref` to machine precision.

        Parameters
        ----------
        load : pycba.load.Load
            The load object.
        EI : SectionEI
            The variable-rigidity description of the member.
        L : float
            The length of the member.
        eType : int
            The element type (1: FF, 2: FP, 3: PF, 4: PP).

        Returns
        -------
        np.ndarray, shape (4,)
            Released end force vector ``[Va, Ma, Vb, Mb]``.
        """
        eType = int(np.asarray(eType).item())
        # Primary (simply-supported) end rotations from flexibility integration.
        # For ordinary loads the integrand is the flexural curvature M(x)/EI(x);
        # for an imposed-curvature (initial-strain) load M = 0 and the free
        # curvature kappa_imp(x) is the primary curvature instead.
        #
        # The integral is split at the section breakpoints (segment joins and
        # pwl kinks) so that any EI discontinuity / kink is honoured exactly;
        # within each piece a fine Simpson grid captures the (possibly
        # non-polynomial, e.g. point-load-kinked) moment diagram.  For a single
        # constant segment this reproduces the prismatic get_ref.
        bps = EI.breakpoints
        edges = np.unique(np.concatenate([bps, [0.0, L]]))
        edges = edges[(edges >= -1e-12) & (edges <= L + 1e-12)]
        edges[0], edges[-1] = 0.0, L
        theta0 = np.zeros(2)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            n = 2001
            xx = np.linspace(a, b, n)
            M = load.get_mbr_results(xx, L).M
            curv = M / EI(xx)
            if isinstance(load, LoadIC):
                curv = curv + load.kappa_imp(xx)
            mi = 1.0 - xx / L
            mj = -xx / L
            theta0[0] += integrate.simpson(mi * curv, x=xx)
            theta0[1] += integrate.simpson(mj * curv, x=xx)

        # Fixed-end moments (PyCBA nodal-moment sign convention)
        Kth = self.k_theta(EI, L)
        FEM = Kth @ theta0
        Ma = FEM[0]
        Mb = FEM[1]

        # Simply-supported reactions: recover from the load's prismatic CNL by
        # removing the (prismatic) end-moment couple, leaving pure SS shears.
        cnl_p = load.get_cnl(L, 1)
        Va_ss = cnl_p.Va - (cnl_p.Ma + cnl_p.Mb) / L
        Vb_ss = cnl_p.Vb + (cnl_p.Ma + cnl_p.Mb) / L

        # Fixed-fixed released end forces with the *non-prismatic* moments
        ref = np.array(
            [
                Va_ss + (Ma + Mb) / L,
                Ma,
                Vb_ss - (Ma + Mb) / L,
                Mb,
            ]
        )

        if eType == 4:
            # Pinned-pinned: release both end moments.  Mirror the prismatic
            # :meth:`pycba.load.Load.get_ref` convention exactly (the vertical
            # correction is ``(Ma + Mb)/L`` applied to the fixed-fixed CNL),
            # so the constant-EI limit is reproduced.
            Va_ff, Vb_ff = ref[0], ref[2]
            ref[0] = Va_ff + (Ma + Mb) / L
            ref[1] = 0.0
            ref[2] = Vb_ff - (Ma + Mb) / L
            ref[3] = 0.0
        elif eType in (2, 3):
            # Single moment release: condense the released rotational DOF so
            # that its moment is nulled, mirroring the prismatic k_FP / k_PF.
            r = [3] if eType == 2 else [1]
            kff = self.k_nonprismatic(EI, L, 1)
            Krr = kff[np.ix_(r, r)]
            theta_r = np.linalg.solve(Krr, ref[r])
            ref = ref - kff[:, r] @ theta_r
        return ref

    def get_span_k(self, i_span: int) -> np.ndarray:
        """
        Returns the stiffness matrix for the ith span

        Parameters
        ----------
        ispan : int
            The index (1-based) of the span in which the point of interest falls

        Returns
        -------
        kb : np.ndarray
            The stiffness matrix for the member

        """
        EI = self.mbr_EIs[i_span]
        L = self.mbr_lengths[i_span]
        eType = self.mbr_eletype[i_span]

        # Non-prismatic (variable-EI) members are handled by flexibility
        # integration; the scalar/prismatic path below is unchanged.
        if isinstance(EI, SectionEI):
            return self.k_nonprismatic(EI, L, eType)

        if eType == 2:
            kb = self.k_FP(EI, L)
        elif eType == 3:
            kb = self.k_PF(EI, L)
        elif eType == 4:
            kb = self.k_PP(EI, L)
        else:
            kb = self.k_FF(EI, L)
        return kb

    def k_FF(self, EI: float, L: float) -> np.ndarray:
        """
        Stiffness matrix for a fixed-fixed element

        Parameters
        ----------
        EI : float
            The flexural rigidity for the member (assumed prismatic)
        L : float
            The length of the member

        Returns
        -------
        k : np.ndarray
            The stiffness matrix for the member
        """
        L2 = L**2
        L3 = L**3

        kfv = 12 * EI / L3
        kmv = 6 * EI / L2
        kft = kmv
        kmt = 4 * EI / L
        kmth = 2 * EI / L

        k = np.array(
            [
                [kfv, kft, -kfv, kft],
                [kmv, kmt, -kmv, kmth],
                [-kfv, -kft, kfv, -kft],
                [kft, kmth, -kft, kmt],
            ]
        )

        return k

    def k_FP(self, EI: float, L: float) -> np.ndarray:
        """
        Stiffness matrix for a fixed-pinned element

        Parameters
        ----------
        EI : float
            The flexural rigidity for the member (assumed prismatic)
        L : float
            The length of the member

        Returns
        -------
        k : np.ndarray
            The stiffness matrix for the member
        """
        L2 = L**2
        L3 = L**3

        kfv = 3 * EI / L3
        kmv = 3 * EI / L2
        kft = kmv
        kmt = 3 * EI / L

        k = np.array(
            [
                [kfv, kft, -kfv, 0],
                [kmv, kmt, -kmv, 0],
                [-kfv, -kft, kfv, 0],
                [0, 0, 0, 0],
            ]
        )

        return k

    def k_PF(self, EI: float, L: float) -> np.ndarray:
        """
        Stiffness matrix for a pinned-fixed element

        Parameters
        ----------
        EI : float
            The flexural rigidity for the member (assumed prismatic)
        L : float
            The length of the member

        Returns
        -------
        k : np.ndarray
            The stiffness matrix for the member
        """
        L2 = L**2
        L3 = L**3

        kfv = 3 * EI / L3
        kmv = 3 * EI / L2
        kft = kmv
        kmt = 3 * EI / L

        k = np.array(
            [
                [kfv, 0, -kfv, kft],
                [0, 0, 0, 0],
                [-kfv, 0, kfv, -kft],
                [kft, 0, -kft, kmt],
            ]
        )

        return k

    def k_PP(self, EI: float, L: float) -> np.ndarray:
        """
        Stiffness matrix for a pinned-pinned element

        Parameters
        ----------
        EI : float
            The flexural rigidity for the member (assumed prismatic)
        L : float
            The length of the member

        Returns
        -------
        k : np.ndarray
            The stiffness matrix for the member
        """

        k = np.zeros((4, 4))

        return k

    # ------------------------------------------------------------------
    #  Non-prismatic (variable-EI) element
    # ------------------------------------------------------------------
    @staticmethod
    def _gauss_nodes(EI: "SectionEI", L: float):
        """
        Breakpoint-aware Gauss-Legendre nodes/weights over the span ``[0, L]``.

        The flexibility integrals are evaluated **piece-by-piece between
        consecutive breakpoints** (segment joins and interior ``pwl`` kinks),
        rather than by a single global quadrature over a smoothed polynomial.
        Splitting at the breakpoints makes kinks (haunch -> flat) and steps
        (a discontinuous ``EI`` at a shared coordinate) exact.

        The per-piece integrand is ``m_a(x) m_b(x) / EI(x)``: an (at most)
        quadratic numerator over the piece's polynomial rigidity ``EI(x)``.
        For a **constant** piece this is a pure quadratic polynomial, integrated
        *exactly* by a 2-point Gauss rule -- so a single ``const`` segment (the
        prismatic limit) reproduces the closed-form stiffness to machine
        precision.  For a non-constant piece the integrand is a smooth (analytic,
        ``EI > 0``) rational function; Gauss-Legendre then converges
        exponentially, so an order ``>= 2*degree + 8`` (floored at 16) brings
        every linear / pwl / poly piece to machine precision as well.

        Parameters
        ----------
        EI : SectionEI
            The variable-rigidity description of the member.
        L : float
            The length of the member.

        Returns
        -------
        x, w : np.ndarray
            The concatenated Gauss nodes and weights mapped onto ``[0, L]``.
        """
        xs = []
        ws = []
        for p in EI.pieces:
            a, b = p.x0, p.x1
            if p.degree == 0:
                n = 2  # quadratic integrand -> 2-point Gauss is exact
            else:
                n = max(2 * p.degree + 8, 16)
            xi, wi = np.polynomial.legendre.leggauss(n)
            xs.append(0.5 * (b - a) * (xi + 1.0) + a)
            ws.append(0.5 * (b - a) * wi)
        return np.concatenate(xs), np.concatenate(ws)

    @staticmethod
    def _flexibility(EI: SectionEI, L: float) -> np.ndarray:
        """
        Rotational flexibility matrix of the simply-supported released element.

        Using the force (flexibility) method, the released (simply-supported)
        element is loaded by unit end moments giving the linear moment diagrams

        .. math::
            m_i(x) = 1 - x/L, \\qquad m_j(x) = -x/L

        (the ``j``-end diagram carries the sign required by PyCBA's
        counter-clockwise-positive nodal-moment convention, so that the
        resulting stiffness reproduces :meth:`k_FF` exactly for constant
        ``EI``) and the 2x2 rotational flexibility about the two end DOFs is

        .. math::
            F = \\begin{bmatrix}
                \\int_0^L \\frac{m_i^2}{EI(x)}\\,dx &
                \\int_0^L \\frac{m_i m_j}{EI(x)}\\,dx \\\\
                \\int_0^L \\frac{m_i m_j}{EI(x)}\\,dx &
                \\int_0^L \\frac{m_j^2}{EI(x)}\\,dx
            \\end{bmatrix}

        The integrals are evaluated by **breakpoint-aware** Gauss-Legendre
        quadrature (see :meth:`_gauss_nodes`): summed piece-by-piece between
        consecutive breakpoints, with a Gauss order sufficient for each piece.
        For a constant ``EI(x)`` (a single ``const`` segment) this reproduces
        the prismatic flexibility exactly.

        Parameters
        ----------
        EI : SectionEI
            The variable-rigidity description of the member.
        L : float
            The length of the member.

        Returns
        -------
        F : np.ndarray, shape (2, 2)
            The end-rotation flexibility matrix.
        """
        x, w = Beam._gauss_nodes(EI, L)

        EIx = EI(x)
        mi = 1.0 - x / L
        mj = -x / L

        F = np.zeros((2, 2))
        F[0, 0] = np.sum(w * mi * mi / EIx)
        F[0, 1] = np.sum(w * mi * mj / EIx)
        F[1, 0] = F[0, 1]
        F[1, 1] = np.sum(w * mj * mj / EIx)
        return F

    def k_theta(self, EI: SectionEI, L: float) -> np.ndarray:
        """
        2x2 moment-rotation stiffness of the (chord-relative) element ends.

        This is the inverse of the rotational flexibility matrix
        (see :meth:`_flexibility`), relating the end moments ``[M_i, M_j]`` to
        the chord-relative end rotations ``[theta_i, theta_j]``.

        Parameters
        ----------
        EI : SectionEI
            The variable-rigidity description of the member.
        L : float
            The length of the member.

        Returns
        -------
        np.ndarray, shape (2, 2)
            The end moment-rotation stiffness matrix ``K_theta = F^-1``.
        """
        return np.linalg.inv(self._flexibility(EI, L))

    def k_nonprismatic(self, EI: SectionEI, L: float, eType: int) -> np.ndarray:
        """
        4x4 stiffness matrix of a non-prismatic (variable-EI) element.

        The 2x2 moment-rotation stiffness ``K_theta`` from flexibility
        integration (see :meth:`k_theta`) is expanded to the full 4-DOF element
        using the kinematic transformation that removes the rigid-body chord
        rotation ``psi = (v_j - v_i)/L``:

        .. math::
            \\begin{bmatrix} \\theta_i \\\\ \\theta_j \\end{bmatrix}
            = T \\begin{bmatrix} v_i \\\\ \\theta_i \\\\ v_j \\\\ \\theta_j
            \\end{bmatrix},
            \\quad
            T = \\begin{bmatrix}
                1/L & 1 & -1/L & 0 \\\\
                1/L & 0 & -1/L & 1
            \\end{bmatrix}

        so the element stiffness is ``k = T^T K_theta T``.  The end shears
        emerge automatically as ``(M_i + M_j)/L`` from this transformation,
        matching PyCBA's DOF order ``[v_i, theta_i, v_j, theta_j]`` and sign
        convention.

        Moment releases (element types 2, 3, 4) are imposed by static
        condensation of the released rotational DOF(s), exactly mirroring the
        prismatic :meth:`k_FP`, :meth:`k_PF`, and :meth:`k_PP`: the released
        row/column are zeroed and their flexibility is condensed out.

        For a constant ``EI(x)`` this reproduces the closed-form prismatic
        :meth:`k_FF`, :meth:`k_FP`, :meth:`k_PF`, :meth:`k_PP` to machine
        precision.

        Parameters
        ----------
        EI : SectionEI
            The variable-rigidity description of the member.
        L : float
            The length of the member.
        eType : int
            The element type (1: FF, 2: FP, 3: PF, 4: PP).

        Returns
        -------
        k : np.ndarray, shape (4, 4)
            The element stiffness matrix.
        """
        Kth = self.k_theta(EI, L)
        T = np.array(
            [
                [1.0 / L, 1.0, -1.0 / L, 0.0],
                [1.0 / L, 0.0, -1.0 / L, 1.0],
            ]
        )
        k = T.T @ Kth @ T

        # Moment releases: condense out released rotational DOF(s).
        released = {1: [], 2: [3], 3: [1], 4: [1, 3]}[int(np.asarray(eType).item())]
        if released:
            k = self._condense(k, released)
        return k

    @staticmethod
    def _condense(k: np.ndarray, released: list) -> np.ndarray:
        """
        Statically condense released DOF(s) out of a 4x4 element stiffness.

        The released rows/columns are zeroed (as in the prismatic released
        matrices) and their stiffness is condensed into the retained DOFs via
        the Schur complement ``k_kk - k_kr k_rr^-1 k_rk``.

        Parameters
        ----------
        k : np.ndarray, shape (4, 4)
            The fully-fixed element stiffness matrix.
        released : list of int
            DOF indices to release (e.g. ``[3]`` for a pin at the ``j`` end).

        Returns
        -------
        np.ndarray, shape (4, 4)
            Condensed stiffness with released rows/columns zeroed.
        """
        kept = [i for i in range(4) if i not in released]
        Kkk = k[np.ix_(kept, kept)]
        Krr = k[np.ix_(released, released)]
        Kkr = k[np.ix_(kept, released)]
        Krk = k[np.ix_(released, kept)]
        Kcond = Kkk - Kkr @ np.linalg.inv(Krr) @ Krk

        kc = np.zeros((4, 4))
        kc[np.ix_(kept, kept)] = Kcond
        return kc
