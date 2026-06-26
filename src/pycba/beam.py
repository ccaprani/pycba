"""
PyCBA - Beam Class definition
"""
from typing import Optional, Sequence, Union
import numpy as np
from scipy import integrate
from .load import parse_LM, LoadType, LoadMatrix, LoadCNL, LoadIC
from .types import MemberType
from .section import SectionEI
from .utils import supports_to_R, SupportType


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
        supports: Optional[Sequence[SupportType]] = None,
        GAv: Optional[Union[float, "SectionEI", Sequence]] = None,
        kf: Optional[Union[float, Sequence]] = None,
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
            A vector describing the support conditions at each member end (the
            low-level restraint vector).  Provide either ``R`` or the friendlier
            ``supports``, not both.
        LM : Optional[list[list[Union[int, float]]]]
            The load matrix: a list of loads on the beam; each load with several
            parameters.
        eletype : Optional[np.ndarray]
            A vector of the member types, each an integer code ``1-4``, a
            :class:`~pycba.MemberType`, or a name string (``"FF"``/``"FP"``/
            ``"PF"``/``"PP"``). Defaults to a fixed-fixed element.
        D : Optional[np.ndarray]
            A vector of prescribed displacements. Must have same length as R.
            Use None for DOFs without prescribed displacement.
        supports : Optional[Sequence[str or [float, float]]]
            A friendlier alternative to ``R``: one entry per node (left to
            right), each a support name (``"p"``/``"pin"``/``"pinned"``,
            ``"r"``/``"roller"``, ``"e"``/``"encastre"``/``"fixed"``,
            ``"f"``/``"free"``) or a raw ``[vertical, rotation]`` DOF pair (e.g.
            ``[5e4, 0]`` for a vertical spring).  Lowered to ``R`` via
            :func:`~pycba.supports_to_R`.  Mutually exclusive with ``R``.
        GAv : Optional[float, SectionEI, or sequence]
            Transverse **shear rigidity** ``G·A_v`` of each span (``A_v`` the
            shear area).  A span with a finite ``GAv`` is analysed as a
            shear-deformable **Timoshenko** element; ``None`` (the default)
            keeps the member on the exact Euler–Bernoulli path, bit-for-bit
            unchanged.  Like ``EI``: a single scalar (or one
            :class:`~pycba.section.SectionEI` describing ``GAv(x)``) applies to
            all spans, otherwise one entry per span (each ``None``, a scalar, or
            a :class:`~pycba.section.SectionEI`).
        kf : Optional[float or sequence]
            Winkler **foundation modulus** (modulus of subgrade reaction per
            unit beam length).  A span with a finite ``kf`` rests on an elastic
            (Winkler) foundation, modelled as a statically-condensed
            beam-on-elastic-foundation super-element; ``None`` (default) leaves
            the span unsupported by a foundation.  A scalar applies to all spans,
            otherwise one entry per span.


        Returns
        -------
        None.
        """
        self._no_spans = 0
        self._no_restraints = 0
        self._length = 0
        self.mbr_lengths = []
        self.mbr_EIs = []
        self.mbr_GAv = []
        self.mbr_kf = []
        self.mbr_eletype = []
        # Cache of foundation super-elements, keyed by span index (rebuilt only
        # when the structure changes, not when loads change).
        self._found_cache = {}
        self._restraints = []
        self._prescribed_displacements = []
        self._loads = []
        self.LM = []
        self._terminal_coords = [0.0]
        # Bumped on any change to the *structure* (geometry, rigidity,
        # restraints, prescribed displacements) so a cached stability check can
        # be invalidated.  Load changes deliberately do not bump it.
        self._structure_version = 0

        # The friendly ``supports`` list is sugar for the low-level ``R`` vector;
        # lower it here so the rest of the constructor is unchanged.
        if supports is not None:
            if R is not None:
                raise ValueError("Pass either R or supports, not both.")
            if L is None:
                raise ValueError("supports requires L to set the node count.")
            R = supports_to_R(supports, n_nodes=len(L) + 1)

        if L is not None and eletype is not None:
            # Normalise GAv / kf to one entry per span (None => off).
            GAv_list = self._broadcast_GAv(GAv, len(L))
            kf_list = self._broadcast_kf(kf, len(L))
            # A single rigidity (scalar EI, or one SectionEI) applies to all
            # spans; otherwise one rigidity per span is required.
            if isinstance(EI, (float, int, SectionEI)):
                for l, et, gav, k in zip(L, eletype, GAv_list, kf_list):
                    self.add_span(l, EI, et, gav, k)
            else:
                if len(L) == len(EI):
                    for l, ei, et, gav, k in zip(L, EI, eletype, GAv_list, kf_list):
                        self.add_span(l, ei, et, gav, k)
                else:
                    raise ValueError("Define EI for each span")
            if R is None:
                raise ValueError("Provide support conditions via R or supports.")
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

    @staticmethod
    def _broadcast_GAv(GAv, n_spans: int) -> list:
        """
        Normalise the ``GAv`` argument to one entry per span.

        ``None`` (the default) marks every span as Euler–Bernoulli.  A scalar
        or single :class:`~pycba.section.SectionEI` is applied to all spans; a
        sequence must give one entry per span (each ``None``, a scalar, or a
        :class:`~pycba.section.SectionEI`).
        """
        if GAv is None:
            return [None] * n_spans
        if isinstance(GAv, (float, int, SectionEI)):
            return [GAv] * n_spans
        GAv = list(GAv)
        if len(GAv) != n_spans:
            raise ValueError("Define GAv for each span (or pass a scalar/None).")
        return GAv

    @staticmethod
    def _broadcast_kf(kf, n_spans: int) -> list:
        """
        Normalise the Winkler foundation modulus ``kf`` to one entry per span.

        ``None`` (the default) marks every span as unsupported by a foundation.
        A scalar is applied to all spans; a sequence must give one entry per
        span (each ``None`` or a scalar modulus).
        """
        if kf is None:
            return [None] * n_spans
        if isinstance(kf, (float, int)):
            return [kf] * n_spans
        kf = list(kf)
        if len(kf) != n_spans:
            raise ValueError("Define kf for each span (or pass a scalar/None).")
        return kf

    def add_span(
        self,
        L: float,
        EI: float,
        eletype: Union[int, str, MemberType] = 1,
        GAv: Optional[Union[float, "SectionEI"]] = None,
        kf: Optional[float] = None,
    ):
        """
        Add a span to the continuous beam

        Parameters
        ----------
        L : float
            The length of the member.
        EI : np.ndarray
            The flexural rigidity of the member.
        eletype : int, str, or pycba.MemberType
            The element type for the member, as the integer code ``1-4``, a
            :class:`~pycba.MemberType` (e.g. ``MemberType.FP``), or its
            case-insensitive name (e.g. ``"FP"``).  See :class:`~pycba.MemberType`.
        GAv : float or pycba.section.SectionEI, optional
            Transverse shear rigidity ``G·A_v`` of the member.  When given (and
            finite) the member is a shear-deformable **Timoshenko** element;
            ``None`` (default) keeps the exact Euler–Bernoulli element.  A
            :class:`~pycba.section.SectionEI` describes a variable ``GAv(x)``.
        kf : float, optional
            Winkler foundation modulus.  When given, the member rests on an
            elastic (Winkler) foundation (prismatic, fixed-fixed, no ``GAv``).

        Returns
        -------
        None.

        """
        if isinstance(EI, SectionEI):
            # A non-prismatic section must span the full member length so its
            # piecewise EI(x) covers the element exactly.
            EI.validate_length(L)
        if isinstance(GAv, SectionEI):
            # A variable shear rigidity must likewise cover the full member.
            GAv.validate_length(L)
        self.mbr_lengths.append(L)
        self.mbr_EIs.append(EI)
        self.mbr_GAv.append(GAv)
        self.mbr_kf.append(kf)
        self.mbr_eletype.append(MemberType.coerce(eletype))
        self._no_spans = len(self.mbr_lengths)
        self._length += L
        self._terminal_coords.append(self._terminal_coords[-1] + L)
        self._structure_version += 1
        self._found_cache = {}  # invalidate foundation super-elements

    def add_member(
        self,
        L: float,
        EI: float,
        mbr_type: Union[int, str, MemberType] = MemberType.FF,
        GAv: Optional[Union[float, "SectionEI"]] = None,
        kf: Optional[float] = None,
    ):
        """
        Add a member (span) to the beam, naming its type.

        A friendlier alias of :meth:`add_span` whose ``mbr_type`` accepts a
        :class:`~pycba.MemberType`, a name string (``"FF"``/``"FP"``/``"PF"``/
        ``"PP"``), or the integer code ``1-4``.

        Parameters
        ----------
        L : float
            The length of the member.
        EI : float or pycba.section.SectionEI
            The flexural rigidity of the member.
        mbr_type : pycba.MemberType, str, or int
            The member type / moment-release pattern (default ``FF``).
        GAv : float or pycba.section.SectionEI, optional
            Transverse shear rigidity ``G·A_v``; when given the member is a
            shear-deformable Timoshenko element (see :meth:`add_span`).
        kf : float, optional
            Winkler foundation modulus; when given the member rests on an elastic
            (Winkler) foundation (see :meth:`add_span`).
        """
        self.add_span(L, EI, mbr_type, GAv, kf)

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

    def _foundation(self, i_span: int):
        """
        Return the (cached) beam-on-elastic-foundation super-element for a
        foundation span, after checking the supported restrictions.
        """
        elem = self._found_cache.get(i_span)
        if elem is not None:
            return elem

        EI = self.mbr_EIs[i_span]
        GAv = self.mbr_GAv[i_span] if i_span < len(self.mbr_GAv) else None
        eType = int(np.asarray(self.mbr_eletype[i_span]).item())
        if isinstance(EI, SectionEI):
            raise NotImplementedError(
                "A Winkler foundation currently requires a prismatic (scalar EI) span."
            )
        if GAv is not None:
            raise NotImplementedError(
                "A Winkler foundation cannot yet be combined with shear "
                "flexibility (GAv) on the same span."
            )
        if eType != 1:
            raise NotImplementedError(
                "A Winkler foundation span must be fixed-fixed (the default "
                "element type); moment releases on a foundation span are not "
                "yet supported."
            )

        from .foundation import FoundationElement

        elem = FoundationElement(self.mbr_lengths[i_span], EI, self.mbr_kf[i_span])
        self._found_cache[i_span] = elem
        return elem

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
        GAv = self.mbr_GAv[i_span] if i_span < len(self.mbr_GAv) else None

        # Winkler foundation span: condense the sub-element load vector.
        if i_span < len(self.mbr_kf) and self.mbr_kf[i_span] is not None:
            loads = [ld for ld in self._loads if ld.i_span == i_span]
            return self._foundation(i_span).ref(loads)

        for load in self._loads:
            if load.i_span != i_span:
                continue
            if GAv is not None:
                ref += self._ref_timoshenko(load, EI, GAv, L, eType)
            elif isinstance(EI, SectionEI):
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
        GAv = self.mbr_GAv[i_span] if i_span < len(self.mbr_GAv) else None

        # Winkler foundation: the span is a condensed beam-on-elastic-foundation
        # super-element (see :meth:`_foundation`).
        if i_span < len(self.mbr_kf) and self.mbr_kf[i_span] is not None:
            return self._foundation(i_span).k

        # Shear-deformable (Timoshenko) member: opt-in via a finite ``GAv``.  A
        # variable EI and/or GAv goes through the flexibility integrator; the
        # common prismatic case uses the closed-form element and static
        # condensation for the released variants.
        if GAv is not None:
            if isinstance(EI, SectionEI) or isinstance(GAv, SectionEI):
                return self.k_timoshenko(EI, GAv, L, eType)
            kff = self.k_FF_timo(EI, GAv, L)
            released = {1: [], 2: [3], 3: [1], 4: [1, 3]}[int(np.asarray(eType).item())]
            return self._condense(kff, released) if released else kff

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

    # ------------------------------------------------------------------
    #  Timoshenko (shear-deformable) element
    # ------------------------------------------------------------------
    def k_FF_timo(self, EI: float, GAv: float, L: float) -> np.ndarray:
        """
        Stiffness matrix for a prismatic fixed-fixed Timoshenko element.

        The standard two-node shear-flexible element introduces the
        dimensionless shear parameter

        .. math::
            \\Phi = \\frac{12 EI}{G A_v L^2}

        so the element stiffness is

        .. math::
            k = \\frac{EI}{(1+\\Phi)L^3}
            \\begin{bmatrix}
                12 & 6L & -12 & 6L \\\\
                6L & (4+\\Phi)L^2 & -6L & (2-\\Phi)L^2 \\\\
                -12 & -6L & 12 & -6L \\\\
                6L & (2-\\Phi)L^2 & -6L & (4+\\Phi)L^2
            \\end{bmatrix}

        As ``GAv -> inf`` (``\\Phi -> 0``) this reduces exactly to the
        Euler–Bernoulli :meth:`k_FF`.  The DOF order and sign convention match
        the rest of PyCBA, ``[v_i, theta_i, v_j, theta_j]``, with ``theta`` the
        cross-section rotation.

        Parameters
        ----------
        EI : float
            The flexural rigidity of the member (assumed prismatic).
        GAv : float
            The transverse shear rigidity ``G·A_v`` of the member.
        L : float
            The length of the member.

        Returns
        -------
        k : np.ndarray, shape (4, 4)
            The element stiffness matrix.
        """
        L2 = L**2
        L3 = L**3
        phi = 12.0 * EI / (GAv * L2)
        c = EI / (1.0 + phi)

        kfv = 12.0 * c / L3
        kft = 6.0 * c / L2
        kmt = (4.0 + phi) * c / L
        kmth = (2.0 - phi) * c / L

        k = np.array(
            [
                [kfv, kft, -kfv, kft],
                [kft, kmt, -kft, kmth],
                [-kfv, -kft, kfv, -kft],
                [kft, kmth, -kft, kmt],
            ]
        )
        return k

    def _shear_flexibility(self, GAv: Union[float, "SectionEI"], L: float) -> float:
        """
        Shear contribution to the end-rotation flexibility.

        On the released (simply-supported) element a unit end moment produces a
        constant shear ``v = -1/L`` (the support reactions), so the shear
        strain energy adds the *same* term

        .. math::
            \\frac{1}{L^2}\\int_0^L \\frac{1}{G A_v(x)}\\,dx

        to **every** entry of the 2x2 end-rotation flexibility matrix.  For a
        constant ``GAv`` this is ``1/(L\\,G A_v)``; for a variable ``GAv(x)``
        (a :class:`~pycba.section.SectionEI`) the integral is evaluated by the
        same breakpoint-aware Gauss quadrature used for the flexural terms.

        Parameters
        ----------
        GAv : float or SectionEI
            The shear rigidity (constant or variable along the member).
        L : float
            The length of the member.

        Returns
        -------
        float
            The scalar shear-flexibility term added to all entries of ``F``.
        """
        if isinstance(GAv, SectionEI):
            xg, wg = self._gauss_nodes(GAv, L)
            integral = float(np.sum(wg / GAv(xg)))
        else:
            integral = L / GAv
        return integral / (L * L)

    def _timo_flexibility(
        self, EI: Union[float, "SectionEI"], GAv: Union[float, "SectionEI"], L: float
    ) -> np.ndarray:
        """
        2x2 end moment-rotation flexibility of a Timoshenko element.

        This is the Euler–Bernoulli flexural flexibility (closed-form for a
        scalar ``EI``, or :meth:`_flexibility` for a
        :class:`~pycba.section.SectionEI`) plus the shear contribution from
        :meth:`_shear_flexibility`.  Inverting it gives the moment-rotation
        stiffness ``K_theta`` used to build the element (see
        :meth:`k_timoshenko`).  For a constant ``EI``/``GAv`` it reproduces the
        closed-form prismatic element :meth:`k_FF_timo` to machine precision.

        Parameters
        ----------
        EI : float or SectionEI
            The flexural rigidity (constant or variable).
        GAv : float or SectionEI
            The shear rigidity (constant or variable).
        L : float
            The length of the member.

        Returns
        -------
        np.ndarray, shape (2, 2)
            The end moment-rotation flexibility matrix.
        """
        if isinstance(EI, SectionEI):
            F = self._flexibility(EI, L)
        else:
            F = np.array(
                [
                    [L / (3.0 * EI), -L / (6.0 * EI)],
                    [-L / (6.0 * EI), L / (3.0 * EI)],
                ]
            )
        return F + self._shear_flexibility(GAv, L)

    def k_timoshenko(
        self,
        EI: Union[float, "SectionEI"],
        GAv: Union[float, "SectionEI"],
        L: float,
        eType: int,
    ) -> np.ndarray:
        """
        4x4 Timoshenko element stiffness for scalar or variable ``EI``/``GAv``.

        The 2x2 moment-rotation stiffness ``K_theta`` (the inverse of the
        shear-augmented flexibility, see :meth:`_timo_flexibility`) is expanded
        to the full 4-DOF element by the same kinematic chord transformation
        used for the non-prismatic element (see :meth:`k_nonprismatic`), and
        moment releases (element types 2, 3, 4) are imposed by static
        condensation.  This single path covers prismatic and non-prismatic,
        constant- and variable-shear members; for a constant ``EI``/``GAv`` it
        reproduces :meth:`k_FF_timo` (and hence the Euler–Bernoulli element as
        ``GAv -> inf``) to machine precision.

        Parameters
        ----------
        EI : float or SectionEI
            The flexural rigidity (constant or variable).
        GAv : float or SectionEI
            The shear rigidity (constant or variable).
        L : float
            The length of the member.
        eType : int
            The element type (1: FF, 2: FP, 3: PF, 4: PP).

        Returns
        -------
        k : np.ndarray, shape (4, 4)
            The element stiffness matrix.
        """
        Kth = np.linalg.inv(self._timo_flexibility(EI, GAv, L))
        T = np.array(
            [
                [1.0 / L, 1.0, -1.0 / L, 0.0],
                [1.0 / L, 0.0, -1.0 / L, 1.0],
            ]
        )
        k = T.T @ Kth @ T

        released = {1: [], 2: [3], 3: [1], 4: [1, 3]}[int(np.asarray(eType).item())]
        if released:
            k = self._condense(k, released)
        return k

    def _timo_theta0(
        self,
        load,
        EI: Union[float, "SectionEI"],
        GAv: Union[float, "SectionEI"],
        L: float,
    ) -> np.ndarray:
        """
        Primary end rotations of a load on a released Timoshenko span.

        The released (simply-supported) span carries the load's moment diagram
        ``M(x)`` and shear ``V(x)`` (from :meth:`~pycba.load.Load.get_mbr_results`).
        The end rotations work-conjugate to the end moments combine the
        flexural and shear contributions,

        .. math::
            \\theta_{0,a} = \\int_0^L m_a\\,\\frac{M}{EI}\\,dx
                          + \\int_0^L v\\,\\frac{V}{G A_v}\\,dx, \\quad
            \\theta_{0,b} = \\int_0^L m_b\\,\\frac{M}{EI}\\,dx
                          + \\int_0^L v\\,\\frac{V}{G A_v}\\,dx

        with the unit-moment diagrams ``m_a = 1 - x/L``, ``m_b = -x/L`` and the
        constant unit-moment shear ``v = -1/L``.  The integral is split at the
        section breakpoints so any ``EI``/``GAv`` kink or step is honoured
        exactly.  Used only for variable ``EI``/``GAv``; the prismatic case has
        an exact closed-form transform in :meth:`_ref_timoshenko`.

        Parameters
        ----------
        load : pycba.load.Load
            The load object.
        EI : float or SectionEI
            The flexural rigidity (constant or variable).
        GAv : float or SectionEI
            The shear rigidity (constant or variable).
        L : float
            The length of the member.

        Returns
        -------
        np.ndarray, shape (2,)
            The primary end rotations ``[theta_0a, theta_0b]``.
        """
        bps = []
        if isinstance(EI, SectionEI):
            bps += list(EI.breakpoints)
        if isinstance(GAv, SectionEI):
            bps += list(GAv.breakpoints)
        if bps:
            edges = np.unique(np.concatenate([[0.0, L], bps]))
            edges = edges[(edges >= -1e-12) & (edges <= L + 1e-12)]
            edges[0], edges[-1] = 0.0, L
        else:
            edges = np.array([0.0, L])

        theta0 = np.zeros(2)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            n = 2001
            xx = np.linspace(a, b, n)
            mr = load.get_mbr_results(xx, L)
            EIx = EI(xx) if isinstance(EI, SectionEI) else EI
            curv = mr.M / EIx
            if isinstance(load, LoadIC):
                curv = curv + load.kappa_imp(xx)
            GAvx = GAv(xx) if isinstance(GAv, SectionEI) else GAv
            gamma = mr.V / GAvx
            mi = 1.0 - xx / L
            mj = -xx / L
            vi = -1.0 / L  # unit-moment shear (constant), same for both ends
            shear = integrate.simpson(vi * gamma, x=xx)
            theta0[0] += integrate.simpson(mi * curv, x=xx) + shear
            theta0[1] += integrate.simpson(mj * curv, x=xx) + shear
        return theta0

    def _ref_timoshenko(
        self,
        load,
        EI: Union[float, "SectionEI"],
        GAv: Union[float, "SectionEI"],
        L: float,
        eType: int,
    ) -> np.ndarray:
        """
        Released end forces of a single load on a Timoshenko member.

        The fixed-end moments are the Timoshenko moment-rotation stiffness
        applied to the released-span end rotations.  For a **prismatic** member
        the released end rotations are purely flexural (the shear term
        integrates to zero on a simply-supported span), giving the exact
        closed-form transform of the Euler–Bernoulli fixed-end moments

        .. math::
            \\begin{bmatrix} M_a \\\\ M_b \\end{bmatrix}_{T}
            = \\frac{1}{2(1+\\Phi)}
            \\begin{bmatrix} 2+\\Phi & -\\Phi \\\\ -\\Phi & 2+\\Phi \\end{bmatrix}
            \\begin{bmatrix} M_a \\\\ M_b \\end{bmatrix}_{EB},

        which recovers the Euler–Bernoulli moments as ``\\Phi -> 0``.  For a
        **variable** ``EI``/``GAv`` the end rotations are obtained from
        :meth:`_timo_theta0` and combined with the shear-augmented
        moment-rotation stiffness.  The end shears follow from statics and any
        moment releases are imposed by static condensation, mirroring
        :meth:`_ref_nonprismatic`.

        Parameters
        ----------
        load : pycba.load.Load
            The load object.
        EI : float or SectionEI
            The flexural rigidity (constant or variable).
        GAv : float or SectionEI
            The shear rigidity (constant or variable).
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
        prismatic = not isinstance(EI, SectionEI) and not isinstance(GAv, SectionEI)
        cnl = load.get_cnl(L, 1)  # fixed-fixed Euler-Bernoulli CNL

        if prismatic:
            phi = 12.0 * EI / (GAv * L**2)
            denom = 2.0 * (1.0 + phi)
            Ma = ((2.0 + phi) * cnl.Ma - phi * cnl.Mb) / denom
            Mb = (-phi * cnl.Ma + (2.0 + phi) * cnl.Mb) / denom
        else:
            theta0 = self._timo_theta0(load, EI, GAv, L)
            Kth = np.linalg.inv(self._timo_flexibility(EI, GAv, L))
            FEM = Kth @ theta0
            Ma, Mb = FEM[0], FEM[1]

        # Simply-supported reactions: recover from the prismatic CNL by removing
        # the (prismatic) end-moment couple, leaving pure SS shears.
        Va_ss = cnl.Va - (cnl.Ma + cnl.Mb) / L
        Vb_ss = cnl.Vb + (cnl.Ma + cnl.Mb) / L

        ref = np.array(
            [
                Va_ss + (Ma + Mb) / L,
                Ma,
                Vb_ss - (Ma + Mb) / L,
                Mb,
            ]
        )

        if eType == 4:
            # Pinned-pinned: release both end moments (mirror _ref_nonprismatic).
            Va_ff, Vb_ff = ref[0], ref[2]
            ref[0] = Va_ff + (Ma + Mb) / L
            ref[1] = 0.0
            ref[2] = Vb_ff - (Ma + Mb) / L
            ref[3] = 0.0
        elif eType in (2, 3):
            # Single moment release: condense the released rotational DOF using
            # the Timoshenko fixed-fixed stiffness.
            r = [3] if eType == 2 else [1]
            kff = self.k_timoshenko(EI, GAv, L, 1)
            Krr = kff[np.ix_(r, r)]
            theta_r = np.linalg.solve(Krr, ref[r])
            ref = ref - kff[:, r] @ theta_r
        return ref
