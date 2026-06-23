"""
PyCBA - Continuous Beam Analysis - Load Cases and Combinations
"""

from __future__ import annotations  # https://bit.ly/3KYiL2o
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
from .beam import Beam
from .analysis import BeamAnalysis
from .results import Envelopes
from .load import LoadMatrix, factor_LM


__all__ = [
    "LoadCase",
    "LoadCombination",
    "LoadCases",
    "build_pycba_model",
    "analyse_load_case",
    "analyze_load_case",
    "collect_response_matrix",
    "additive_envelope",
    "sign_selective_envelope",
    "make_patterned_udl",
    "make_span_udl_cases",
    "plot_response_envelope",
    "plot_load_patterns",
]


@dataclass
class LoadCase:
    """
    Neutral load-case definition for independent full-beam analyses.

    ``loads`` are ordinary PyCBA load-matrix rows. Span numbers inside each
    load row therefore remain 1-based, matching :class:`BeamAnalysis`.
    ``loaded_spans`` is stored as 0-based indices for convenient Python use.
    """

    name: str
    loads: list = field(default_factory=list)
    loaded_spans: tuple[int, ...] = ()
    metadata: Optional[dict] = None

    def add_load(self, load: list) -> LoadCase:
        """Append one raw PyCBA load-matrix row."""

        self.loads.append(deepcopy(load))
        return self

    def to_LM(self) -> LoadMatrix:
        """Return a copy of this load case's PyCBA load matrix."""

        return deepcopy(self.loads)

    def add_udl(self, i_span: int, w: float) -> LoadCase:
        """Append a full-span UDL to this load case."""

        return self.add_load([i_span, 1, w])

    def add_pl(self, i_span: int, p: float, a: float) -> LoadCase:
        """Append a point load to this load case."""

        return self.add_load([i_span, 2, p, a])

    def add_pudl(self, i_span: int, w: float, a: float, c: float) -> LoadCase:
        """Append a partial UDL to this load case."""

        return self.add_load([i_span, 3, w, a, c])

    def add_segment_udl(
        self, beam_model: Union[BeamAnalysis, Beam], x0: float, x1: float, w: float
    ) -> LoadCase:
        """Append a UDL covering a global beam segment from ``x0`` to ``x1``."""

        for load in _make_segment_udl(beam_model, x0, x1, w):
            self.add_load(load)
        return self

    def add_all_spans_udl(
        self, beam_model: Union[BeamAnalysis, Beam], w: float
    ) -> LoadCase:
        """
        Append a full-span UDL of intensity ``w`` to every span.

        This replaces the hand-written
        ``for i in range(1, no_spans + 1): case.add_udl(i, w)`` loop. The span
        count is read from ``beam_model``, which is required here (mirroring
        :meth:`add_segment_udl`).

        Parameters
        ----------
        beam_model : BeamAnalysis or Beam
            The beam definition used to count spans.
        w : float
            UDL intensity applied to every span.

        Returns
        -------
        LoadCase
            ``self``, for fluent chaining.

        Notes
        -----
        This adds all spans into the one (this) load case, unlike
        :meth:`LoadCases.add_span_udl`, which creates one separate case per span.
        """

        no_spans = _template_beam(beam_model).no_spans
        for i_span in range(1, no_spans + 1):
            self.add_udl(i_span, w)
        return self

    def add_ml(self, i_span: int, m: float, a: float) -> LoadCase:
        """Append a concentrated moment load to this load case."""

        return self.add_load([i_span, 4, m, a])

    def add_trap(
        self,
        i_span: int,
        w1: float,
        w2: float,
        a: Optional[float] = None,
        c: Optional[float] = None,
    ) -> LoadCase:
        """Append a full-span or partial trapezoidal load to this load case."""

        if a is not None and c is None:
            raise ValueError("If 'a' is specified, 'c' must also be provided")
        if a is None:
            return self.add_load([i_span, 5, w1, w2])
        return self.add_load([i_span, 5, w1, w2, a, c])

    def add_ic(self, i_span: int, kappa) -> LoadCase:
        """Append an imposed-curvature load to this load case."""

        coeffs = np.atleast_1d(np.asarray(kappa, dtype=float)).tolist()
        return self.add_load([i_span, 6] + coeffs)


@dataclass
class LoadCombination:
    """
    Named linear combination of load cases.

    ``factors`` may be a numeric vector in load-case order, or a mapping keyed
    by load-case name or zero-based load-case index.
    """

    name: str
    factors: Union[Sequence[float], Mapping[Union[str, int], float]]
    metadata: Optional[dict] = None

    def _resolve_load_cases(self, load_cases: Optional[LoadCases]) -> LoadCases:
        """
        Return an explicit or bound :class:`LoadCases` basis.

        ``load_cases`` takes precedence when supplied. Otherwise the basis bound
        by :meth:`LoadCases.target_combination` (stored as ``_bound``) is used.
        Combinations constructed directly have no bound basis and must be passed
        ``load_cases`` explicitly.
        """

        lc = load_cases if load_cases is not None else getattr(self, "_bound", None)
        if lc is None:
            raise ValueError(
                "No LoadCases bound to this combination; pass load_cases=... or "
                "create it via LoadCases.target_combination()."
            )
        return lc

    def factor_vector(self, load_cases: Optional[LoadCases] = None) -> np.ndarray:
        """
        Return this combination's factors in load-case order.

        Parameters
        ----------
        load_cases : LoadCases, optional
            The basis collection. When omitted, the basis bound by
            :meth:`LoadCases.target_combination` is used.
        """

        return self._resolve_load_cases(load_cases).factors(self)

    def to_LM(self, load_cases: Optional[LoadCases] = None) -> LoadMatrix:
        """
        Return the factored PyCBA load matrix for this combination.

        Parameters
        ----------
        load_cases : LoadCases, optional
            The basis collection. When omitted, the basis bound by
            :meth:`LoadCases.target_combination` is used.
        """

        return self._resolve_load_cases(load_cases).combined_loads(self)

    def to_load_case(self, load_cases: Optional[LoadCases] = None) -> LoadCase:
        """
        Return this combination as one ordinary factored ``LoadCase``.

        Parameters
        ----------
        load_cases : LoadCases, optional
            The basis collection. When omitted, the basis bound by
            :meth:`LoadCases.target_combination` is used.
        """

        loads = self.to_LM(load_cases)
        return LoadCase(
            name=self.name,
            loads=loads,
            loaded_spans=_load_case_loaded_spans(loads),
            metadata=deepcopy(self.metadata) if self.metadata is not None else None,
        )

    def response(
        self, load_cases: Optional[LoadCases] = None, response: str = "M"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return this combination's response.

        Parameters
        ----------
        load_cases : LoadCases, optional
            The basis collection. When omitted, the basis bound by
            :meth:`LoadCases.target_combination` is used.
        response : str, optional
            Station response to return (default ``"M"``).
        """

        return self._resolve_load_cases(load_cases).combine(self, response=response)

    def analyze(
        self, load_cases: Optional[LoadCases] = None, npts: Optional[int] = None
    ) -> BeamAnalysis:
        """
        Analyse this combination as one PyCBA beam analysis.

        Parameters
        ----------
        load_cases : LoadCases, optional
            The basis collection. When omitted, the basis bound by
            :meth:`LoadCases.target_combination` is used.
        npts : int, optional
            Number of evaluation points along a member.
        """

        return self._resolve_load_cases(load_cases).analyze_combination(
            self, npts=npts
        )

    analyse = analyze

    def envelope(
        self, load_cases: Optional[LoadCases] = None, npts: Optional[int] = None
    ) -> Envelopes:
        """
        Analyse this combination and return a plottable :class:`Envelopes`.

        This lands a target or explicit combination directly on a first-class
        :class:`pycba.results.Envelopes`, which can be merged with
        :meth:`Envelopes.combine` (or the ``|``/``+`` operators) and plotted.

        Parameters
        ----------
        load_cases : LoadCases, optional
            The basis collection. When omitted, the basis bound by
            :meth:`LoadCases.target_combination` is used.
        npts : int, optional
            Number of evaluation points along a member.

        Returns
        -------
        Envelopes
            A single-result :class:`pycba.results.Envelopes` for this combination.
        """

        model = self.analyze(load_cases, npts=npts)
        return Envelopes([model.beam_results])


class LoadCases:
    """
    Collection of named load cases for one PyCBA beam model.

    The collection is list-like for simple scripting, but also provides methods
    to analyse every case, form response matrices, and combine cases with
    arbitrary linear factors.
    """

    def __init__(
        self,
        beam_model: Union[BeamAnalysis, Beam],
        load_cases: Optional[Sequence[LoadCase]] = None,
    ):
        self.beam_model = beam_model
        self._cases: list[LoadCase] = []
        if load_cases is not None:
            self.extend(load_cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self):
        return iter(self._cases)

    def __getitem__(self, index):
        return self._cases[index]

    def __repr__(self) -> str:
        return f"LoadCases(n={len(self)}, beam_model={type(self.beam_model).__name__})"

    @property
    def cases(self) -> tuple[LoadCase, ...]:
        """Load cases as an immutable tuple."""

        return tuple(self._cases)

    @property
    def names(self) -> tuple[str, ...]:
        """Load-case names in collection order."""

        return tuple(case.name for case in self._cases)

    def load_positions(self) -> list[int]:
        """
        Return integer identifiers for load-position tracking.

        Cases generated by segmented-load helpers store a ``load_position`` in
        their metadata. Other cases default to one bit per case in collection
        order.
        """

        positions = []
        for i, case in enumerate(self._cases):
            metadata = case.metadata or {}
            positions.append(int(metadata.get("load_position", 1 << i)))
        return positions

    def to_LM(self) -> dict[str, LoadMatrix]:
        """
        Return the load matrices keyed by load-case name.

        The dictionary preserves the collection order and each load matrix is
        copied so callers can inspect or mutate it without changing the
        stored load cases.
        """

        return {case.name: case.to_LM() for case in self._cases}

    def case(self, name: str, create: bool = True) -> LoadCase:
        """Return a named load case, optionally creating it when missing."""

        for case in self._cases:
            if case.name == name:
                return case
        if not create:
            raise KeyError(f"Unknown load case name: {name}")
        return self.add_case(name)

    def add_case(
        self,
        name: str,
        loads: Optional[LoadMatrix] = None,
        loaded_spans: tuple[int, ...] = (),
        metadata: Optional[dict] = None,
    ) -> LoadCase:
        """Add and return a named load case."""

        return self.add(
            name=name,
            loads=[] if loads is None else loads,
            loaded_spans=loaded_spans,
            metadata=metadata,
        )

    def add(
        self,
        name: str,
        loads: LoadMatrix,
        loaded_spans: tuple[int, ...] = (),
        metadata: Optional[dict] = None,
    ) -> LoadCase:
        """
        Add a named load case to the collection.

        ``loads`` is a normal PyCBA load matrix. It is copied on entry so later
        caller-side mutation does not alter the stored case.
        """

        if name in self.names:
            raise ValueError(f"Load case already exists: {name}")
        case = LoadCase(
            name=name,
            loads=deepcopy(loads),
            loaded_spans=loaded_spans,
            metadata=deepcopy(metadata) if metadata is not None else None,
        )
        self._cases.append(case)
        return case

    def set(
        self,
        name: str,
        loads: LoadMatrix,
        loaded_spans: tuple[int, ...] = (),
        metadata: Optional[dict] = None,
    ) -> LoadCase:
        """Replace or create a named load case."""

        try:
            case = self.case(name, create=False)
        except KeyError:
            return self.add(name, loads, loaded_spans=loaded_spans, metadata=metadata)

        case.loads = deepcopy(loads)
        case.loaded_spans = loaded_spans
        case.metadata = deepcopy(metadata) if metadata is not None else None
        return case

    def append(self, load_case: LoadCase) -> LoadCase:
        """Append an existing ``LoadCase`` or load-case-like object."""

        name = _load_case_name(load_case, len(self._cases))
        if name in self.names:
            raise ValueError(f"Load case already exists: {name}")
        case = LoadCase(
            name=name,
            loads=_load_case_loads(load_case),
            loaded_spans=_load_case_loaded_spans(load_case),
            metadata=deepcopy(_get_value(load_case, "metadata", None)),
        )
        self._cases.append(case)
        return case

    def extend(self, load_cases: Sequence[LoadCase]) -> None:
        """Append several load cases."""

        for load_case in load_cases:
            self.append(load_case)

    def add_span_udl(self, w: float) -> list[LoadCase]:
        """Add one full-span UDL case per span."""

        added = []
        for i, _ in enumerate(_beam_spans(self.beam_model)):
            added.append(
                self.add(
                    name=f"UDL on span {i + 1}",
                    loads=[[i + 1, 1, w]],
                    loaded_spans=(i,),
                    metadata={"type": "span_udl", "span": i, "w": w},
                )
            )
        return added

    def add_load(self, load_case: str, load: list) -> LoadCase:
        """Append one raw load row to a named load case."""

        return self.case(load_case).add_load(load)

    def add_udl(self, load_case: str, i_span: int, w: float) -> LoadCase:
        """Append a full-span UDL to a named load case."""

        return self.case(load_case).add_udl(i_span, w)

    def add_pl(self, load_case: str, i_span: int, p: float, a: float) -> LoadCase:
        """Append a point load to a named load case."""

        return self.case(load_case).add_pl(i_span, p, a)

    def add_pudl(
        self, load_case: str, i_span: int, w: float, a: float, c: float
    ) -> LoadCase:
        """Append a partial UDL to a named load case."""

        return self.case(load_case).add_pudl(i_span, w, a, c)

    def add_segment_udl(
        self, load_case: str, x0: float, x1: float, w: float
    ) -> LoadCase:
        """
        Append a UDL covering a global beam segment to a named load case.

        The segment is split at span boundaries into full-span UDL and partial
        UDL rows as required.
        """

        return self.case(load_case).add_segment_udl(self.beam_model, x0, x1, w)

    def add_all_spans_udl(self, load_case: str, w: float) -> LoadCase:
        """
        Append a full-span UDL of intensity ``w`` to every span of a named case.

        The beam is supplied implicitly from this collection (consistent with
        :meth:`add_segment_udl`). The named case is created when it does not yet
        exist.

        Parameters
        ----------
        load_case : str
            Name of the load case to add the UDLs to.
        w : float
            UDL intensity applied to every span.

        Returns
        -------
        LoadCase
            The (created or existing) load case with the UDLs appended.

        Notes
        -----
        This adds all spans into the one named case, which is distinct from
        :meth:`add_span_udl` (one separate case per span).
        """

        return self.case(load_case).add_all_spans_udl(self.beam_model, w)

    def add_ml(self, load_case: str, i_span: int, m: float, a: float) -> LoadCase:
        """Append a concentrated moment load to a named load case."""

        return self.case(load_case).add_ml(i_span, m, a)

    def add_trap(
        self,
        load_case: str,
        i_span: int,
        w1: float,
        w2: float,
        a: Optional[float] = None,
        c: Optional[float] = None,
    ) -> LoadCase:
        """Append a full-span or partial trapezoidal load to a named load case."""

        return self.case(load_case).add_trap(i_span, w1, w2, a=a, c=c)

    def add_ic(self, load_case: str, i_span: int, kappa) -> LoadCase:
        """Append an imposed-curvature load to a named load case."""

        return self.case(load_case).add_ic(i_span, kappa)

    def response_matrix(self, response: str = "M") -> tuple[np.ndarray, np.ndarray]:
        """Analyse all cases and return the common station vector and matrix."""

        return collect_response_matrix(self.beam_model, self._cases, response=response)

    def envelope(self, npts: Optional[int] = None) -> Envelopes:
        """
        Analyse every load case independently and return their envelope.

        Each case is analysed on its own (no combination) and the per-station
        pointwise extremes of moment and shear (plus coincident effects and
        reaction extremes) are enveloped into a first-class
        :class:`pycba.results.Envelopes`. This gives a one-call path to a full,
        plottable envelope from helpers such as :func:`make_span_udl_cases` and
        :func:`make_patterned_udl`.

        Parameters
        ----------
        npts : int, optional
            Number of evaluation points along a member. Defaults to the template
            beam's ``npts`` when available.

        Returns
        -------
        Envelopes
            The pointwise envelope over the analysed load cases.

        Notes
        -----
        This is the pointwise extreme of each case analysed alone, which is a
        genuinely different result from :func:`additive_envelope` (a same-station
        superposition of the ``n_combine`` governing cases).
        """

        results = []
        for load_case in self._cases:
            model = build_pycba_model(self.beam_model, load_case)
            model.analyze(
                npts if npts is not None else _template_npts(self.beam_model)
            )
            results.append(model.beam_results)
        return Envelopes(results)

    def factors(
        self,
        factors: Union[
            LoadCombination, Sequence[float], Mapping[Union[str, int], float]
        ],
    ):
        """
        Convert sequence or mapping factors to a numeric vector.

        A sequence must have one value per load case. A mapping can use either
        load-case names or zero-based integer indices as keys; unspecified
        cases receive a zero factor.
        """

        if isinstance(factors, LoadCombination):
            factors = factors.factors

        if isinstance(factors, Mapping):
            out = np.zeros(len(self._cases), dtype=float)
            name_to_index = {case.name: i for i, case in enumerate(self._cases)}
            for key, value in factors.items():
                if isinstance(key, str):
                    if key not in name_to_index:
                        raise KeyError(f"Unknown load case name: {key}")
                    idx = name_to_index[key]
                else:
                    idx = int(key)
                out[idx] = value
            return out

        out = np.asarray(factors, dtype=float)
        if out.ndim == 1 and out.shape[0] != len(self._cases):
            raise ValueError(f"Need {len(self._cases)} factors, got {out.shape[0]}.")
        if out.ndim == 2 and out.shape[1] != len(self._cases):
            raise ValueError(
                f"Need factor rows with {len(self._cases)} entries, got {out.shape[1]}."
            )
        if out.ndim not in (1, 2):
            raise ValueError("factors must be a 1D or 2D array, or a mapping.")
        return out

    def target_combination(
        self,
        name: str,
        x: float,
        sense: str = "min",
        response: str = "M",
        selected_factor: float = 1.0,
        unselected_factor: float = 0.0,
    ) -> LoadCombination:
        """
        Select a linear combination by response sign at a target coordinate.

        For ``sense="min"``, cases with a negative effect at ``x`` receive
        ``selected_factor``. For ``sense="max"``, cases with a positive effect
        receive ``selected_factor``. Other cases receive
        ``unselected_factor``.
        """

        return _make_target_combination(
            self,
            name=name,
            x=x,
            sense=sense,
            response=response,
            selected_factor=selected_factor,
            unselected_factor=unselected_factor,
        )

    def combine(
        self,
        factors: Union[
            LoadCombination, Sequence[float], Mapping[Union[str, int], float]
        ],
        response: str = "M",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a linear response combination using arbitrary case factors.

        A 1D factor vector returns one response vector. A 2D factor matrix
        returns one response row per factor row.
        """

        x, B = self.response_matrix(response=response)
        f = self.factors(factors)
        return x, f @ B

    def combined_loads(
        self,
        factors: Union[
            LoadCombination, Sequence[float], Mapping[Union[str, int], float]
        ],
    ) -> LoadMatrix:
        """
        Return one load matrix formed from factored named load cases.

        ``factors`` must define a single combination, either as a 1D sequence
        with one factor per case, or as a mapping from case names or indices to
        factors. Cases omitted from a mapping receive a zero factor.
        """

        f = self.factors(factors)
        if f.ndim != 1:
            raise ValueError("combined_loads requires one factor set.")

        loads = []
        for factor, load_case in zip(f, self._cases):
            if factor == 0.0:
                continue
            loads.extend(factor_LM(load_case.loads, float(factor)))
        return loads

    def analyze_combination(
        self,
        factors: Union[
            LoadCombination, Sequence[float], Mapping[Union[str, int], float]
        ],
        npts: Optional[int] = None,
    ) -> BeamAnalysis:
        """
        Analyse one factored load-case combination and return the analysis.

        This is useful when a combined case should be treated like an ordinary
        :class:`BeamAnalysis`, for example to use
        :meth:`BeamAnalysis.plot_results`.
        """

        load_case = LoadCase(name="combined", loads=self.combined_loads(factors))
        model = build_pycba_model(self.beam_model, load_case)
        model.analyze(npts if npts is not None else _template_npts(self.beam_model))
        return model

    analyse_combination = analyze_combination

    def linear_combination(
        self,
        factors: Union[
            LoadCombination, Sequence[float], Mapping[Union[str, int], float]
        ],
        response: str = "M",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alias for :meth:`combine`."""

        return self.combine(factors, response=response)


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    """Return ``name`` from either a dataclass-like object or a dictionary."""

    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _template_beam(beam_model: Union[BeamAnalysis, Beam]) -> Beam:
    if isinstance(beam_model, BeamAnalysis):
        return beam_model.beam
    if isinstance(beam_model, Beam):
        return beam_model
    raise TypeError("beam_model must be a pycba.BeamAnalysis or pycba.Beam object.")


def _template_npts(beam_model: Union[BeamAnalysis, Beam]) -> Optional[int]:
    if isinstance(beam_model, BeamAnalysis):
        return beam_model.npts
    return None


def _beam_spans(beam_model: Union[BeamAnalysis, Beam]) -> list:
    return list(_template_beam(beam_model).mbr_lengths)


def _load_case_name(load_case: LoadCase, i: Optional[int] = None) -> str:
    default = f"LC {i + 1}" if i is not None else "LC"
    if isinstance(load_case, list):
        return default
    return _get_value(load_case, "name", default)


def _load_case_loads(load_case: LoadCase) -> list:
    if isinstance(load_case, list):
        return deepcopy(load_case)
    loads = _get_value(load_case, "loads", [])
    return deepcopy(loads)


def _load_case_loaded_spans(load_case: LoadCase) -> tuple[int, ...]:
    if isinstance(load_case, list):
        loads = load_case
    else:
        loads = _get_value(load_case, "loads", [])
    spans = _get_value(load_case, "loaded_spans", ())
    if len(spans) > 0:
        return tuple(int(i) for i in spans)

    loaded = []
    for load in loads:
        if len(load) < 1:
            continue
        loaded.append(int(load[0]) - 1)
    return tuple(sorted(set(loaded)))


def _span_boundaries(beam_model: Union[BeamAnalysis, Beam]) -> np.ndarray:
    return np.cumsum([0.0] + _beam_spans(beam_model))


def _load_case_loaded_regions(
    beam_model: Union[BeamAnalysis, Beam], load_case: LoadCase
) -> list[tuple[float, float]]:
    boundaries = _span_boundaries(beam_model)
    spans = np.diff(boundaries)
    regions = []

    for load in _load_case_loads(load_case):
        if len(load) < 2:
            continue
        span_idx = int(load[0]) - 1
        if span_idx < 0 or span_idx >= len(spans):
            continue

        load_type = int(load[1])
        span_length = spans[span_idx]
        if load_type == 1 or (load_type == 5 and len(load) <= 4):
            a = 0.0
            c = span_length
        elif load_type == 3 and len(load) >= 5:
            a = float(load[3])
            c = float(load[4])
        elif load_type == 5 and len(load) >= 6:
            a = float(load[4])
            c = float(load[5])
        else:
            continue

        a = max(0.0, min(a, span_length))
        c = max(0.0, min(c, span_length - a))
        if c > 0.0:
            regions.append((boundaries[span_idx] + a, c))

    if regions:
        return regions

    return [
        (boundaries[i], spans[i])
        for i in _load_case_loaded_spans(load_case)
        if 0 <= i < len(spans)
    ]


def _factor_load_matrix(loads: LoadMatrix, factor: float) -> list:
    return factor_LM(loads, factor)


def _choose_max_by_span(wmin: LoadMatrix, wmax: LoadMatrix, spans) -> list:
    max_spans = {int(span) for span in spans}
    return [
        deepcopy(max_load if int(max_load[0]) in max_spans else min_load)
        for min_load, max_load in zip(wmin, wmax)
    ]


def build_pycba_model(
    beam_model: Union[BeamAnalysis, Beam], load_case: LoadCase
) -> BeamAnalysis:
    """
    Build a fresh :class:`pycba.analysis.BeamAnalysis` for one load case.

    ``beam_model`` is an existing PyCBA beam definition, normally a
    :class:`BeamAnalysis` with the required spans, stiffnesses, supports,
    springs, releases, and prescribed displacements already defined. The
    returned model is a new analysis object so independent load cases do not
    mutate each other.
    """

    beam = _template_beam(beam_model)
    return BeamAnalysis(
        L=list(beam.mbr_lengths),
        EI=list(beam.mbr_EIs),
        R=list(beam.restraints),
        LM=_load_case_loads(load_case),
        eletype=list(beam.mbr_eletype),
        D=list(beam.prescribed_displacements),
    )


def analyse_load_case(
    beam_model: Union[BeamAnalysis, Beam], load_case: LoadCase, response: str = "M"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analyse one independent load case on the full beam.

    Parameters
    ----------
    beam_model : BeamAnalysis or Beam
        Existing arbitrary PyCBA beam definition used as the analysis template.
    load_case : LoadCase, dict, or LoadMatrix
        Independent load case to analyse over the full beam.
    response : str, optional
        Station response to return. Supported values are ``"M"`` (moment),
        ``"V"`` (shear), ``"D"`` or ``"deflection"``, and ``"R"`` or
        ``"rotation"``.

    Returns
    -------
    x, y : ndarray
        Global station coordinates and the requested response at those stations.
    """

    attr = {
        "M": "M",
        "MOMENT": "M",
        "V": "V",
        "SHEAR": "V",
        "D": "D",
        "DEFLECTION": "D",
        "DISPLACEMENT": "D",
        "R": "R",
        "ROTATION": "R",
    }.get(response.upper())
    if attr is None:
        raise ValueError(
            "response must be one of 'M', 'V', 'D'/'deflection', or "
            "'R'/'rotation'."
        )

    model = build_pycba_model(beam_model, load_case)
    model.analyze(_template_npts(beam_model))
    results = model.beam_results.results
    return np.asarray(results.x), np.asarray(getattr(results, attr))


analyze_load_case = analyse_load_case


def collect_response_matrix(
    beam_model: Union[BeamAnalysis, Beam],
    load_cases: Sequence[LoadCase],
    response: str = "M",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analyse all load cases and stack their full-beam responses into a matrix.

    Returns
    -------
    x : ndarray
        Common global station coordinates.
    B : ndarray
        Response matrix with shape ``(n_load_cases, n_stations)``.
    """

    x_ref = None
    rows = []

    for load_case in load_cases:
        x, y = analyse_load_case(beam_model, load_case, response=response)

        if x_ref is None:
            x_ref = x
        elif not np.allclose(x, x_ref):
            raise ValueError("Station coordinates differ between load cases.")

        rows.append(y)

    if x_ref is None:
        raise ValueError("At least one load case is required.")

    return x_ref, np.vstack(rows)


def additive_envelope(
    B: np.ndarray, n_combine: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute additive positive and negative pointwise envelopes.

    The negative envelope sums the ``n_combine`` smallest values at each
    station. The positive envelope sums the ``n_combine`` largest values at
    each station. Governing load-case indices are returned for both envelopes.
    """

    B = np.asarray(B)
    if B.ndim != 2:
        raise ValueError("B must be a 2D array with shape (n_load_cases, n_stations).")
    if n_combine < 1:
        raise ValueError("n_combine must be at least 1.")

    n_load_cases, _ = B.shape
    if n_load_cases < n_combine:
        raise ValueError(
            f"Need at least {n_combine} load cases, got {n_load_cases}."
        )

    order = np.argsort(B, axis=0)
    idx_neg = order[:n_combine, :]
    idx_pos = order[-n_combine:, :]

    env_neg = np.take_along_axis(B, idx_neg, axis=0).sum(axis=0)
    env_pos = np.take_along_axis(B, idx_pos, axis=0).sum(axis=0)

    return env_neg, env_pos, idx_neg, idx_pos


def sign_selective_envelope(
    B: np.ndarray, load_positions: Optional[Sequence[int]] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sum same-sign load-case contributions at each station.

    The negative envelope is the sum of all negative contributors at each
    station. The positive envelope is the sum of all positive contributors.
    The returned position masks identify which load cases contributed to each
    station envelope.
    """

    B = np.asarray(B)
    if B.ndim != 2:
        raise ValueError("B must be a 2D array with shape (n_load_cases, n_stations).")

    n_load_cases, n_stations = B.shape
    if load_positions is None:
        positions = [1 << i for i in range(n_load_cases)]
    else:
        if len(load_positions) != n_load_cases:
            raise ValueError(
                f"Need {n_load_cases} load positions, got {len(load_positions)}."
            )
        positions = [int(position) for position in load_positions]

    env_neg = np.where(B < 0.0, B, 0.0).sum(axis=0)
    env_pos = np.where(B > 0.0, B, 0.0).sum(axis=0)

    mask_neg = np.zeros(n_stations, dtype=object)
    mask_pos = np.zeros(n_stations, dtype=object)
    for k in range(n_stations):
        neg = 0
        pos = 0
        for j, position in enumerate(positions):
            if B[j, k] < 0.0:
                neg |= position
            elif B[j, k] > 0.0:
                pos |= position
        mask_neg[k] = neg
        mask_pos[k] = pos

    return env_neg, env_pos, mask_neg, mask_pos


def _response_at(x: np.ndarray, B: np.ndarray, x_target: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    B = np.asarray(B, dtype=float)
    x_target = float(x_target)

    if x.ndim != 1:
        raise ValueError("x must be a 1D station array.")
    if B.ndim != 2 or B.shape[1] != x.shape[0]:
        raise ValueError("B must have shape (n_load_cases, len(x)).")
    if x_target < x.min() or x_target > x.max():
        raise ValueError(
            f"Target coordinate {x_target} lies outside the analysis range "
            f"[{x.min()}, {x.max()}]."
        )

    unique_x, unique_index = np.unique(x, return_index=True)
    unique_B = B[:, unique_index]
    return np.array([np.interp(x_target, unique_x, row) for row in unique_B])


def _make_target_combination(
    load_cases: LoadCases,
    name: str,
    x: float,
    sense: str = "min",
    response: str = "M",
    selected_factor: float = 1.0,
    unselected_factor: float = 0.0,
) -> LoadCombination:
    """
    Select a linear combination by response sign at a target coordinate.

    This is useful when a set of basis load cases represents independently
    placeable load fragments. The returned combination selects the fragments
    that drive one target load effect in the requested direction.
    """

    sense_key = sense.lower()
    if sense_key in {"min", "minimum", "neg", "negative", "hogging"}:
        select_minimum = True
    elif sense_key in {"max", "maximum", "pos", "positive", "sagging"}:
        select_minimum = False
    else:
        raise ValueError("sense must be 'min'/'negative' or 'max'/'positive'.")

    x_ref, B = load_cases.response_matrix(response=response)
    target_values = _response_at(x_ref, B, x)
    if select_minimum:
        selected = target_values < 0.0
    else:
        selected = target_values > 0.0

    factors = np.full(len(load_cases), float(unselected_factor), dtype=float)
    factors[selected] = float(selected_factor)

    selected_indices = np.flatnonzero(selected)
    metadata = {
        "type": "target_combination",
        "x": float(x),
        "response": response,
        "sense": sense,
        "selected_indices": selected_indices.tolist(),
        "selected_names": [load_cases[i].name for i in selected_indices],
        "target_values": target_values.tolist(),
    }
    combination = LoadCombination(name=name, factors=factors, metadata=metadata)
    # Bind the originating basis as a convenience back-reference so the analysis
    # methods can be called with no arguments. Set as a plain attribute (not a
    # dataclass field) to leave __init__/__eq__/__repr__ untouched.
    combination._bound = load_cases
    return combination


def _make_segment_udl(
    beam_model: Union[BeamAnalysis, Beam], x0: float, x1: float, w: float
) -> LoadMatrix:
    """
    Create load-matrix rows for a UDL over a global beam segment.

    The input coordinates ``x0`` and ``x1`` are measured from the left end of
    the whole beam. The returned rows are ordinary PyCBA load-matrix rows split
    at span boundaries, using full-span UDL rows where possible and partial UDL
    rows otherwise.
    """

    boundaries = _span_boundaries(beam_model)
    total_length = boundaries[-1]
    tol = 1e-9

    if x1 <= x0:
        raise ValueError("x1 must be greater than x0 for a segmented UDL.")
    if x0 < -tol or x1 > total_length + tol:
        raise ValueError(
            f"Segment [{x0}, {x1}] must lie within the beam length "
            f"[0, {total_length}]."
        )

    x0 = max(0.0, min(float(x0), float(total_length)))
    x1 = max(0.0, min(float(x1), float(total_length)))

    loads = []
    for i, (span_start, span_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        overlap_start = max(x0, span_start)
        overlap_end = min(x1, span_end)
        cover = overlap_end - overlap_start
        if cover <= tol:
            continue

        span_length = span_end - span_start
        a = overlap_start - span_start
        if np.isclose(a, 0.0, atol=tol):
            a = 0.0
        if np.isclose(cover, span_length, atol=tol):
            cover = span_length

        if a == 0.0 and cover == span_length:
            loads.append([i + 1, 1, w])
        else:
            loads.append([i + 1, 3, w, a, cover])

    return loads


def _validate_n_segments(n_segments: int) -> int:
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)):
        raise ValueError("n_segments must be a positive integer.")
    n_segments = int(n_segments)
    if n_segments < 1:
        raise ValueError("n_segments must be a positive integer.")
    return n_segments


def make_patterned_udl(
    beam_model: Union[BeamAnalysis, Beam], w: float, n_segments: int = 20
) -> LoadCases:
    """
    Create partial-UDL basis load cases for load patterning.

    Each span is divided into ``n_segments`` equal local segments, and each
    segment is stored as one ordinary :class:`LoadCase`. Use
    :meth:`LoadCases.target_combination` on the returned collection to create a
    :class:`LoadCombination` for a target effect such as support hogging.
    """

    n_segments = _validate_n_segments(n_segments)

    load_cases = LoadCases(beam_model)
    boundaries = _span_boundaries(beam_model)
    spans = np.diff(boundaries)
    case_index = 0

    for i_span, span_length in enumerate(spans, start=1):
        segment_length = span_length / n_segments
        for i_segment in range(n_segments):
            a = i_segment * segment_length
            c = span_length - a if i_segment == n_segments - 1 else segment_length
            x0 = boundaries[i_span - 1] + a
            x1 = x0 + c
            load_cases.add(
                name=f"UDL span {i_span} segment {i_segment + 1}",
                loads=[[i_span, 3, w, float(a), float(c)]],
                loaded_spans=(i_span - 1,),
                metadata={
                    "type": "segmented_udl",
                    "span": i_span - 1,
                    "segment": i_segment,
                    "n_segments": n_segments,
                    "w": w,
                    "x0": float(x0),
                    "x1": float(x1),
                    "load_position": 1 << case_index,
                },
            )
            case_index += 1

    return load_cases


def make_span_udl_cases(beam_model: Union[BeamAnalysis, Beam], w: float) -> LoadCases:
    """
    Create one independent full-span UDL load case per beam span.

    The returned load matrix rows use PyCBA's load convention:
    ``[span, 1, w]`` with 1-based span numbers and positive ``w`` acting
    downward. The returned object is a :class:`LoadCases` collection.
    """

    load_cases = LoadCases(beam_model)
    load_cases.add_span_udl(w)
    return load_cases


def plot_response_envelope(
    x: np.ndarray,
    B: np.ndarray,
    env_neg: np.ndarray,
    env_pos: np.ndarray,
    beam_model: Optional[Union[BeamAnalysis, Beam]] = None,
    load_cases: Optional[Sequence[LoadCase]] = None,
    show: bool = True,
):
    """
    Plot individual full-beam load-case responses and additive envelopes.

    Returns the ``(fig, ax)`` pair so callers can further customise or test the
    plot. ``show=False`` suppresses ``plt.show()``.
    """

    fig, ax = plt.subplots()

    for j in range(B.shape[0]):
        label = (
            _load_case_name(load_cases[j], j)
            if load_cases is not None
            else f"LC {j + 1}"
        )
        ax.plot(x, B[j, :], linewidth=1.0, alpha=0.7, label=label)

    ax.plot(x, env_neg, "k--", linewidth=2.0, label="Additive negative envelope")
    ax.plot(x, env_pos, "k:", linewidth=2.0, label="Additive positive envelope")
    ax.axhline(0.0, color="0.25", linewidth=0.8)

    if beam_model is not None:
        boundaries = _span_boundaries(beam_model)
        for xb in boundaries:
            ax.axvline(xb, color="0.5", linewidth=0.6, alpha=0.5)
        for i, (x0, x1) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            ax.text(
                0.5 * (x0 + x1),
                0.98,
                f"S{i + 1}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize="small",
                color="0.35",
            )

    ax.set_xlabel("Position along beam")
    ax.set_ylabel("Response")
    ax.grid(True)
    ax.legend()

    if show:
        plt.show()
    return fig, ax


def plot_load_patterns(
    beam_model: Union[BeamAnalysis, Beam],
    load_cases: Sequence[LoadCase],
    show: bool = True,
):
    """
    Plot arbitrary span UDL-style load patterns, one row per load case.

    For generic load cases, loaded spans are taken from ``loaded_spans`` when
    supplied, otherwise they are inferred from the span number in each load
    matrix row.
    """

    load_cases = list(load_cases)
    boundaries = _span_boundaries(beam_model)
    spans = np.diff(boundaries)

    fig_height = max(2.5, 0.45 * len(load_cases) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))

    for j, load_case in enumerate(load_cases):
        y0 = j - 0.3
        ax.broken_barh(
            [(boundaries[i], spans[i]) for i in range(len(spans))],
            (y0, 0.6),
            facecolors="0.92",
            edgecolors="0.75",
            linewidth=0.8,
        )

        bars = _load_case_loaded_regions(beam_model, load_case)
        if bars:
            ax.broken_barh(
                bars,
                (y0, 0.6),
                facecolors="C0",
                edgecolors="C0",
                alpha=0.65,
            )

    for xb in boundaries:
        ax.axvline(xb, color="0.5", linewidth=0.6, alpha=0.5)

    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.set_ylim(-0.7, len(load_cases) - 0.3)
    ax.set_xlabel("Position along beam")
    ax.set_yticks(range(len(load_cases)))
    ax.set_yticklabels([_load_case_name(lc, i) for i, lc in enumerate(load_cases)])
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    if show:
        plt.show()
    return fig, ax
