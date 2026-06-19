"""
PyCBA - Continuous Beam Analysis - Load Patterning Module
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

    def add_udl(self, i_span: int, w: float) -> LoadCase:
        """Append a full-span UDL to this load case."""

        return self.add_load([i_span, 1, w])

    def add_pl(self, i_span: int, p: float, a: float) -> LoadCase:
        """Append a point load to this load case."""

        return self.add_load([i_span, 2, p, a])

    def add_pudl(self, i_span: int, w: float, a: float, c: float) -> LoadCase:
        """Append a partial UDL to this load case."""

        return self.add_load([i_span, 3, w, a, c])

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

    def factors(self, factors: Union[Sequence[float], Mapping[Union[str, int], float]]):
        """
        Convert sequence or mapping factors to a numeric vector.

        A sequence must have one value per load case. A mapping can use either
        load-case names or zero-based integer indices as keys; unspecified
        cases receive a zero factor.
        """

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

    def combine(
        self,
        factors: Union[Sequence[float], Mapping[Union[str, int], float]],
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
        self, factors: Union[Sequence[float], Mapping[Union[str, int], float]]
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
            loads.extend(factor_LM(load_case.loads, factor))
        return loads

    def analyze_combination(
        self,
        factors: Union[Sequence[float], Mapping[Union[str, int], float]],
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
        factors: Union[Sequence[float], Mapping[Union[str, int], float]],
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

    explicit_spans = _get_value(load_case, "loaded_spans", ())
    if len(explicit_spans) > 0:
        return [
            (boundaries[i], spans[i])
            for i in _load_case_loaded_spans(load_case)
            if 0 <= i < len(spans)
        ]

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


class LoadPattern:
    """
    Automatically patterns dead and live loads to achieve critical load effects.
    """

    def __init__(self, ba: BeamAnalysis):
        """
        Initialize the LoadPattern class with the beam.

        Parameters
        ----------
        ba : BeamAnalysis, optional
            A :class:`pycba.analysis.BeamAnalysis` object. The default is None.

        Returns
        -------
        None.
        """

        self.ba = ba
        self.LMg = None
        self.LMq = None
        self.gamma_g_min = 0
        self.gamma_g_max = 0
        self.gamma_q_min = 0
        self.gamma_q_max = 0

    def set_dead_loads(
        self, LM: Union[LoadMatrix, LoadCase], gamma_max: float, gamma_min: float
    ):
        """
        Set the nominal dead loads acting on the beam, and the maximum and
        minimum load factor.

        Parameters
        ----------
        LM : LoadMatrix or LoadCase
            The load matrix, or a :class:`LoadCase` whose loads provide the
            matrix, for this load case.
        gamma_max : float
            The maximum load factor.
        gamma_min : float
            The minimum load factor.

        Returns
        -------
        None.
        """

        self.LMg = _load_case_loads(LM)
        self.gamma_g_max = gamma_max
        self.gamma_g_min = gamma_min

    def set_live_loads(
        self, LM: Union[LoadMatrix, LoadCase], gamma_max: float, gamma_min: float
    ):
        """
        Set the nominal live loads acting on the beam, and the maximum and
        minimum load factor.

        Parameters
        ----------
        LM : LoadMatrix or LoadCase
            The load matrix, or a :class:`LoadCase` whose loads provide the
            matrix, for this load case.
        gamma_max : float
            The maximum load factor.
        gamma_min : float
            The minimum load factor.

        Returns
        -------
        None.
        """

        self.LMq = _load_case_loads(LM)
        self.gamma_q_max = gamma_max
        self.gamma_q_min = gamma_min

    def analyze(self, npts: Optional[int] = None) -> Envelopes:
        """
        Conduct the load patterning analysis.

        Parameters
        ----------
        npts : Optional[int]
            The number of evaluation points along a member for load effects.

        Returns
        -------
        Envelopes : `pycba.Envelopes`
            The load effect envelopes from the patterning.
        """

        results = []
        for load_case in self.to_load_cases():
            model = build_pycba_model(self.ba, load_case)
            model.analyze(npts if npts is not None else self.ba.npts)
            self.ba.set_loads(_load_case_loads(load_case))
            self.ba.npts = model.npts
            self.ba._beam_results = model.beam_results
            results.append(model.beam_results)

        return Envelopes(results)

    def to_load_cases(self) -> LoadCases:
        """
        Generate the factored load arrangements used by :meth:`analyze`.

        Returns
        -------
        LoadCases
            A collection containing one generated factored load case for each
            design load pattern.
        """

        if self.LMg is None or self.LMq is None:
            raise ValueError("Dead and live loads must be set before patterning.")

        wmax = (
            _factor_load_matrix(self.LMg, self.gamma_g_max)
            + _factor_load_matrix(self.LMq, self.gamma_q_max)
        )
        wmin = (
            _factor_load_matrix(self.LMg, self.gamma_g_min)
            + _factor_load_matrix(self.LMq, self.gamma_q_min)
        )
        N = self.ba.beam.no_spans
        load_cases = LoadCases(self.ba)

        # Maximum support hogging and reaction -
        # adjacent spans fully loaded with MAX, other spans loaded with MIN
        n_max_hog = max(N - 1, 0)
        for i in range(1, n_max_hog + 1):
            adjacent_spans = (i, i + 1)
            load_cases.add(
                name=f"Max hogging spans {i}-{i + 1}",
                loads=_choose_max_by_span(wmin, wmax, adjacent_spans),
                loaded_spans=tuple(span - 1 for span in adjacent_spans),
                metadata={"type": "load_pattern", "max_spans": adjacent_spans},
            )

        # Odd numbered spans loaded with MAX for maximum sagging moments
        odd_spans = tuple(i for i in range(1, N + 1, 2))
        load_cases.add(
            name="Max odd spans",
            loads=_choose_max_by_span(wmin, wmax, odd_spans),
            loaded_spans=tuple(span - 1 for span in odd_spans),
            metadata={"type": "load_pattern", "max_spans": odd_spans},
        )

        # Even numbered spans loaded with MAX for maximum sagging moments
        even_spans = tuple(i for i in range(2, N + 1, 2))
        load_cases.add(
            name="Max even spans",
            loads=_choose_max_by_span(wmin, wmax, even_spans),
            loaded_spans=tuple(span - 1 for span in even_spans),
            metadata={"type": "load_pattern", "max_spans": even_spans},
        )

        # All spans loaded with MAX
        all_spans = tuple(range(1, N + 1))
        load_cases.add(
            name="All spans max",
            loads=wmax,
            loaded_spans=tuple(span - 1 for span in all_spans),
            metadata={"type": "load_pattern", "max_spans": all_spans},
        )

        return load_cases
