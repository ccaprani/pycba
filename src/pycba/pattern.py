"""
PyCBA - Continuous Beam Analysis - Design Load Patterning
"""

from __future__ import annotations  # https://bit.ly/3KYiL2o
from typing import Optional, Union
from .analysis import BeamAnalysis
from .results import Envelopes
from .load import LoadMatrix
from .load_cases import (
    LoadCase,
    LoadCases,
    _choose_max_by_span,
    _factor_load_matrix,
    _load_case_loads,
    build_pycba_model,
)


__all__ = ["LoadPattern"]


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

        wmax = _factor_load_matrix(self.LMg, self.gamma_g_max) + _factor_load_matrix(
            self.LMq, self.gamma_q_max
        )
        wmin = _factor_load_matrix(self.LMg, self.gamma_g_min) + _factor_load_matrix(
            self.LMq, self.gamma_q_min
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

    def to_LM(self) -> dict[str, LoadMatrix]:
        """
        Generate the factored load matrices used by :meth:`analyze`.

        This is a debugging convenience for users who want to inspect the
        actual PyCBA load matrix for each generated load pattern without
        working through the intermediate :class:`LoadCases` object.

        Returns
        -------
        dict[str, LoadMatrix]
            Mapping from generated load-pattern name to its factored load
            matrix. The dictionary preserves the analysis order.
        """

        return self.to_load_cases().to_LM()
