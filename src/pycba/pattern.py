"""
PyCBA - Continuous Beam Analysis - Load Patterning Module
"""

from __future__ import annotations  # https://bit.ly/3KYiL2o
from typing import Optional, Union, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from .analysis import BeamAnalysis
from .results import Envelopes, BeamResults
from .load import LoadMatrix


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

    def set_dead_loads(self, LM: LoadMatrix, gamma_max: float, gamma_min: float):
        """
        Set the nominal dead loads acting on the beam, and the maximum and
        minimum load factor.

        Parameters
        ----------
        LM : List[List[Union[int, float]]]
            The load matrix for the beam, for this loadcase.
        gamma_max : float
            The maximum load factor.
        gamma_min : float
            The minimum load factor.

        Returns
        -------
        None.
        """

        self.LMg = LM
        self.gamma_g_max = gamma_max
        self.gamma_g_min = gamma_min

    def set_live_loads(self, LM: LoadMatrix, gamma_max: float, gamma_min: float):
        """
        Set the nominal live loads acting on the beam, and the maximum and
        minimum load factor.

        Parameters
        ----------
        LM : List[List[Union[int, float]]]
            The load matrix for the beam, for this loadcase.
        gamma_max : float
            The maximum load factor.
        gamma_min : float
            The minimum load factor.

        Returns
        -------
        None.
        """

        self.LMq = LM
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

        # Helper function to get the BeamResults object easily
        def analyze_loadcase(w):
            self.ba.set_loads(w.tolist())
            self.ba.analyze(npts)
            return self.ba.beam_results

        # Basic copies of the load matrices
        wg = np.array(self.LMg)
        wq = np.array(self.LMq)
        gr, gc = wg.shape
        qr, qc = wq.shape
        w = np.zeros((gr + qr, max(gc, qc)))
        wmax = w.copy()
        wmin = w.copy()

        # Maximum load matrix
        wmax[:gr, :] = wg
        wmax[:gr, 2] = self.gamma_g_max * wmax[:gr, 2]
        wmax[gr : gr + qr, :] = wq
        wmax[gr : gr + qr, 2] = self.gamma_q_max * wmax[gr : gr + qr, 2]

        # Minimum load matrix
        wmin[:gr, :] = wg
        wmin[:gr, 2] = self.gamma_g_min * wmin[:gr, 2]
        wmin[gr : gr + qr, :] = wq
        wmin[gr : gr + qr, 2] = self.gamma_q_min * wmin[gr : gr + qr, 2]

        # Parameters for looping over loadcases
        N = self.ba.beam.no_spans
        vResults = []

        # Maximum support hogging and reaction -
        # adjacent spans fully loaded with MAX, other spans loaded with MIN
        n_max_hog = max(N - 1, 0)
        for i in range(1, n_max_hog + 1):
            w = wmin.copy()
            adjacent_spans = np.array([i, i + 1])
            mask = np.isin(wmax[:, 0], adjacent_spans)
            w[mask] = wmax[mask]
            res = analyze_loadcase(w)
            vResults.append(res)

        # Odd numbered spans loaded with MAX for maximum sagging moments
        w = wmax.copy()  # set all loading to maximum
        odd_spans = np.array([i for i in range(2, N + 1, 2)])
        mask = np.isin(wmax[:, 0], odd_spans)
        w[mask] = wmin[mask]
        res = analyze_loadcase(w)
        vResults.append(res)

        # Even numbered spans loaded with MAX for maximum sagging moments
        w = wmin.copy()  # set all loading to minimum
        even_spans = np.array([i for i in range(2, N + 1, 2)])
        mask = np.isin(wmin[:, 0], even_spans)
        w[mask] = wmax[mask]  # make the load a max
        res = analyze_loadcase(w)
        vResults.append(res)

        # All spans loaded with MAX
        res = analyze_loadcase(wmax)
        vResults.append(res)

        return Envelopes(vResults)
