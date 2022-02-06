"""
PyCBA - Continuous Beam Analysis - Influence Lines Module
"""
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from .analysis import BeamAnalysis


class InfluenceLines:
    """
    Creates influence lines for an arbitrary beam configuration using CBA
    """

    def __init__(
        self,
        L: np.ndarray,
        EI: Union[float, np.ndarray],
        R: np.ndarray,
        eletype: Optional[np.ndarray] = None,
    ):
        """
        Constructs an influence line object for a beam

        Parameters
        ----------
        L : np.ndarray
            A vector of span lengths.
        EI : Union[float, np.ndarray]
            A vector of member flexural rigidities.
        R : np.ndarray
            A vector describing the support conditions at each member end.
        eletype : Optional[np.ndarray]
            A vector of the member types. Defaults to a fixed-fixed element.

        Returns
        -------
        None.

        """
        self.ba = BeamAnalysis(L=L, EI=EI, R=R, eletype=eletype)
        self.L = self.ba.beam.length
        self.vResults = []
        self.pos = []

    def create_ils(self, step: Optional[float] = None, load_val: float = 1.0):
        """
        Creates the influence lines by marching the unit load (`load_val`) across
        the defined beam configuration in `step` distance increments, storing the
        static analysis results in a vector of :class:`pycba.results.BeamResults`.

        Parameters
        ----------
        step : Optional[float]
            The distance increment to move the unit load; defaults to beam length / 100.
        load_val : float, optional
            The nominal value of the "unit load". The default is 1.0.

        Raises
        ------
        ValueError
            If a static beam analysis does not succeed, usually due to a beam
            configuration error.

        Returns
        -------
        None.
        """
        self.vResults = []  # reset

        if step is None:
            step = self.L / 100

        npts = round(self.L / step) + 1

        for i in range(npts):
            # load position
            pos = i * step
            self.pos.append(pos)
            # locate load on span
            ispan, pos_in_span = self.ba.beam.get_local_span_coords(pos)
            if ispan == -1:
                load_val = 0.0
            # assemble and set load matrix
            self.ba.set_loads([[ispan + 1, 2, load_val, pos_in_span, 0]])
            # analyze
            out = self.ba.analyze()
            if out != 0:
                raise ValueError("IL analysis did not succeed")
                return
            self.vResults.append(self.ba.beam_results)

    def get_il(self, poi: float, load_effect: str) -> (np.ndarray, np.ndarray):
        """
        Returns the influence line at a point of interest for a load effect.

        Parameters
        ----------
        poi : float
            The position of interest in global coordinates along the length of the
            beam.
        load_effect : str
            A single character to identify the load effect of interest, currently
            one of:

                - **V**: shear force
                - **M**: bending moment
                - **R**: vertical reaction at a fully restrained support

            The vertical reaction nearest the `poi` is used. For moment reactions
            use a poi at or just beside the support.

        Returns
        -------
        (x,eta) : tuple(np.ndarray,np.ndarray)
            A tuple of the vectors of abcissa and influence ordinates.
        """
        if not self.vResults:
            self.create_ils()

        x = self.vResults[0].results.x
        idx = (np.abs(x - poi)).argmin()
        # find the nearest support to the poi
        idxr = (
            np.abs(np.cumsum(np.insert(self.ba.beam.mbr_lengths, 0, 0)) - poi)
        ).argmin()
        npts = len(self.vResults)
        eta = np.zeros(npts)

        for i, res in enumerate(self.vResults):
            if load_effect == "V":
                eta[i] = res.results.V[idx]
            elif load_effect == "R":
                eta[i] = res.R[idxr]
            else:
                eta[i] = res.results.M[idx]

        return (np.array(self.pos), eta)

    def plot_il(self, poi: float, load_effect: str, ax: Optional[plt.Axes] = None):
        """
        Retrieves and plots the IL on either a supplied or new axes.

        Parameters
        ----------
        poi : float
            The position of interest in global coordinates along the length of the
            beam.
        load_effect : str
            A single character to identify the load effect of interest, currently
            one of:

                - **V**: shear force
                - **M**: bending moment
                - **R**: vertical reaction at a fully restrained support

            The vertical reaction nearest the `poi` is used. For moment reactions
            use a poi at or just beside the support.
        ax : Optional[plt.Axes]
            A user-supplied matplotlib Axes object; when None (default), one is
            created for the plot.
        """

        (x, y) = self.get_il(poi, load_effect)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot([0, self.L], [0, 0], "k", lw=2)
        ax.plot(x, y, "r")
        ax.grid()
        ax.set_ylabel("Influence Ordinate")
        ax.set_xlabel("Distance along beam (m)")
        ax.set_title(f"IL for {load_effect} at {poi}")
        plt.tight_layout()
