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
        npts = len(self.vResults)
        eta = np.zeros(npts)

        # Preparations for reaction ILs
        #
        # Get vector of the node locations
        node_locations = np.cumsum(np.insert(self.ba.beam.mbr_lengths, 0, 0))
        # Link the supported DOF to the index in the BeamAnalysis reactions vector
        idx_mask = np.zeros_like(self.ba._beam.restraints)
        idx_mask[np.where(np.array(self.ba._beam.restraints) == -1)] = np.arange(
            self.ba.beam.no_fixed_restraints
        )

        # idx = np.abs(x - poi).argmin()

        if load_effect.upper() == "V":
            dx = x[2] - x[1]
            idx = np.where(np.abs(x - poi) <= dx * 1e-6)[0][0]
            for i, res in enumerate(self.vResults):
                eta[i] = res.results.V[idx]

        elif load_effect.upper() == "R":
            #
            # Getting the correct reaction is tricky
            #
            # The indices of the supported DOFs wrt the node locations vector
            vert_sups_dof_idx = np.where(np.array(self.ba._beam.restraints)[::2] == -1)[
                0
            ]
            # The locations then of these supports
            vert_sups_locs = node_locations[vert_sups_dof_idx]
            # The index of the closest support
            closest_vert_sup_idx = np.abs(vert_sups_locs - poi).argmin()
            # And its value
            closest_vert_sup = vert_sups_locs[closest_vert_sup_idx]
            # And now the index of this support in the node locations vector
            vert_sup_node_idx = np.where(node_locations == closest_vert_sup)[0][0]
            # And hence its index in the overall DOFs vector
            vert_sup_dof_idx = 2 * vert_sup_node_idx
            # And finally the index of the support nearest the POI in the reactions vector
            vert_sup_idx = idx_mask[vert_sup_dof_idx]

            for i, res in enumerate(self.vResults):
                eta[i] = res.R[vert_sup_idx]

        elif load_effect.upper() == "MR":
            #
            # Follows the same logic for the vertical reaction
            #
            mt_sups_dof_idx = np.where(np.array(self.ba._beam.restraints)[1::2] == -1)[
                0
            ]
            mt_sups_locs = node_locations[mt_sups_dof_idx]
            closest_mt_sup_idx = np.abs(mt_sups_locs - poi).argmin()
            closest_mt_sup = mt_sups_locs[closest_mt_sup_idx]
            mt_sup_node_idx = np.where(node_locations == closest_mt_sup)[0][0]
            mt_sup_dof_idx = 2 * mt_sup_node_idx + 1
            mt_sup_idx = idx_mask[mt_sup_dof_idx]

            for i, res in enumerate(self.vResults):
                eta[i] = res.R[mt_sup_idx]

        else:
            dx = x[2] - x[1]
            idx = np.where(np.abs(x - poi) <= dx * 1e-6)[0][0]
            for i, res in enumerate(self.vResults):
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
