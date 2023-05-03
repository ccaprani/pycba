"""
PyCBA - Continuous Beam Analysis - Bridge Crossing Module
"""
from __future__ import annotations  # https://bit.ly/3KYiL2o
from typing import Optional, Union, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from .analysis import BeamAnalysis
from .results import Envelopes, BeamResults
from .vehicle import Vehicle


class BridgeAnalysis:
    """
    Performs a bridge crossing analysis for a defined vehicle. The vehicle is moved
    from the zero global x-coordinate of the beam until it has left the beam at the
    far end.
    """

    def __init__(
        self, ba: Optional[BeamAnalysis] = None, veh: Optional[Vehicle] = None
    ):
        """
        Can instantiate with nothing and later add or define the objects, or
        with instantiate with pre-defined bridge and vehicle objects.

        Parameters
        ----------
        ba : Optional[BeamAnalysis], optional
            A :class:`pycba.analysis.BeamAnalysis` object. The default is None.
        veh : Optional[Vehicle], optional
            A :class:`pycba.bridge.Vehicle` object. The default is None.

        Returns
        -------
        None.
        """
        self.ba = ba
        self.veh = veh
        self.vResults = []
        self.pos = []

    def add_bridge(
        self,
        L: np.ndarray,
        EI: Union[float, np.ndarray],
        R: np.ndarray,
        eletype: Optional[np.ndarray] = None,
    ):
        """
        Create and add a beam to a bridge analysis

        Parameters
        ----------
        L : np.ndarray
            A vector of span lengths.
        EI : Union[float, np.ndarray]
            A vector of member flexural rigidities.
        R : np.ndarray
            A vector describing the support conditions at each member end.
        LM : Optional[list[list[Union[int, float]]]]
            The load matrix: a list of loads on the beam; each load with several
            parameters.
        eletype : Optional[np.ndarray]
            A vector of the member types. Defaults to a fixed-fixed element.

        Returns
        -------
        ba : BeamAnalysis
            A :class:`pycba.analysis.BeamAnalysis` object.
        """
        self.ba = BeamAnalysis(L=L, EI=EI, R=R, eletype=eletype)
        return self.ba

    def set_bridge(self, ba: BeamAnalysis):
        """
        Set the bridge for the bridge analysis.

        Parameters
        ----------
        ba : BeamAnalysis
            A :class:`pycba.analysis.BeamAnalysis` object.

        Returns
        -------
        None.

        """
        self.ba = ba

    def add_vehicle(self, axle_spacings: np.ndarray, axle_weights: np.ndarray):
        """
        Create and add the vehicle to the analysis

        Parameters
        ----------
        axle_spacings : np.ndarray
            A vector of axle spacings of length one fewer than the length of the
            vector of axle weights
        axle_weights : np.ndarray
            A vector of axle weights, length one greater than the length of the
            axle spacings vector.

        Returns
        -------
        veh : Vehicle
            A :class:`pycba.bridge.Vehicle` object.
        """
        self.veh = Vehicle(axle_spacings, axle_weights)
        return self.veh

    def set_vehicle(self, veh: Vehicle):
        """
        Set the vehicle for the bridge analysis.

        Parameters
        ----------
        veh : Vehicle
            A :class:`pycba.bridge.Vehicle` object.

        Returns
        -------
        None.

        """
        self.veh = veh

    def static_vehicle(self, pos: float, plotflag: bool = False) -> BeamResults:
        """
        Performs a single analysis for the vehicle, static at a given position

        Parameters
        ----------
        pos : float
            The location of the front axle of the vehicle in global beam coordinates.
        plotflag : bool, optional
            Whether or not to plot the results. The default is False.

        Raises
        ------
        ValueError
            If a static beam analysis does not succeed, usually due to a beam
            configuration error.

        Returns
        -------
        ba: BeamResults
            The `pycba.Beamresults` object containing the analysis results.
        """
        self._check_objects()

        out = self._single_analysis(pos)
        if out != 0:
            raise ValueError("Bridge analysis did not succeed")
            return

        if plotflag:
            self.plot_static(pos)

        return self.ba.beam_results

    def _single_analysis(self, pos: float) -> int:
        """
        Internal function for efficiency in run_vehicle - assumes Bridge and
        Vehicle are already defined/checked in UI functions

        Parameters
        ----------
        pos : float
            The location of the front axle of the vehicle in global beam coordinates.

        Returns
        -------
        int
            0 if the analysis succeeds.

        """

        axle_positions = pos - self.veh.axle_coords

        # Create the CBA Load Matrix, checking axle positions, etc
        LM = []
        for iaxle in range(self.veh.NoAxles):
            load = self.veh.axw[iaxle]
            ispan, pos_in_span = self.ba.beam.get_local_span_coords(
                axle_positions[iaxle]
            )
            if ispan != -1:
                LM.append([ispan + 1, 2, load, pos_in_span, 0])

        self.ba.set_loads(LM)
        return self.ba.analyze()

    def run_vehicle(
        self, step: float, plot_env: bool = False, plot_all: bool = False
    ) -> Envelopes:
        """
        Runs the vehicle over the bridge performing a static analysis at each point

        Parameters
        ----------
        step : float
            The distance increment to move the vehicle.
        plot_env : bool, optional
            Whether or not to plot the results envelope. The default is False.
        plot_all : bool, optional
            Whether or not to plot the results for each position as an animation.
            The default is False.

        Raises
        ------
        ValueError
            If a static beam analysis does not succeed, usually due to a beam
            configuration error.

        Returns
        -------
        Envelopes
            The load effect envelopes for the traverse; a `pycba.Envelopes` object.

        """
        self._check_objects()
        self.pos = []
        self.vResults = []
        npts = round((self.ba.beam.length + self.veh.L) / step) + 1

        if plot_all:
            fig, axs = plt.subplots(2, 1, sharex=True)

        for i in range(npts):
            # load position
            pos = i * step
            self.pos.append(pos)
            out = self._single_analysis(pos)
            if out != 0:
                raise ValueError("Bridge analysis did not succeed at {pos=}")
                return
            if plot_all:
                self.plot_static(pos, axs)
                plt.pause(0.01)
            self.vResults.append(self.ba.beam_results)

        env = Envelopes(self.vResults)

        if plot_env:
            self.plot_envelopes(env)

        return env

    def critical_values(
        self, env: Envelopes
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        From the envelopes output, returns the extreme values, their locations,
        and the position of the vehicle for each in a dictionary of dictionaries

        Parameters
        ----------
        env : Envelopes
            An `pycba.Envelopes` object containing the results of a moving load
            analysis.

        Raises
        ------
        ValueError
            If the supplied envelope is inconsistent with the current
            `pycba.bridge.BridgeAnalysis` object.

        Returns
        -------
        crit_values : Dict[str, Dict[str, Union[float, np.ndarray]]]
            A dictionary of dictionaries containing the critical values (i.e. extremes)
            of each of the load effects, both maximum and minimum.
        """

        crit_values = {}
        indx = {}

        Mmax = env.Mmax.max()
        Mmin = env.Mmin.min()
        Vmax = env.Vmax.max()
        Vmin = env.Vmin.min()

        # Find the indices of the critical vehicle positions
        indx["Mmax"] = [
            i
            for i, res in enumerate(self.vResults)
            if np.isclose(res.results.M.max(), Mmax)
        ]
        indx["Mmin"] = [
            i
            for i, res in enumerate(self.vResults)
            if np.isclose(res.results.M.min(), Mmin)
        ]
        indx["Vmax"] = [
            i
            for i, res in enumerate(self.vResults)
            if np.isclose(res.results.V.max(), Vmax)
        ]
        indx["Vmin"] = [
            i
            for i, res in enumerate(self.vResults)
            if np.isclose(res.results.V.min(), Vmin)
        ]

        # Now check for any errors
        if [] in indx.values():
            raise ValueError("Envelope not from the current bridge analysis")

        # Good to proceed
        crit_values["Mmax"] = {
            "val": Mmax,
            "at": env.x[env.Mmax.argmax()],
            "pos": [self.pos[i] for i in indx["Mmax"]],
        }
        crit_values["Mmin"] = {
            "val": Mmin,
            "at": env.x[env.Mmin.argmin()],
            "pos": [self.pos[i] for i in indx["Mmin"]],
        }
        crit_values["Vmax"] = {
            "val": Vmax,
            "at": env.x[env.Vmax.argmax()],
            "pos": [self.pos[i] for i in indx["Vmax"]],
        }
        crit_values["Vmin"] = {
            "val": Vmin,
            "at": env.x[env.Vmin.argmin()],
            "pos": [self.pos[i] for i in indx["Vmin"]],
        }
        crit_values["nsup"] = env.nsup
        for i in range(env.nsup):
            crit_values[f"Rmax{i}"] = {
                "val": env.Rmax[i, :].max(),
                "pos": self.pos[env.Rmax[i, :].argmax()],
            }
            crit_values[f"Rmin{i}"] = {
                "val": env.Rmin[i, :].min(),
                "pos": self.pos[env.Rmin[i, :].argmin()],
            }

        return crit_values

    def envelopes_ratios(
        self, trial_env: Envelopes, ref_env: Envelopes
    ) -> Dict[str, np.ndarray]:
        """
        Returns the ratios of two sets of envelopes considering zero values and
        reactions. Note that ratios are only meaningful for any one location on the
        beam, and so reaction envelopes ratios reduce to a scalar value. Ratios are
        absolute, and zeroed if within 1e-3 absolute tolerance of zero.

        Parameters
        ----------
        trial_env : Envelopes
            The numerator `pycba.Envelopes` object, usually from the vehicle seeking
            access to the bridge.
        ref_env : Envelopes
            The denominator `pycba.Envelopes` object, usually from the reference or
            benchamrk of acceptable load effects on the bridge. Can be from a single
            notional vehicle, or a suite of such vehicles.

        Raises
        ------
        ValueError
            The envelopes need to be from the same bridge analysis object.

        Returns
        -------
        Dict[str,np.ndarray]
            A dictionary of ratios between the two envelopes, considering the
            maximum and minimum of each load effect.
        """

        if ref_env.npts != trial_env.npts or ref_env.nsup != trial_env.nsup:
            raise ValueError("Ratios can only be found for compatible envelopes")

        def get_ratio(a, b):
            """
            Zeroes infinities when a ref load effect is zero. b is reference vector
            """
            return np.abs(
                np.divide(
                    a,
                    b,
                    out=np.zeros_like(a),
                    where=~np.isclose(b, np.zeros_like(b), atol=1e-3, rtol=0.0),
                )
            )

        env_ratios = {}
        env_ratios["x"] = ref_env.x
        env_ratios["Mmax"] = get_ratio(trial_env.Mmax, ref_env.Mmax)
        env_ratios["Mmin"] = get_ratio(trial_env.Mmin, ref_env.Mmin)
        env_ratios["Vmax"] = get_ratio(trial_env.Vmax, ref_env.Vmax)
        env_ratios["Vmin"] = get_ratio(trial_env.Vmin, ref_env.Vmin)
        env_ratios["nsup"] = ref_env.nsup

        maxvals = get_ratio(trial_env.Rmaxval, ref_env.Rmaxval)
        minvals = get_ratio(trial_env.Rminval, ref_env.Rminval)
        for i in range(ref_env.nsup):
            env_ratios[f"Rmax{i}"] = maxvals[i]
            env_ratios[f"Rmin{i}"] = minvals[i]

        return env_ratios

    def plot_static(self, pos: float, axs: Optional[plt.Axes] = None):
        """
        Plots for analysis of static vehicle

        Parameters
        ----------
        pos : float
            The position of the front axle of the vehicle in global bridge
            coordinates.
        axs : Optional[plt.Axes], optional
            The axes on which to plot; if None is supplied, one is created.
            The default is None.

        Returns
        -------
        None.
        """
        res = self.ba.beam_results.results
        L = self.ba.beam.length

        if axs is None:
            fig, axs = plt.subplots(3, 1, sharex=True)

        ax0 = 1
        if len(axs) == 2:  # load effect only
            ax0 = 0
        else:
            ax = axs[0]
            ax.bar(pos - self.veh.axle_coords, self.veh.axw, color="r")
            ax.set_ylabel("Axle Weights (kN)")
            ax.grid()

        ax = axs[ax0]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, -res.M, "r")
        ax.grid()
        ax.set_ylabel("Bending Moment (kNm)")

        ax = axs[ax0 + 1]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, res.V, "r")
        ax.grid()
        ax.set_ylabel("Shear Force (kN)")
        ax.set_xlabel("Distance along beam (m)")

    def plot_envelopes(self, env: Envelopes):
        """
        Plots the envelopes of load effects from a vehicle traverse analysis

        Parameters
        ----------
        env : Envelopes
            An `pycba.Envelopes` object containing the results of a moving load
            analysis.

        Returns
        -------
        None
        """

        L = self.ba.beam.length
        x = env.x
        nreactions = env.nsup

        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07)

        # Shear and moment in left panel
        subfigs[0].suptitle("Stress Resultants")
        axsLeft = subfigs[0].subplots(2, 1, sharex=True)

        ax = axsLeft[0]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(x, -env.Mmax, "r")
        ax.plot(x, -env.Mmin, "b")
        ax.grid()
        ax.set_ylabel("Bending Moment (kNm)")

        ax = axsLeft[1]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(x, env.Vmax, "r")
        ax.plot(x, env.Vmin, "b")
        ax.grid()
        ax.set_ylabel("Shear Force (kN)")
        ax.set_xlabel("Distance along beam (m)")

        # Reactions in right panel
        subfigs[1].suptitle("Support Reactions")

        # Check if consistent envelope
        if len(self.pos) == env.Rmax.shape[1]:
            axsRight = subfigs[1].subplots(nreactions, 1, sharex=True)
            for i, ax in enumerate(axsRight):
                ax.plot([0, L], [0, 0], "k", lw=2)
                ax.plot(self.pos, env.Rmax[i, :], "r")
                ax.plot(self.pos, env.Rmin[i, :], "b")
                ax.grid()
                ax.set_ylabel(f"Reaction {i+1} (kN/kNm)")
            axsRight[-1].set_xlabel("Position of Front Axle (m)")

        else:  # Otherwise envelope of envelopes
            axsRight = subfigs[1].subplots(2, 1, sharex=True)
            for i, (ax, le, col) in enumerate(
                zip(axsRight, ["max", "min"], ["r", "b"])
            ):
                r = eval(f"env.R{le}val")  # kinda yuk!
                ax.bar(np.arange(env.nsup), r, color=col)
                ax.set_xticks(np.arange(env.nsup))
                ax.set_xticklabels([f"R{i+1}" for i in range(env.nsup)])
                ax.set_ylabel(f"Reactions [{le}]")
                ax.grid()
            axsRight[1].set_xlabel("Reaction ID")

    def plot_ratios(self, env_ratios: Dict[str, np.ndarray]):
        """
        Plots the output of :meth:`pycba.bridge.BridgeAnalysis.envelopes_ratios`.

        Parameters
        ----------
        env_ratios : Dict[str,np.ndarray]
            The dictionary of envelopes ratios.

        Raises
        ------
        ValueError
            Inconsistency in the dictionary entries.

        Returns
        -------
        None.
        """

        # Set of keys we want to confirm are present
        check_keys = set(
            ["x", "Mmax", "Mmin", "Vmax", "Vmin", "nsup", "Rmax0", "Rmin0"]
        )
        if not check_keys.issubset(env_ratios.keys()):
            raise ValueError(
                "Dictionary argument does not contain sufficient ratios information"
            )

        x = env_ratios["x"]
        L = x[-1]
        nsup = env_ratios["nsup"]

        fig = plt.figure(constrained_layout=True, figsize=(10, 4))
        subfigs = fig.subfigures(1, 2, wspace=0.07)

        # Shear and moment in left panel
        subfigs[0].suptitle("Stress Resultants")
        axsLeft = subfigs[0].subplots(2, 1, sharex=True)

        ax = axsLeft[0]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(x, env_ratios["Mmax"], "r")
        ax.plot(x, env_ratios["Mmin"], "b")
        ax.grid()
        ax.set_ylabel("Bending Moment Ratio")

        ax = axsLeft[1]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(x, env_ratios["Vmax"], "r")
        ax.plot(x, env_ratios["Vmin"], "b")
        ax.grid()
        ax.set_ylabel("Shear Force Ratio")
        ax.set_xlabel("Distance along beam (m)")

        # Reactions in right panel
        subfigs[1].suptitle("Support Reactions")
        axsRight = subfigs[1].subplots(2, 1, sharex=True)

        for i, (ax, le, col) in enumerate(zip(axsRight, ["max", "min"], ["r", "b"])):
            r = np.array([env_ratios[f"R{le}{i}"] for i in range(nsup)])
            ax.bar(np.arange(nsup), r, color=col)
            ax.set_xticks(np.arange(nsup))
            ax.set_xticklabels([f"R{i+1}" for i in range(nsup)])
            ax.set_ylabel(f"Reaction Ratio [{le}]")
            ax.grid()
        axsRight[1].set_xlabel("Reaction ID")

    def _check_objects(self):
        """
        Check that suitable objects are defined before an analysis is run.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not self.ba:
            raise ValueError("A bridge must be defined in advance")
        if not self.veh:
            raise ValueError("A vehicle must be defined in advance")
