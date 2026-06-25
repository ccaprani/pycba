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
from .load import add_LM


def resolve_shear_points(
    beam,
    points: Optional[Union[float, List[float], np.ndarray]] = None,
    d_from_supports: Optional[float] = None,
) -> (Dict[int, np.ndarray], np.ndarray):
    """
    Resolve requested "shear point" sections to per-member local coordinates.

    Shear points are sections at which the shear is recovered exactly (on both
    sides), for example the face of a support or a code-specified distance ``d``
    (or ``d_v``) from a support.  They are specified in global beam coordinates
    and are fixed for the analysis, so the evaluation grid stays consistent
    across a moving-load traverse.

    Parameters
    ----------
    beam : Beam
        The :class:`pycba.beam.Beam` to resolve against.
    points : float or array-like, optional
        Global coordinate(s), measured from the left end of the beam, at which
        shear points are required.
    d_from_supports : float, optional
        Convenience: add a shear point at this distance either side of every
        vertical support (clipped to the deck).  Useful for shear checks at a
        distance ``d``/``d_v`` from the support faces.

    Returns
    -------
    sp : Dict[int, np.ndarray]
        Mapping of 0-based member index to a sorted vector of member-local
        coordinates, suitable for ``BeamResults(shear_points=...)``.
    points_global : np.ndarray
        The sorted vector of global section coordinates actually placed.
    """
    globals_ = []
    if points is not None:
        globals_.extend(float(p) for p in np.atleast_1d(points))
    if d_from_supports is not None:
        node_x = np.cumsum(np.insert(np.asarray(beam.mbr_lengths, dtype=float), 0, 0.0))
        restraints = np.asarray(beam.restraints)
        for i, x0 in enumerate(node_x):
            if restraints[2 * i] != 0:  # a vertical support (fixed or spring)
                globals_.extend([x0 - d_from_supports, x0 + d_from_supports])

    # Keep only on-deck interior sections, de-duplicated and sorted.
    points_global = np.array(sorted({g for g in globals_ if 0.0 < g < beam.length}))

    sp: Dict[int, list] = {}
    for g in points_global:
        ispan, pos_in_span = beam.get_local_span_coords(g)
        if ispan != -1:
            sp.setdefault(ispan, []).append(pos_in_span)

    return {k: np.array(sorted(v)) for k, v in sp.items()}, points_global


class BridgeAnalysis:
    """
    Performs a bridge crossing analysis for a defined vehicle. The vehicle is moved
    from the zero global x-coordinate of the beam until it has left the beam at the
    far end.

    Any loads already defined on the `BeamAnalysis` object are retained and
    superimposed in each vehicle position analysis.
    """

    def __init__(
        self, ba: Optional[BeamAnalysis] = None, veh: Optional[Vehicle] = None
    ):
        """
        Can instantiate with nothing and later add or define the objects, or
        with instantiate with pre-defined bridge and vehicle objects.

        Any loads already defined on the `BeamAnalysis` object are retained in
        each vehicle position analysis.

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
        # Global coordinates of any shear-point sections placed in the last
        # analysis (populated by run_vehicle/static_vehicle when requested).
        self.shear_points_x = np.array([])

        self.static_LM = []

        if self.ba:
            self.static_LM = self.ba.beam.loads

    def add_bridge(
        self,
        L: np.ndarray,
        EI: Union[float, np.ndarray],
        R: np.ndarray,
        eletype: Optional[np.ndarray] = None,
        GAv: Optional[Union[float, np.ndarray]] = None,
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
        eletype : Optional[np.ndarray]
            A vector of the member types. Defaults to a fixed-fixed element.
        GAv : Optional[Union[float, np.ndarray]]
            Transverse shear rigidity ``G·A_v`` of each span.  When given, the
            bridge is analysed with shear-deformable Timoshenko elements; the
            default (``None``) uses Euler–Bernoulli elements.

        Returns
        -------
        ba : BeamAnalysis
            A :class:`pycba.analysis.BeamAnalysis` object.
        """
        self.ba = BeamAnalysis(L=L, EI=EI, R=R, eletype=eletype, GAv=GAv)
        self.static_LM = self.ba.beam.loads
        return self.ba

    def set_bridge(self, ba: BeamAnalysis):
        """
        Set the bridge for the bridge analysis.

        Any loads already defined on the `BeamAnalysis` object are retained in
        each vehicle position analysis.

        Parameters
        ----------
        ba : BeamAnalysis
            A :class:`pycba.analysis.BeamAnalysis` object.

        Returns
        -------
        None.

        """
        self.ba = ba
        self.static_LM = self.ba.beam.loads

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

    def static_vehicle(
        self,
        pos: float,
        plotflag: bool = False,
        shear_points: Optional[Union[float, List[float], np.ndarray, Dict]] = None,
        d_from_supports: Optional[float] = None,
    ) -> BeamResults:
        """
        Performs a single analysis for the vehicle, static at a given position

        Parameters
        ----------
        pos : float
            The location of the front axle of the vehicle in global beam coordinates.
        plotflag : bool, optional
            Whether or not to plot the results. The default is False.
        shear_points : float, array-like or dict, optional
            Section(s) at which the shear is recovered exactly on both sides.
            Either global coordinate(s) (resolved via :func:`resolve_shear_points`)
            or a pre-resolved ``{member_index: local_coords}`` mapping.
        d_from_supports : float, optional
            Convenience: add a shear point this distance either side of every
            vertical support (see :func:`resolve_shear_points`).

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
        self._apply_shear_points(shear_points, d_from_supports)

        try:
            out = self._single_analysis(pos)
            if out != 0:
                raise ValueError("Bridge analysis did not succeed")
        finally:
            self.ba.shear_points = None

        if plotflag:
            self.plot_static(pos)

        return self.ba.beam_results

    def _apply_shear_points(self, shear_points, d_from_supports):
        """
        Resolve and arm shear-point sections on the underlying beam analysis.

        Sets ``self.ba.shear_points`` (consumed by each ``analyze()`` call) and
        records the placed global section coordinates in ``self.shear_points_x``
        for :meth:`critical_values`.  A pre-resolved ``dict`` is used as-is; any
        other spec is resolved with :func:`resolve_shear_points`.
        """
        if isinstance(shear_points, dict):
            sp = shear_points
            self.shear_points_x = np.array([])
        elif shear_points is None and d_from_supports is None:
            sp = None
            self.shear_points_x = np.array([])
        else:
            sp, self.shear_points_x = resolve_shear_points(
                self.ba.beam, shear_points, d_from_supports
            )
        self.ba.shear_points = sp if sp else None

    def _axle_LM(self, pos: float) -> List[list]:
        """
        Build the load-matrix rows for the vehicle axles with the front axle at
        ``pos``.  Only axles currently on the deck are included.

        Parameters
        ----------
        pos : float
            The location of the front axle in global beam coordinates.

        Returns
        -------
        List[list]
            Point-load (type 2) rows for the on-deck axles.
        """
        axle_positions = pos - self.veh.axle_coords

        rows = []
        for iaxle in range(self.veh.NoAxles):
            load = self.veh.axw[iaxle]
            ispan, pos_in_span = self.ba.beam.get_local_span_coords(
                axle_positions[iaxle]
            )
            if ispan != -1:
                rows.append([ispan + 1, 2, load, pos_in_span, 0])
        return rows

    def _interval_udl_LM(self, xa: float, xb: float, w: float) -> List[list]:
        """
        Convert a global UDL interval ``[xa, xb]`` of intensity ``w`` into
        per-span partial-UDL (type 3) load-matrix rows, splitting the interval
        at the span boundaries.  Portions off the deck are ignored.

        Parameters
        ----------
        xa, xb : float
            Start and end of the loaded interval in global coordinates.
        w : float
            The UDL intensity.

        Returns
        -------
        List[list]
            Partial-UDL rows ``[span, 3, w, a, c]`` for each loaded span portion.
        """
        beam = self.ba.beam
        node_x = np.cumsum(np.insert(np.asarray(beam.mbr_lengths, dtype=float), 0, 0.0))
        rows = []
        for i in range(beam.no_spans):
            lo = max(xa, node_x[i])
            hi = min(xb, node_x[i + 1])
            if hi - lo > 1e-9:
                rows.append([i + 1, 3, w, lo - node_x[i], hi - lo])
        return rows

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
        # Superimpose the axle loads on any pre-existing loads on the beam
        LM = add_LM(self.static_LM, self._axle_LM(pos))

        self.ba.set_loads(LM)
        return self.ba.analyze()

    def run_vehicle(
        self,
        step: float,
        plot_env: bool = False,
        plot_all: bool = False,
        pos_start: Optional[float] = None,
        pos_end: Optional[float] = None,
        shear_points: Optional[Union[float, List[float], np.ndarray, Dict]] = None,
        d_from_supports: Optional[float] = None,
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
        pos_start : Optional[float], optional
            The starting position of the front axle. Defaults to 0 (front axle at
            the left end of the beam).
        pos_end : Optional[float], optional
            The ending position of the front axle. Defaults to beam length plus
            vehicle length (front axle past the right end of the beam).
        shear_points : float, array-like or dict, optional
            Section(s) at which the shear is recovered exactly on both sides for
            every vehicle position - e.g. support faces or a distance ``d``/``d_v``
            from a support, for bridge shear assessment.  Either global
            coordinate(s) (resolved via :func:`resolve_shear_points`) or a
            pre-resolved ``{member_index: local_coords}`` mapping.  The resulting
            sections are reported by :meth:`critical_values`.
        d_from_supports : float, optional
            Convenience: add a shear point this distance either side of every
            vertical support (see :func:`resolve_shear_points`).

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
        self._apply_shear_points(shear_points, d_from_supports)

        if pos_start is None:
            pos_start = 0.0
        if pos_end is None:
            pos_end = self.ba.beam.length + self.veh.L

        npts = round((pos_end - pos_start) / step) + 1

        if plot_all:
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

        try:
            for i in range(npts):
                # load position
                pos = pos_start + i * step
                self.pos.append(pos)
                out = self._single_analysis(pos)
                if out != 0:
                    raise ValueError("Bridge analysis did not succeed at {pos=}")
                if plot_all:
                    self.plot_static(pos, axs)
                    plt.pause(0.01)
                self.vResults.append(self.ba.beam_results)
        finally:
            self.ba.shear_points = None

        env = Envelopes(self.vResults)

        if plot_env:
            self.plot_envelopes(env)

        return env

    def run_load_model(
        self,
        step: float,
        w_lane: float,
        gap: float = 0.0,
        plot_env: bool = False,
        pos_start: Optional[float] = None,
        pos_end: Optional[float] = None,
        shear_points: Optional[Union[float, List[float], np.ndarray, Dict]] = None,
        d_from_supports: Optional[float] = None,
    ) -> Envelopes:
        """
        Run a moving load model: the vehicle together with a co-travelling lane
        UDL, enveloped over the traverse.

        At each vehicle position the axles are placed as point loads (as in
        :meth:`run_vehicle`) and a uniform lane UDL of intensity ``w_lane`` is
        applied across the whole deck *except* a gap spanning the vehicle
        footprint ``[pos - vehicle.L, pos]``, optionally widened by ``gap`` on
        each side.  This supports code load models that pair a notional truck
        with a lane UDL (e.g. AS5100 M1600, AASHTO HL-93, Eurocode LM1).

        The lane UDL is applied over the full deck outside the gap.  For a
        continuous beam, loading only the same-sign influence-line regions can
        be more adverse for a given effect; influence-line patterning of the
        lane UDL is a planned refinement.  To pattern manually, combine separate
        envelopes with :meth:`Envelopes.sum` / :meth:`Envelopes.augment`.

        Parameters
        ----------
        step : float
            The distance increment to move the vehicle.
        w_lane : float
            The lane UDL intensity (same sign convention as a beam UDL).
        gap : float, optional
            Length of deck kept clear of the lane UDL on each side of the
            vehicle footprint.  The default (``0.0``) clears exactly the
            footprint; a single-axle vehicle then leaves no gap unless ``gap``
            is set.
        plot_env : bool, optional
            Whether to plot the resulting envelope. The default is False.
        pos_start, pos_end : float, optional
            The traverse range of the front axle (see :meth:`run_vehicle`).
        shear_points : float, array-like or dict, optional
            Sections for exact both-sided shear recovery (see
            :meth:`run_vehicle`).
        d_from_supports : float, optional
            Convenience shear points either side of every support.

        Raises
        ------
        ValueError
            If a static beam analysis does not succeed.

        Returns
        -------
        Envelopes
            The load-effect envelopes for the traverse.
        """
        self._check_objects()
        self.pos = []
        self.vResults = []
        self._apply_shear_points(shear_points, d_from_supports)

        length = self.ba.beam.length
        if pos_start is None:
            pos_start = 0.0
        if pos_end is None:
            pos_end = length + self.veh.L

        npts = round((pos_end - pos_start) / step) + 1

        try:
            for i in range(npts):
                pos = pos_start + i * step
                self.pos.append(pos)

                # Lane UDL across the deck except the gap over the vehicle.
                excl_lo = pos - self.veh.L - gap
                excl_hi = pos + gap
                udl_rows = []
                if excl_lo > 0.0:
                    udl_rows += self._interval_udl_LM(0.0, min(excl_lo, length), w_lane)
                if excl_hi < length:
                    udl_rows += self._interval_udl_LM(max(excl_hi, 0.0), length, w_lane)

                LM = add_LM(self.static_LM, self._axle_LM(pos) + udl_rows)
                self.ba.set_loads(LM)
                out = self.ba.analyze()
                if out != 0:
                    raise ValueError(f"Bridge analysis did not succeed at {pos=}")
                self.vResults.append(self.ba.beam_results)
        finally:
            self.ba.shear_points = None

        env = Envelopes(self.vResults)

        if plot_env:
            self.plot_envelopes(env)

        return env

    def critical_values(
        self, env: Envelopes
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        From the envelopes output, returns the extreme values, their locations,
        and the position of the vehicle for each in a dictionary of dictionaries.

        Each moment entry (``Mmax``, ``Mmin``) also contains a ``"Vco"`` key with
        the coincident shear at the critical location. Each shear entry (``Vmax``,
        ``Vmin``) contains a ``"Mco"`` key with the coincident moment.

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
            of each of the load effects, both maximum and minimum, along with
            coincident values of the other effect.
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
            "Vco": env.Vco_Mmax[env.Mmax.argmax()],
        }
        crit_values["Mmin"] = {
            "val": Mmin,
            "at": env.x[env.Mmin.argmin()],
            "pos": [self.pos[i] for i in indx["Mmin"]],
            "Vco": env.Vco_Mmin[env.Mmin.argmin()],
        }
        crit_values["Vmax"] = {
            "val": Vmax,
            "at": env.x[env.Vmax.argmax()],
            "pos": [self.pos[i] for i in indx["Vmax"]],
            "Mco": env.Mco_Vmax[env.Vmax.argmax()],
        }
        crit_values["Vmin"] = {
            "val": Vmin,
            "at": env.x[env.Vmin.argmin()],
            "pos": [self.pos[i] for i in indx["Vmin"]],
            "Mco": env.Mco_Vmin[env.Vmin.argmin()],
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

        # Report the corrected critical shears at any requested shear points.
        # Each section was sampled by a station pair straddling it, so the
        # left- and right-hand shear limits are read from the stations either
        # side of the section coordinate.
        if len(self.shear_points_x) > 0:
            sp_block = {}
            for xg in self.shear_points_x:
                left = np.nonzero(env.x < xg)[0]
                right = np.nonzero(env.x > xg)[0]
                if len(left) == 0 or len(right) == 0:
                    continue
                il = left[np.argmax(env.x[left])]
                ir = right[np.argmin(env.x[right])]
                sp_block[float(xg)] = {
                    "Vmax": max(env.Vmax[il], env.Vmax[ir]),
                    "Vmin": min(env.Vmin[il], env.Vmin[ir]),
                    "Vmax_left": env.Vmax[il],
                    "Vmax_right": env.Vmax[ir],
                    "Vmin_left": env.Vmin[il],
                    "Vmin_right": env.Vmin[ir],
                }
            crit_values["shear_points"] = sp_block

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

    def plot_static(self, pos: float, axs: Optional[plt.Axes] = None, units=None):
        """
        Draw the bridge with the vehicle at a given position, above the
        instantaneous bending-moment and shear-force diagrams.

        The top panel shows the structural schematic - the deck with its real
        support symbols and any permanent loads - with the vehicle drawn as a
        small truck on the deck at ``pos`` (a wheel at each axle and a
        downward load arrow, labelled with its weight, for every axle that is
        currently on the bridge).  The two panels below show the bending
        moment and shear force for that single ("stationary") position; faint
        vertical lines mark the axle positions so the load effects can be
        related to the loads that cause them.

        Parameters
        ----------
        pos : float
            The position of the front axle of the vehicle in global bridge
            coordinates.
        axs : array_like of matplotlib.axes.Axes, optional
            Axes to draw into.  If ``None`` (default) a new three-panel figure
            is created and the analysis for ``pos`` is run first.  When axes
            are supplied the current results are used as-is: pass three axes
            (schematic + moment + shear) or two (moment + shear only).
        units : str or pycba.units.UnitSystem, optional
            Display unit system for the labels (see :func:`pycba.set_units`).

        Returns
        -------
        tuple(matplotlib.figure.Figure, numpy.ndarray) or None
            The figure and its axes when a new figure is created, otherwise
            ``None``.
        """
        from .units import resolve

        us = resolve(units)
        own_fig = axs is None
        fig = None
        if own_fig:
            self._check_objects()
            self._single_analysis(pos)
            fig, axs = plt.subplots(
                3,
                1,
                sharex=True,
                figsize=(10, 8),
                gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
            )

        res = self.ba.beam_results.results
        L = self.ba.beam.length

        if len(axs) >= 3:  # schematic + load effects
            self._draw_deck_and_vehicle(axs[0], pos, us)
            m_ax, v_ax = axs[1], axs[2]
        else:  # load effects only (e.g. the run_vehicle animation)
            m_ax, v_ax = axs[0], axs[1]

        # Faint markers linking on-deck axle positions to the diagrams below
        axle_x = pos - self.veh.axle_coords
        on_deck = axle_x[(axle_x >= 0.0) & (axle_x <= L)]

        m_ax.plot([0, L], [0, 0], "k", lw=2)
        m_ax.plot(res.x, -res.M, "r")
        for x in on_deck:
            m_ax.axvline(x, color="0.7", ls=":", lw=0.8, zorder=0)
        m_ax.grid()
        m_ax.set_ylabel(us.moment_axis)

        v_ax.plot([0, L], [0, 0], "k", lw=2)
        v_ax.plot(res.x, res.V, "r")
        for x in on_deck:
            v_ax.axvline(x, color="0.7", ls=":", lw=0.8, zorder=0)
        v_ax.grid()
        v_ax.set_ylabel(us.shear_axis)
        v_ax.set_xlabel(us.distance_axis)

        if own_fig:
            return fig, axs

    def _draw_deck_and_vehicle(self, ax, pos: float, us=None):
        """
        Draw the deck schematic (supports + permanent loads) with the vehicle
        rendered as a small truck at ``pos`` on the given axes.
        """
        from matplotlib.patches import Rectangle, Circle
        from .render import BeamPlotter
        from .units import resolve

        us = resolve(us)
        L = self.ba.beam.length
        u = 0.05 * L  # symbol unit, matching the schematic renderer

        # Bare deck with real supports (and any permanent loads), stretched to
        # fill the panel so it stays x-aligned with the diagrams below.
        BeamPlotter(self.ba.beam, self.static_LM).render_mpl(
            ax=ax,
            dimensions=False,
            labels=True,
            load_values=bool(self.static_LM),
            equal_aspect=False,
            units=us,
        )
        ax.set_xlabel("")

        axle_x = pos - self.veh.axle_coords
        w = self.veh.axw
        x_lo, x_hi = float(np.min(axle_x)), float(np.max(axle_x))

        rw = 0.30 * u  # wheel radius
        body_y0 = 2.0 * rw
        body_h = 1.1 * u
        body_top = body_y0 + body_h
        pad = 0.6 * u  # body overhang past the outer axles

        # Vehicle body, with a small cab on the leading (travel +x) end
        ax.add_patch(
            Rectangle(
                (x_lo - pad, body_y0),
                (x_hi - x_lo) + 2 * pad,
                body_h,
                fc="0.82",
                ec="0.35",
                lw=1.2,
                alpha=0.9,
                zorder=6,
            )
        )
        cab_w = 0.8 * u
        ax.add_patch(
            Rectangle(
                (x_hi + pad - cab_w, body_top),
                cab_w,
                0.5 * u,
                fc="0.82",
                ec="0.35",
                lw=1.2,
                alpha=0.9,
                zorder=6,
            )
        )
        # Wheels at every axle (a hub highlight for a touch of polish)
        for x in axle_x:
            ax.add_patch(
                Circle(
                    (x, rw),
                    rw,
                    fc="0.15",
                    ec="k",
                    lw=1.0,
                    zorder=7,
                    gid="vehicle_wheel",
                )
            )
            ax.add_patch(Circle((x, rw), 0.32 * rw, fc="0.7", ec="k", lw=0.6, zorder=8))

        # Downward load arrow for each on-deck axle (off-deck axles apply none)
        arr_top = body_top + 1.0 * u
        on_mask = (axle_x >= 0.0) & (axle_x <= L)
        for x in axle_x[on_mask]:
            ax.annotate(
                "",
                xy=(x, 0.03 * u),
                xytext=(x, arr_top),
                arrowprops=dict(arrowstyle="-|>", color="r", lw=1.6),
                zorder=9,
            )

        # Travel-direction arrow ahead of the leading axle
        ax.annotate(
            "",
            xy=(x_hi + pad + 1.6 * u, body_top + 0.25 * u),
            xytext=(x_hi + pad + 0.4 * u, body_top + 0.25 * u),
            arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.5),
        )

        # Fit the glyph and keep the deck aligned in x with the diagrams below
        ax.set_ylim(-1.9 * u, arr_top + 1.5 * u)
        ax.set_xlim(
            min(-0.06 * L, x_lo - 0.8 * u),
            max(1.06 * L, x_hi + pad + 2.4 * u),
        )
        ax.set_ylabel("")
        ax.set_yticks([])

        # Axle weights are summarised once in a caption (per-axle labels would
        # collide for closely-spaced axle groups); the arrows show positions.
        ax.text(
            0.012,
            0.98,
            "Vehicle: " + self._vehicle_spec(us),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="0.15",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
            zorder=10,
        )

    def _vehicle_spec(self, us=None) -> str:
        """A compact axle-group summary, e.g. ``3x150 + 2x120 + 40 kN``."""
        from .units import resolve

        us = resolve(us)
        w = self.veh.axw
        parts = []
        i = 0
        while i < len(w):
            j = i
            while j + 1 < len(w) and np.isclose(w[j + 1], w[i]):
                j += 1
            n = j - i + 1
            parts.append(f"{n}×{w[i]:g}" if n > 1 else f"{w[i]:g}")
            i = j + 1
        fu = f" {us.force}" if us.force else ""
        return " + ".join(parts) + f"{fu}  (ΣW = {self.veh.W:g}{fu})"

    def animate(
        self,
        step: float,
        save: Optional[str] = None,
        fps: int = 12,
        pos_start: Optional[float] = None,
        pos_end: Optional[float] = None,
        dpi: int = 110,
        units=None,
    ):
        """
        Animate the vehicle crossing the bridge.

        Steps the vehicle across the bridge and animates three stacked,
        x-aligned panels: the deck with the moving truck on top, then the
        instantaneous bending-moment and shear-force diagrams.  As the vehicle
        advances, the running envelope of each load effect is shaded behind the
        instantaneous diagram and grows to the full traverse envelope - so the
        envelope can be seen being *built up* position by position.

        All positions are analysed up front (this is the slow part); playback
        and saving then just redraw the stored results.

        Parameters
        ----------
        step : float
            Distance increment to move the vehicle between frames.
        save : str, optional
            If given, write the animation to this path.  A ``.gif`` is written
            with Pillow; any other extension (e.g. ``.mp4``) uses ffmpeg.
        fps : int
            Frames per second for playback / the saved file.
        pos_start, pos_end : float, optional
            First and last front-axle positions (defaults: ``0`` to
            ``beam length + vehicle length``, i.e. a full on-and-off crossing).
        dpi : int
            Resolution used when ``save`` is given.
        units : str or pycba.units.UnitSystem, optional
            Display unit system for the labels (see :func:`pycba.set_units`).

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation.  In a notebook, display it with
            ``HTML(anim.to_jshtml())`` (or ``anim.to_html5_video()``); the
            ``FuncAnimation`` must be kept referenced while it plays.
        """
        from matplotlib.animation import FuncAnimation
        from .units import resolve

        us = resolve(units)
        # Analyse every position once, then animate from the stored results.
        self.run_vehicle(step, pos_start=pos_start, pos_end=pos_end)
        positions = list(self.pos)
        x = self.vResults[0].results.x
        # Plot bending moment as -M (sagging below the axis), matching plot_static
        bmd = np.array([-r.results.M for r in self.vResults])
        sfd = np.array([r.results.V for r in self.vResults])

        # Running (cumulative) envelope up to each frame
        emax_b = np.maximum.accumulate(bmd, axis=0)
        emin_b = np.minimum.accumulate(bmd, axis=0)
        emax_s = np.maximum.accumulate(sfd, axis=0)
        emin_s = np.minimum.accumulate(sfd, axis=0)

        L = self.ba.beam.length

        def _lims(lo_arr, hi_arr):
            lo, hi = float(lo_arr.min()), float(hi_arr.max())
            span = hi - lo if hi > lo else (abs(hi) + 1.0)
            return lo - 0.08 * span, hi + 0.08 * span

        bmd_lo, bmd_hi = _lims(emin_b[-1], emax_b[-1])
        sfd_lo, sfd_hi = _lims(emin_s[-1], emax_s[-1])
        deck_xlim = (-0.06 * L, 1.06 * L)

        fig, axs = plt.subplots(
            3,
            1,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
        )

        def _draw_effect(ax, y, emin, emax, ylo, yhi, ylabel, xlabel=False):
            ax.plot([0, L], [0, 0], "k", lw=2)
            ax.fill_between(x, emin, emax, color="0.82", alpha=0.8, lw=0)
            ax.plot(x, emax, color="0.55", lw=0.7)
            ax.plot(x, emin, color="0.55", lw=0.7)
            ax.plot(x, y, "r", lw=1.6)
            ax.set_xlim(*deck_xlim)
            ax.set_ylim(ylo, yhi)
            ax.grid(True)
            ax.set_ylabel(ylabel)
            if xlabel:
                ax.set_xlabel(us.distance_axis)

        def update(i):
            for ax in axs:
                ax.cla()
            # Deck + moving truck (fixed deck view; the truck clips in/out)
            self._draw_deck_and_vehicle(axs[0], positions[i], us)
            axs[0].set_xlim(*deck_xlim)
            lu = f" {us.length}" if us.length else ""
            axs[0].set_title(f"Front axle at x = {positions[i]:.1f}{lu}", fontsize=10)
            _draw_effect(
                axs[1],
                bmd[i],
                emin_b[i],
                emax_b[i],
                bmd_lo,
                bmd_hi,
                us.moment_axis,
            )
            _draw_effect(
                axs[2],
                sfd[i],
                emin_s[i],
                emax_s[i],
                sfd_lo,
                sfd_hi,
                us.shear_axis,
                xlabel=True,
            )
            return axs

        anim = FuncAnimation(
            fig, update, frames=len(positions), interval=1000.0 / fps, blit=False
        )

        if save is not None:
            writer = "pillow" if str(save).lower().endswith(".gif") else "ffmpeg"
            anim.save(str(save), writer=writer, fps=fps, dpi=dpi)

        return anim

    def plot_envelopes(self, env: Envelopes, units=None):
        """
        Plots the envelopes of load effects from a vehicle traverse analysis

        Parameters
        ----------
        env : Envelopes
            An `pycba.Envelopes` object containing the results of a moving load
            analysis.
        units : str or pycba.units.UnitSystem, optional
            Display unit system for the labels (see :func:`pycba.set_units`).

        Returns
        -------
        None
        """
        from .units import resolve

        us = resolve(units)
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
        ax.plot(x, env.Mmax, "r")
        ax.plot(x, env.Mmin, "b")
        ax.invert_yaxis()
        ax.grid()
        ax.set_ylabel(us.moment_axis)

        ax = axsLeft[1]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(x, env.Vmax, "r")
        ax.plot(x, env.Vmin, "b")
        ax.grid()
        ax.set_ylabel(us.shear_axis)
        ax.set_xlabel(us.distance_axis)

        # Reactions in right panel
        subfigs[1].suptitle("Support Reactions")

        # Reactions may be force or moment; show both unit labels.
        react_u = f" ({us.force}/{us.moment})" if us.force and us.moment else ""
        # Check if consistent envelope
        if len(self.pos) == env.Rmax.shape[1]:
            axsRight = np.atleast_1d(subfigs[1].subplots(nreactions, 1, sharex=True))
            pos = np.asarray(self.pos)
            for i, ax in enumerate(axsRight):
                ax.plot([0, L], [0, 0], "k", lw=2)
                ax.plot(pos, env.Rmax[i, :], "r")
                ax.plot(pos, env.Rmin[i, :], "b")
                # Mark and label the governing (extreme) reaction values so the
                # critical magnitude is readable directly off the plot.
                self._mark_reaction_extreme(ax, pos, env.Rmax[i, :], "r", np.argmax)
                self._mark_reaction_extreme(ax, pos, env.Rmin[i, :], "b", np.argmin)
                ax.grid()
                ax.set_ylabel(f"Reaction {i+1}{react_u}")
            axsRight[-1].set_xlabel(us.length_axis("Position of Front Axle"))

        else:  # Otherwise envelope of envelopes
            axsRight = subfigs[1].subplots(2, 1, sharex=True)
            for i, (ax, le, col) in enumerate(
                zip(axsRight, ["max", "min"], ["r", "b"])
            ):
                r = eval(f"env.R{le}val")  # kinda yuk!
                bars = ax.bar(np.arange(env.nsup), r, color=col)
                # Label each bar with its governing reaction value.
                ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=2)
                ax.set_xticks(np.arange(env.nsup))
                ax.set_xticklabels([f"R{i+1}" for i in range(env.nsup)])
                ax.set_ylabel(f"Reactions [{le}]{react_u}")
                ax.grid()
            axsRight[1].set_xlabel("Reaction ID")

    @staticmethod
    def _mark_reaction_extreme(ax, pos, series, color, arg_fn):
        """
        Mark and annotate the extreme of a reaction time-history on ``ax``.

        ``arg_fn`` is :func:`numpy.argmax` (governing maximum, ``r`` line) or
        :func:`numpy.argmin` (governing minimum, ``b`` line); a marker is drawn
        at the extreme and labelled with its value.
        """
        series = np.asarray(series)
        if series.size == 0:
            return
        idx = int(arg_fn(series))
        xv, yv = pos[idx], series[idx]
        ax.plot([xv], [yv], color=color, marker="o", ms=5, zorder=5)
        above = arg_fn is np.argmax
        ax.annotate(
            f"{yv:.3g}",
            (xv, yv),
            textcoords="offset points",
            xytext=(0, 5 if above else -5),
            ha="center",
            va="bottom" if above else "top",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    def plot_ratios(self, env_ratios: Dict[str, np.ndarray], units=None):
        """
        Plots the output of :meth:`pycba.bridge.BridgeAnalysis.envelopes_ratios`.

        Parameters
        ----------
        env_ratios : Dict[str,np.ndarray]
            The dictionary of envelopes ratios.
        units : str or pycba.units.UnitSystem, optional
            Display unit system for the distance axis (the ratios themselves
            are dimensionless).  See :func:`pycba.set_units`.

        Raises
        ------
        ValueError
            Inconsistency in the dictionary entries.

        Returns
        -------
        None.
        """
        from .units import resolve

        us = resolve(units)
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
        ax.set_xlabel(us.distance_axis)

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
