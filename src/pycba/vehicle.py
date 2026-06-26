"""
PyCBA - Continuous Beam Analysis - Bridge Crossing Module
"""
from __future__ import annotations  # https://bit.ly/3KYiL2o
from typing import Optional, Union, Dict, List
import numpy as np
import matplotlib.pyplot as plt


class Vehicle:
    """
    A basic vehicle definition for a bridge crossing analysis
    """

    def __init__(self, axle_spacings: np.ndarray, axle_weights: np.ndarray):
        """
        Constructs the :class:`pycba.bridge.Vehicle` object from the supplied data.

        Parameters such as vehicle length, weight, and axle coordinates are
        calculated from the supplied data.

        Parameters
        ----------
        axle_spacings : np.ndarray
            A vector of axle spacings of length one fewer than the length of the
            vector of axle weights
        axle_weights : np.ndarray
            A vector of axle weights, length one greater than the length of the
            axle spacings vector.

        Raises
        ------
        ValueError
            If the lengths of the vectors of axle weights and spacings are
            inconsistent.

        Returns
        -------
        None.

        """
        self.axs = np.asarray(axle_spacings)
        self.axw = np.asarray(axle_weights)

        if len(self.axs) + 1 != len(self.axw):
            raise ValueError("Inconsistent axle spacing and weight counts")

        self.L = sum(axle_spacings)
        self.W = sum(axle_weights)
        self.NoAxles = len(axle_weights)
        self.axle_coords = np.zeros(self.NoAxles)
        for i, s in enumerate(self.axs):
            self.axle_coords[i + 1] = self.axle_coords[i] + s

    def reverse(self, in_place=True):
        """
        Reverses the vehicle; now the `pos` coordinate will refer to the rear axle
        as it traverses the bridge in reverse from zero in the global x-coordinate
        system.

        Parameters
        ----------

        in_place : Bool
            Whether or not to reverse the current vehicle (`True` - default),
            or return a new copy of this vehicle reversed (`False`).

        Returns
        -------
        None or Vehicle.

        """
        if in_place:
            self.axle_coords = +self.L - self.axle_coords
        else:
            return Vehicle(np.copy(self.axs[::-1]), np.copy(self.axw[::-1]))


def make_train(vehicles: List[Vehicle], spacings: np.ndarray):
    """
    Makes a train of vehicles from a sequence from multiple
    :class:`pycba.bridge.Vehicle` objects behind one another (e.g. superload
    queued vehicles, train).

    Parameters
    ----------

    vehicles : List[Vehicles]
        A list of :class:`pycba.bridge.Vehicle` objects, length one greater
        than the length of the vehicle spacings vector.
    spacings : np.ndarray
        A vector of spacings between vehicles, either of length one, for equal
        spacings between all vehicles, or of of length one fewer than the
        length of the list of vehicles.

    Raises
    ------
    ValueError
        If the lengths of the list of vehicles and
        vector of spacings are inconsistent.
    ValueError
        If all list entries are not
        :class:`pycba.bridge.Vehicle` objects

    Returns
    -------
    :class:`pycba.bridge.Vehicle` object

    """

    if len(vehicles) - 1 != len(spacings):
        raise ValueError("Inconsistent vehicle and spacing counts")

    if not all(isinstance(v, Vehicle) for v in vehicles):
        raise ValueError("List must contain only Vehicle objects")

    new_vehicle_axles = np.array([])
    new_vehicle_spaces = np.array([])

    # loop through each vehicle
    for veh in vehicles:
        new_vehicle_axles = np.append(new_vehicle_axles, veh.axw)
        new_vehicle_spaces = np.append(new_vehicle_spaces, np.insert(veh.axs, 0, 0))

    new_vehicle_spaces[new_vehicle_spaces == 0] = np.insert(spacings, 0, 0)

    return Vehicle(new_vehicle_spaces[1:], new_vehicle_axles)


class VehicleLibrary:
    """
    A repository of some useful vehicles for analysis
    """

    abag_bdouble_aw = (
        np.array(
            [
                6,
                17 / 2,
                17 / 2,
                22.5 / 3,
                22.5 / 3,
                22.5 / 3,
                22.5 / 3,
                22.5 / 3,
                22.5 / 3,
            ]
        )
        * 9.81
    )  # t to kN

    abag_bdouble_as = [
        [3.0, 1.2, 5.5, 1.2, 1.2, 6.5, 1.2, 1.2],
        [3.0, 1.2, 6.0, 1.2, 1.2, 6.0, 1.2, 1.2],
        [3.0, 1.2, 6.5, 1.2, 1.2, 5.5, 1.2, 1.2],
    ]

    abag_semitrailer_aw = (
        np.array([6, 17 / 2, 17 / 2, 22.5 / 2, 22.5 / 2, 22.5 / 3]) * 9.81
    )  # t to kN

    abag_semitrailer_as = [
        [3.0, 1.2, 4.4, 1.2, 1.2],
        [3.0, 1.2, 6.4, 1.2, 1.2],
        [3.0, 1.2, 8.4, 1.2, 1.2],
        [3.0, 1.2, 10.4, 1.2, 1.2],
    ]

    @classmethod
    def get_abag_bdouble(cls, iax: int) -> Vehicle:
        """
        Creates one of the Australian Bridge Assessment Guidelines (ABAG) 68.5 t
        B-double trucks.

        Parameters
        ----------
        iax : int
            The 0-based index of the variable axle spacing required:

                - **0**: Variable spacings are 5.5 and 6.5 from front
                - **1**: Variable spacings are 6.0 and 6.0 from front
                - **2**: Variable spacings are 6.5 and 5.5 from front

        Returns
        -------
        Vehicle
            The :class:`pycba.bridge.Vehicle` object.

        """
        return Vehicle(cls.abag_bdouble_as[iax], cls.abag_bdouble_aw)

    @classmethod
    def get_abag_semitrailer(cls, iax: int) -> Vehicle:
        """
        Creates one of the Australian Bridge Assessment Guidelines (ABAG) 45 t
        semi-trailer trucks

        Parameters
        ----------
        iax : int
            The 0-based index of the variable axle spacing required:

                - **0**: Variable spacing is 4.4 m
                - **1**: Variable spacing is 6.4 m
                - **2**: Variable spacing is 8.4 m
                - **3**: Variable spacing is 10.4 m

        Returns
        -------
        Vehicle
            The :class:`pycba.bridge.Vehicle` object.
        """
        return Vehicle(cls.abag_semitrailer_as[iax], cls.abag_semitrailer_aw)

    @staticmethod
    def get_example_permit() -> Vehicle:
        """
        An example B-double type truck that might seek a network access permit.

        Parameters
        ----------
        None

        Returns
        -------
        Vehicle
            The :class:`pycba.bridge.Vehicle` object.
        """
        axs = np.array([1.793, 2.507, 1.370, 5.403, 1.232, 1.232, 6.303, 1.232, 1.232])
        axw = np.array([5.5, 5.5, 8.5, 8.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5]) * 9.81
        return Vehicle(axs, axw)

    @staticmethod
    def get_m1600(middle_spacing: float) -> Vehicle:
        """
        The AS5100 M1600 load model truck. This is the notional load model for
        moving traffic and is usually critical for short to medium span bridges
        once the dyanmic load allowance is included.

        Parameters
        ----------
        middle_spacing : float
            The spacing for the variable middle axle spacing; min. 6.25 m.

        Raises
        ------
        ValueError
            If the supplied spacing is less than the code-specified minimum
            of 6.25 m.

        Returns
        -------
        Vehicle
            The :class:`pycba.bridge.Vehicle` object.
        """
        if middle_spacing < 6.25:
            raise ValueError("Min. M1600 middle spacing is 6.25 m")
        axs = [
            1.25,
            1.25,
            3.75,
            1.25,
            1.25,
            middle_spacing,
            1.25,
            1.25,
            5.0,
            1.25,
            1.25,
        ]
        axw = 120 * np.ones(12)
        return Vehicle(axs, axw)

    @staticmethod
    def get_s1600(middle_spacing: float) -> Vehicle:
        """
        The AS5100 S1600 load model truck. This is the notional load model for
        stationary traffic and is criticial for longer spans.

        Parameters
        ----------
        middle_spacing : float
            The spacing for the variable middle axle spacing; min. 6.25 m.

        Raises
        ------
        ValueError
            If the supplied spacing is less than the code-specified minimum
            of 6.25 m.

        Returns
        -------
        Vehicle
            The :class:`pycba.bridge.Vehicle` object.
        """
        if middle_spacing < 6.25:
            raise ValueError("Min. S1600 middle spacing is 6.25 m")
        axs = [
            1.25,
            1.25,
            3.75,
            1.25,
            1.25,
            middle_spacing,
            1.25,
            1.25,
            5.0,
            1.25,
            1.25,
        ]
        axw = 80 * np.ones(12)
        return Vehicle(axs, axw)

    @staticmethod
    def get_validation_truck() -> Vehicle:
        """
        A set of moving loads used for validation on a 20 m span against the
        textbook, "Structural and Stress Analysis"", 2nd edn., Megson, p. 579.

        Parameters
        ----------
        None

        Returns
        -------
        Vehicle
            The :class:`pycba.bridge.Vehicle` object.
        """
        return Vehicle(np.array([4, 4]), np.array([5, 4, 3]))

    @staticmethod
    def get_la_rail(axle_group_count=10, axle_group_spacing=12, axle_weight=300):
        """
        The AS5100.2 LA rail load model, including the simulated locomotive.

        Parameters
        ----------
        axle_group_count : int
            The number of axle groups. This should be sufficient to cover the
            adverse portion of the influence line for the component being
            examined.

        axle_weight : float
            The nominal axle weight. Default is 300 kN, but 150LA and others are
            also used.

        axle_group_spacing : float
            The spacing between each axle group, min. 12 m, max. 20 m. Note
            that this is not a gap but the centre-of-group to centre-of-group
            distance.

        Raises
        ------
        ValueError
            If the supplied axle group spacing is outside the code-specified
            limits of 12 to 20 m, or the number of axle groups is < 1.

        Returns
        -------
        Vehicle
            The :class:`pycba.bridge.Vehicle` object.
        """

        if axle_group_spacing < 12.0 or axle_group_spacing > 20.0:
            raise ValueError("Axle group spacing should be between 12 and 20 m")
        if axle_group_count < 1:
            raise ValueError("Need at least one axle group")

        axs = [2.0]  # simulated locomotive
        axw = [360.0]
        single_ags = [1.7, 1.1, 1.7]
        single_agw = [300.0, 300.0, 300.0, 300.0]
        group_length = sum(single_ags)

        # first group
        axs += single_ags
        axw += single_agw

        # additional groups
        for _ in range(2, axle_group_count + 1):
            axs += [axle_group_spacing - group_length] + single_ags
            axw += single_agw

        # Now scale the weights
        axw = (axle_weight / 300.0) * np.array(axw)
        return Vehicle(np.array(axs), axw)

    # ------------------------------------------------------------------ #
    # International code load models (axle-based).  The accompanying lane
    # UDL, where a code pairs the truck with a distributed load, is given in
    # each docstring for use as ``BridgeAnalysis.run_load_model(..., w_lane=)``.
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_hl93_truck(rear_spacing: float = 4.3) -> Vehicle:
        """
        AASHTO LRFD HL-93 design truck (Art. 3.6.1.2.2): three axles of 35, 145
        and 145 kN.  The front-to-middle spacing is fixed at 4.3 m; the rear
        (middle-to-back) spacing is varied between 4.3 m and 9.0 m to maximise
        the effect.  HL-93 is the truck *or* the design tandem (whichever
        governs) combined with the design lane load (use ``w_lane=9.3`` kN/m);
        the 33% dynamic load allowance applies to the truck only.

        Parameters
        ----------
        rear_spacing : float
            The variable middle-to-rear axle spacing, 4.3-9.0 m (default 4.3).
        """
        if not (4.3 <= rear_spacing <= 9.0):
            raise ValueError("HL-93 rear axle spacing must be between 4.3 and 9.0 m")
        return Vehicle(np.array([4.3, rear_spacing]), np.array([35.0, 145.0, 145.0]))

    @staticmethod
    def get_hl93_tandem() -> Vehicle:
        """
        AASHTO LRFD HL-93 design tandem (Art. 3.6.1.2.3): two 110 kN axles at
        1.2 m.  Combined with the design lane load (``w_lane=9.3`` kN/m); the
        heavier of the truck and tandem governs.
        """
        return Vehicle(np.array([1.2]), np.array([110.0, 110.0]))

    @staticmethod
    def get_lm1(alpha_Q: float = 1.0) -> Vehicle:
        """
        Eurocode EN 1991-2 Load Model 1 tandem system (TS), Lane 1 (Table 4.2):
        two 300 kN axles at 1.2 m.  Accompanied by the Lane-1 UDL of 9 kN/m²,
        i.e. ``w_lane=27`` kN/m over the 3 m notional lane.

        Parameters
        ----------
        alpha_Q : float
            National-Annex adjustment factor on the axle loads (default 1.0).
        """
        Q = 300.0 * alpha_Q
        return Vehicle(np.array([1.2]), np.array([Q, Q]))

    @staticmethod
    def get_lm71(alpha: float = 1.0) -> Vehicle:
        """
        Eurocode EN 1991-2 rail Load Model 71 (cl 6.3.2): four 250 kN axles at
        1.6 m centres, with two 80 kN/m distributed loads acting on each side
        starting 0.8 m beyond the outer axles (use ``w_lane=80`` for the
        distributed component).

        Parameters
        ----------
        alpha : float
            Classification factor (0.75-1.46; default 1.0 for international lines).
        """
        Q = 250.0 * alpha
        return Vehicle(np.array([1.6, 1.6, 1.6]), np.full(4, Q))

    @staticmethod
    def get_cl625() -> Vehicle:
        """
        CSA S6 CL-625 design truck (cl 3.8.3.1.2): five axles of 50, 125, 125,
        175 and 150 kN (625 kN total) at spacings 3.6, 1.2, 6.6 and 6.6 m.  The
        CL-625 lane load is 80% of these axle loads superimposed on a
        ``w_lane=9`` kN/m UDL.
        """
        return Vehicle(
            np.array([3.6, 1.2, 6.6, 6.6]),
            np.array([50.0, 125.0, 125.0, 175.0, 150.0]),
        )

    @staticmethod
    def get_hb(units: float = 45.0, inner_spacing: float = 6.0) -> Vehicle:
        """
        BS 5400-2 / CS 454 HB abnormal vehicle (cl 6.3): a four-axle bogie, each
        axle ``units × 10`` kN, at spacings 1.8, ``inner_spacing`` and 1.8 m.

        Parameters
        ----------
        units : float
            Number of HB units, typically 25-45 (250-450 kN/axle). Default 45.
        inner_spacing : float
            The variable inner spacing; the code uses 6, 11, 16, 21 or 26 m for
            the worst effect. Default 6 m.
        """
        P = units * 10.0
        return Vehicle(np.array([1.8, inner_spacing, 1.8]), np.full(4, P))

    @staticmethod
    def get_jtg_vehicle() -> Vehicle:
        """
        China JTG D60-2015 standard vehicle load (车辆荷载, cl 4.3.1): five axles
        of 30, 120, 120, 140 and 140 kN (550 kN total) at spacings 3.0, 1.4, 7.0
        and 1.4 m.  The companion lane load (车道荷载) is a span-dependent UDL
        (10.5 kN/m for Highway Class I) plus a concentrated load.
        """
        return Vehicle(
            np.array([3.0, 1.4, 7.0, 1.4]),
            np.array([30.0, 120.0, 120.0, 140.0, 140.0]),
        )

    @staticmethod
    def get_a160() -> Vehicle:
        """AS 5100.2 A160 individual axle load (cl 7.2): a single 160 kN axle."""
        return Vehicle(np.array([]), np.array([160.0]))

    @staticmethod
    def get_w80() -> Vehicle:
        """AS 5100.2 W80 individual wheel load (cl 7.2): a single 80 kN wheel."""
        return Vehicle(np.array([]), np.array([80.0]))
