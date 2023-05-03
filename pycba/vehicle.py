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

    @classmethod
    def from_convoy(cls, vehicles: List[Vehicle], vehicle_spacings: np.ndarray):
        """
        Alternative constructor for :class:`pycba.bridge.Vehicle` object
        as multiple :class:`pycba.bridge.Vehicle` objects
        behind one another (eg. superload, queued vehicles, train)

        Parameters
        ----------

        vehicles : List[Vehicles]
            A list of :class:`pycba.bridge.Vehicle` objects,
            length one greater than the length of the
            vehicle spacings vector.
        vehicle_spacings : np.ndarray
            A vector of spacings between vehicles of length one
            fewer than the length of the
            list of vehicles.

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

        if len(vehicles) - 1 != len(vehicle_spacings):
            raise ValueError("Inconsistent vehicle and spacing counts")

        if not all(isinstance(v, Vehicle) for v in vehicles):
            raise ValueError("List must contain only Vehicle objects")

        # pre-allocate axle weights and spacings
        new_vehicle_axles = np.array([])
        new_vehicle_spaces = np.array([])

        # loop through each vehicle
        for veh in vehicles:
            new_vehicle_axles = np.append(new_vehicle_axles, veh.axw)
            new_vehicle_spaces = np.append(new_vehicle_spaces, np.insert(veh.axs, 0, 0))

        # replace 0 spacing (first axle), with vehicle spacings
        new_vehicle_spaces[new_vehicle_spaces == 0] = np.insert(vehicle_spacings, 0, 0)

        return cls(new_vehicle_spaces[1:], new_vehicle_axles)


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
