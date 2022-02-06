"""
PyCBA - Beam Class definition
"""
from typing import Optional
import numpy as np
from .load import parse_LM, LoadMatrix, LoadCNL


class Beam:
    """
    Class definition
    """

    def __init__(
        self,
        L: Optional[np.ndarray] = None,
        EI: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        LM: Optional[LoadMatrix] = None,
        eletype: Optional[np.ndarray] = None,
    ):
        """
        Constructs a beam object

        Parameters
        ----------
        L : np.ndarray
            A vector of span lengths.
        EI : np.ndarray
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
        None.
        """
        self._no_spans = 0
        self._no_restraints = 0
        self._length = 0
        self.mbr_lengths = []
        self.mbr_EIs = []
        self.mbr_eletype = []
        self._restraints = []
        self._loads = []
        self._terminal_coords = [0.0]

        if L is not None and eletype is not None:
            # scalar EI - same for all spans
            if isinstance(EI, float):
                for l, et in zip(L, eletype):
                    self.add_span(l, EI, et)
            else:
                if len(L) == len(EI):
                    for l, ei, et in zip(L, EI, eletype):
                        self.add_span(l, ei, et)
                else:
                    raise ValueError("Define EI for each span")
            if len(R) == 2 * len(L) + 2:
                self._restraints = R
            else:
                raise ValueError("Insufficient restraints defined")
        if LM is not None:
            self._set_loads(LM)

    def add_span(self, L: float, EI: float, eletype: int):
        """
        Add a span to the continuous beam

        Parameters
        ----------
        L : float
            The length of the member.
        EI : np.ndarray
            The flexural rigidity of the member.
        eletype : int
            The element type for the member

        Returns
        -------
        None.

        """
        self.mbr_lengths.append(L)
        self.mbr_EIs.append(EI)
        self.mbr_eletype.append(eletype)
        self._no_spans = len(self.mbr_lengths)
        self._length += L
        self._terminal_coords.append(self._terminal_coords[-1] + L)

    @property
    def loads(self) -> LoadMatrix:
        """
        Returns the load matrix for the beam

        Returns
        -------
        LM : LoadMatrix
            The load matrix for the beam

        """
        return self._loads

    @loads.setter
    def loads(self, LM):
        """
        Sets the load matrix for the beam

        Parameters
        -------
        LM : LoadMatrix
            The load matrix for the beam

        Returns
        -------
        None

        """
        self._set_loads(LM)

    def _set_loads(self, LM):
        """
        Explicit internal setter for loads

        Parameters
        -------
        LM : LoadMatrix
            The load matrix for the beam

        Returns
        -------
        None

        """
        self._loads = parse_LM(LM)
        self.no_loads = len(LM)

    @property
    def restraints(self) -> np.ndarray:
        """
        Returns the restraints vector for the beam

        Returns
        -------
        _restraints : np.ndarray
            The restraints vector for the beam

        """
        return self._restraints

    @restraints.setter
    def restraints(self, r):
        """
        Stores support conditions

        Parameters
        -------
        r : np.ndarray
            The restraint vector for the beam

        Returns
        -------
        None

        """
        self._restraints = r
        pass

    def _set_element_type(self, i_span):
        """
        Stores element type for a span based on support conditions
        """
        raise NotImplementedError("Changing element type not supported")

    @property
    def no_spans(self):
        """
        Returns the no. of spans in the beam

        Returns
        -------
        no_spans : int
            The number of spans in the beam
        """
        return self._no_spans

    @property
    def no_restraints(self):
        """
        Returns the number of restraints of the beam

        Returns
        -------
        no_restraints : int
            The number of restraints in the beam
        """
        return len(self._restraints)

    @property
    def length(self):
        """
        Returns
        -------
        length : float
            The total length of the beam
        """
        return self._length

    def get_local_span_coords(self, pos: float) -> (int, float):
        """
        Returns the span index and position in span for a position given in global
        coordinates on the beam

        Parameters
        ----------
        pos : float
            The position of interest in global coordinates along the length of the beam

        Returns
        -------
        ispan : int
            The index (1-based) of the span in which the point of interest falls
        pos_in_span : float
            The local coordinate along the member of the point of interest

        """
        if pos < 0.0 or pos > self.length:
            return -1, 0
        """
        # more basic algorithim
        ispan = 0
        pos_in_span = pos
        while ispan < self.no_spans and pos_in_span > self.mbr_lengths[ispan]:
            pos_in_span -= self.mbr_lengths[ispan]
            ispan += 1
            if ispan > self.no_spans:
                ispan = -1
        """
        try:
            ispan = next(i - 1 for i, x in enumerate(self._terminal_coords) if x > pos)
        except StopIteration:  # at the end of the beam
            ispan = self._no_spans - 1
        pos_in_span = pos - self._terminal_coords[ispan]

        return ispan, pos_in_span

    def get_cnl(self, i_span: int) -> LoadCNL:
        """
        Returns Consistent Nodal Loads for the member

        Parameters
        ----------
        ispan : int
            The index (1-based) of the span in which the point of interest falls

        Returns
        -------
        cnl : LoadCNL
            The totalled CNL object for the member, considering all loads.

        """
        cnl = np.zeros(4)
        L = self.mbr_lengths[i_span]
        eType = self.mbr_eletype[i_span]

        for load in self._loads:
            if load.i_span == i_span:
                cnl += load.get_cnl(L, eType)
        return cnl

    def get_span_k(self, i_span: int) -> np.ndarray:
        """
        Returns the stiffness matrix for the ith span

        Parameters
        ----------
        ispan : int
            The index (1-based) of the span in which the point of interest falls

        Returns
        -------
        kb : np.ndarray
            The stiffness matrix for the member

        """
        EI = self.mbr_EIs[i_span]
        L = self.mbr_lengths[i_span]
        eType = self.mbr_eletype[i_span]
        if eType == 2:
            kb = self.k_FP(EI, L)
        elif eType == 3:
            kb = self.k_PF(EI, L)
        elif eType == 4:
            kb = self.k_PP(EI, L)
        else:
            kb = self.k_FF(EI, L)
        return kb

    def k_FF(self, EI: float, L: float) -> np.ndarray:
        """
        Stiffness matrix for a fixed-fixed element

        Parameters
        ----------
        EI : float
            The flexural rigidity for the member (assumed prismatic)
        L : float
            The length of the member

        Returns
        -------
        k : np.ndarray
            The stiffness matrix for the member
        """
        L2 = L**2
        L3 = L**3

        kfv = 12 * EI / L3
        kmv = 6 * EI / L2
        kft = kmv
        kmt = 4 * EI / L
        kmth = 2 * EI / L

        k = np.array(
            [
                [kfv, kft, -kfv, kft],
                [kmv, kmt, -kmv, kmth],
                [-kfv, -kft, kfv, -kft],
                [kft, kmth, -kft, kmt],
            ]
        )

        return k

    def k_FP(self, EI: float, L: float) -> np.ndarray:
        """
        Stiffness matrix for a fixed-pinned element

        Parameters
        ----------
        EI : float
            The flexural rigidity for the member (assumed prismatic)
        L : float
            The length of the member

        Returns
        -------
        k : np.ndarray
            The stiffness matrix for the member
        """
        L2 = L**2
        L3 = L**3

        kfv = 3 * EI / L3
        kmv = 3 * EI / L2
        kft = kmv
        kmt = 3 * EI / L

        k = np.array(
            [
                [kfv, kft, -kfv, 0],
                [kmv, kmt, -kmv, 0],
                [-kfv, -kft, kfv, 0],
                [0, 0, 0, 0],
            ]
        )

        return k

    def k_PF(self, EI: float, L: float) -> np.ndarray:
        """
        Stiffness matrix for a pinned-fixed element

        Parameters
        ----------
        EI : float
            The flexural rigidity for the member (assumed prismatic)
        L : float
            The length of the member

        Returns
        -------
        k : np.ndarray
            The stiffness matrix for the member
        """
        L2 = L**2
        L3 = L**3

        kfv = 3 * EI / L3
        kmv = 3 * EI / L2
        kft = kmv
        kmt = 3 * EI / L

        k = np.array(
            [
                [kfv, 0, -kfv, kft],
                [0, 0, 0, 0],
                [-kfv, 0, kfv, -kft],
                [kft, 0, -kft, kmt],
            ]
        )

        return k

    def k_PP(self, EI: float, L: float) -> np.ndarray:
        """
        Stiffness matrix for a pinned-pinned element

        Parameters
        ----------
        EI : float
            The flexural rigidity for the member (assumed prismatic)
        L : float
            The length of the member

        Returns
        -------
        k : np.ndarray
            The stiffness matrix for the member
        """

        k = np.zeros((4, 4))

        return k
