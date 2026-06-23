"""
PyCBA - Shared type definitions
"""
from __future__ import annotations
import enum
from typing import List, Union, NamedTuple, Optional, Tuple
import numpy as np


LoadType = List[Union[int, float]]
LoadMatrix = List[LoadType]


class MemberType(enum.IntEnum):
    """
    Beam member (element) type, encoding the internal moment releases.

    The value is the integer ``eletype`` used throughout PyCBA, so a member
    type may be passed anywhere an ``eletype`` is accepted (it *is* an int).
    The two letters read left end then right end (``F`` = fixed/continuous,
    ``P`` = pinned/released):

    * ``FF`` (1) - fixed-fixed: moment continuous at both ends (the default).
    * ``FP`` (2) - fixed-pinned: moment released at the right end.
    * ``PF`` (3) - pinned-fixed: moment released at the left end.
    * ``PP`` (4) - pinned-pinned: moment released at both ends.

    At an internal hinge only one of the two members meeting at the node
    should carry the release.
    """

    FF = 1
    FP = 2
    PF = 3
    PP = 4

    @classmethod
    def coerce(cls, value: Union["MemberType", int, str]) -> int:
        """
        Normalise a member type to its integer ``eletype``.

        Accepts a :class:`MemberType`, an ``int`` (1-4), or a case-insensitive
        name string (``"FF"``, ``"FP"``, ``"PF"``, ``"PP"``).

        Raises
        ------
        ValueError
            If a string is not a recognised member-type name, or an int is
            outside ``1-4``.
        """
        if isinstance(value, str):
            try:
                return int(cls[value.strip().upper()])
            except KeyError:
                names = ", ".join(m.name for m in cls)
                raise ValueError(
                    f"Unknown member type {value!r}; use one of {names} (or 1-4)."
                )
        # MemberType, a Python/NumPy scalar, or a 1-element array (the default
        # eletype is np.ones((N, 1)), so each entry arrives as a length-1 row).
        arr = np.asarray(value)
        if arr.size != 1:
            raise ValueError(f"A single member type is required, got {arr.size} values.")
        iv = int(arr.reshape(-1)[0])
        if iv not in (1, 2, 3, 4):
            raise ValueError(f"eletype must be 1-4 (or a MemberType/name), got {iv}.")
        return iv


class LoadCNL(NamedTuple):
    Va: float
    Ma: float
    Vb: float
    Mb: float


class MemberResults:
    """
    Class for storing the results for a single member
    """

    def __init__(
        self,
        vals: Optional[
            Tuple[np.array, np.array, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None,
        n: Optional[int] = None,
    ):
        if vals is not None:
            (x, M, V, R, D) = vals
            self.n = len(x)
            self.x = x
            self.V = V
            self.M = M
            self.R = R
            self.D = D
        elif n is not None:
            self.n = n
            self._zero(n)
        else:
            raise ValueError("MemberResults requires either vals or n")

    def _zero(self, n: int):
        self.x = np.zeros(n)
        self.M = np.zeros(n)
        self.V = np.zeros(n)
        self.R = np.zeros(n)
        self.D = np.zeros(n)

    def __add__(self, o: MemberResults):
        np.testing.assert_equal(
            self.x, o.x, err_msg="Cannot superimpose results of different members"
        )

        x = self.x
        M = self.M + o.M
        V = self.V + o.V
        R = self.R + o.R
        D = self.D + o.D

        return MemberResults(vals=(x, M, V, R, D))

    def apply_EI(self, EI: float):
        self.R /= EI
        self.D /= EI
