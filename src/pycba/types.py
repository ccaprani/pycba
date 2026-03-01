"""
PyCBA - Shared type definitions
"""
from __future__ import annotations
from typing import List, Union, NamedTuple, Optional, Tuple
import numpy as np


LoadType = List[Union[int, float]]
LoadMatrix = List[LoadType]


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
