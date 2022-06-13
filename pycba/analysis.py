"""
PyCBA - Continuous Beam Analysis

An OO Python adaptation of the CBA, originally written for Matlab here:
http://www.colincaprani.com/programming/matlab/
"""

from typing import Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from .beam import Beam, LoadMatrix
from .results import BeamResults


class BeamAnalysis:
    """
    The base class for Continuous Beam Analysis
    """

    def __init__(
        self,
        L: np.ndarray,
        EI: Union[float, np.ndarray],
        R: np.ndarray,
        LM: Optional[LoadMatrix] = None,
        eletype: Optional[np.ndarray] = None,
    ):
        """
        Constructs a beam analysis object given the structural information necessary.


        Parameters
        ----------
        L : np.ndarray
            A vector of span lengths.
        EI : Union[float, np.ndarray]
            A vector of member flexural rigidities.
        R : np.ndarray
            A vector describing the support conditions at each member end.
        LM : Optional[list[list[Union[int, float]]]]
            The load matrix: a list of loads on the beam; each load with several parameters.
        eletype : Optional[np.ndarray]
            A vector of the member types. Defaults to a fixed-fixed element.

        Returns
        -------
        None.

        """
        self.npts = 100
        self._beam_results = None

        if eletype is None:
            self.eletype = np.ones((len(L), 1))
        else:
            self.eletype = eletype
        # Create the beam
        self._beam = Beam(L=L, EI=EI, R=R, LM=LM, eletype=self.eletype)

        self._n = self._beam.no_spans
        self._no_nodes = self._n + 1
        self._nDOF = 2 * self._no_nodes

    @property
    def beam_results(self):
        return self._beam_results

    @property
    def beam(self):
        return self._beam

    def set_loads(self, LM: LoadMatrix):
        """
        Set load matrix for pre-defined beam

        Parameters
        ----------
        LM : List[List[Union[int, float]]]
            The load matrix for the beam.

        Returns
        -------
        None.

        """
        self._beam.loads = LM

    def analyze(self, npts: Optional[int] = None) -> int:
        """
        Conducts the analysis on the constructed BeamAnalysis object

        Parameters
        ----------
        npts : Optional[int]
            The number of evaluation points along a member for load effects.

        Returns
        -------
        0 for a succesful execution

        """
        if npts and npts > 3:
            self.npts = npts

        fU = self._forces()
        f = np.copy(fU)
        ksysU = self._assemble()
        ksys = np.copy(ksysU)
        ksys, f = self._apply_bc(ksys, f)
        d = self._solver(ksys, f)
        r = self._reactions(ksysU, d, fU)

        self._beam_results = BeamResults(self._beam, d, r, self.npts)
        return 0

    def _forces(self) -> np.ndarray:
        """
        Construct the nodal force vector

        Parameters
        ----------
        None

        Returns
        -------
        f : np.ndarray
            The global nodal force vector

        """
        f = np.zeros(self._nDOF)

        for i in range(self._n):
            dof_i = 2 * i
            fmbr = self._beam.get_cnl(i)
            # Cumulatively apply forces in opposite direction
            f[dof_i : dof_i + 4] -= fmbr
        return f

    def _assemble(self) -> np.ndarray:
        """
        Construct the unrestricted global stiffness matrix

        Parameters
        ----------
        None

        Returns
        -------
        ksys : np.ndarray
            The global stiffness matrix

        """
        ksys = np.zeros((self._nDOF, self._nDOF))

        for i in range(self._n):
            kb = self._beam.get_span_k(i)
            dof_i = 2 * i
            ksys[dof_i : dof_i + 4, dof_i : dof_i + 4] += kb
        return ksys

    def _apply_bc(self, k: np.ndarray, f: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Apply the boundary conditions

        Parameters
        ----------
        k : np.ndarray
            The unrestricted global stiffness matrix
        f : np.ndarray
            The global force vector

        Returns
        -------
        k : np.ndarray
            The restricted global stiffness matrix
        f : np.ndarray
            The force vector with boundary conditions imposed
        """
        r = self._beam.restraints
        for i in range(self._nDOF):
            # Negative means fully restrained
            if r[i] < 0:
                # Set off diagonals to zero
                for j in range(self._nDOF):
                    k[i][j] = 0
                    k[j][i] = 0
                # Set diagonal to 1 and the force to 0
                k[i][i] = 1
                f[i] = 0
            elif r[i] > 0:
                # Positive means spring support so add the stiffness
                k[i][i] += r[i]
        return k, f

    def _reactions(self, k: np.ndarray, d: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Calculate the reactions

        Parameters
        ----------
        k : np.ndarray
            The unrestricted global stiffness matrix
        d : np.ndarray
            The global nodal displacement vector
        f : np.ndarray
            The global nodal force vector

        Returns
        -------
        r : np.ndarray
            The reactions corresponding to full restraints
        """
        r = k @ d
        r -= f
        r2 = []

        # Report reactions corresponding to full restraints
        for i in range(self._nDOF):
            if self._beam.restraints[i] < 0:
                r2.append(r[i])
        return np.array(r2)

    def _solver(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solves the matrix equation

        Parameters
        ----------
        A : np.ndarray
            The restricted global stiffness matrix
        b : np.ndarray
            The restricted force vector

        Returns
        -------
        x : np.ndarray
            The nodal displacements
        """
        x = np.linalg.solve(A, b)
        return x

    def plot_results(self):
        """
        Plots the results of the analysis

        Returns
        -------
        None.

        """

        if self._beam_results is None:
            print("Nothing to plot - run analysis first")
            return
        res = self._beam_results.results
        L = self._beam.length

        fig, axs = plt.subplots(3, 1)

        ax = axs[0]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, -res.M, "r")
        ax.grid()
        ax.set_ylabel("Bending Moment (kNm)")

        ax = axs[1]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, res.V, "r")
        ax.grid()
        ax.set_ylabel("Shear Force (kN)")

        ax = axs[2]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(res.x, res.D * 1e3, "r")
        ax.grid()
        ax.set_ylabel("Deflection (mm)")
        ax.set_xlabel("Distance along beam (m)")

        plt.show()
