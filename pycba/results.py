"""
PyCBA - Beam Results module
"""

from __future__ import annotations  # https://bit.ly/3KYiL2o
from typing import List, Tuple
import numpy as np
from scipy import integrate
from .beam import Beam
from .load import MemberResults, LoadMaMb
from copy import deepcopy


class BeamResults:
    """
    BeamResults Class for processing and containing the results for each member
    """

    def __init__(self, beam: Beam, d: np.ndarray, r: np.ndarray, npts: int = 100):
        """
        Initialize member results from global results

        Parameters
        ----------
        beam : Beam
            The :class:`pycba.beam.Beam` object for which the results are to be stored.
        d : np.ndarray
            The vector of nodal displacements (from the stiffness method).
        r : np.ndarray
            The vector of reactions for the member degrees of freedom (if any).
        npts : int, optional
            The number of points along the member at which to calculate the load
            effects. The default is 100.

        Returns
        -------
        None.
        """
        self.npts = npts
        self.vRes = self._member_analysis(beam, d)
        self.D = d  # nodal displacements
        self.R = r  # reactions
        self.results = self._concatenate_results()

    def _concatenate_results(self):
        """
        Assemble the vector of results by member into a long vector of each result
        and store in a :class:`pycba.MemberResults` object.

        Parameters
        ----------
        None

        Returns
        -------
        MemberResults
            Stores all results along the whole beam as if it were a notional
            member.

        """
        x = []  # global coordinate along member
        M = []  # bending moment
        V = []  # shear force
        R = []  # rotation
        D = []  # deflection

        for res in self.vRes:
            x.append(res.x)
            M.append(res.M)
            V.append(res.V)
            R.append(res.R)
            D.append(res.D)
        x = np.concatenate(x)
        M = np.concatenate(M)
        V = np.concatenate(V)
        R = np.concatenate(R)
        D = np.concatenate(D)

        return MemberResults(vals=(x, M, V, R, D))

    def _member_analysis(self, beam: Beam, d: np.ndarray) -> List[MemberResults]:
        """
        Establish the results for each member from the stiffness method results.

        Parameters
        ----------
        beam : Beam
            The :class:`pycba.beam.Beam` object for which the results are required.
        d : np.ndarray
            The vector of nodal displacements from the stiffness analysis.

        Returns
        -------
        List[MemberResults]
            A list of the :class:`pycba.MemberResults` objects for each member.
        """

        vRes = []
        sumL = 0
        for i in range(beam.no_spans):
            kb = beam.get_span_k(i)
            dof_i = 2 * i
            dmbr = d[dof_i : dof_i + 4]
            fmbr = np.zeros(4)
            for j in range(4):
                fmbr[j] = np.sum(kb[j][:] * dmbr[:])
            fmbr += beam.get_cnl(i)
            res = self._member_values(beam, i, fmbr, dmbr)
            # Shift x vals by location of mbr starting point
            res.x += sumL
            sumL += beam.mbr_lengths[i]
            vRes.append(res)
        return vRes

    def _member_values(
        self, beam: Beam, i_span: int, f: List[float], d: List[float]
    ) -> MemberResults:
        """
        Calculate the load effects along a single member given its nodal
        displacements and forces.

        Parameters
        ----------
        beam : Beam
            The :class:`pycba.beam.Beam` object for which the results are required.
        i_span : : int
            The index of the member along the beam.
        f : List[float]
            The vector of nodal forces from the stiffness analysis.
        d : List[float]
            The vector of nodal displacements from the stiffness analysis.

        Returns
        -------
        MemberResults
            The load effects values along the member.
        """

        L = beam.mbr_lengths[i_span]
        dx = L / self.npts
        x = np.zeros(self.npts + 3)
        x[1 : self.npts + 2] = dx * np.arange(0, self.npts + 1)
        x[self.npts + 2] = L

        # Get the results for the end moments alone
        MaMb = LoadMaMb(i_span=i_span, Ma=f[1], Mb=f[3])
        res = MaMb.get_mbr_results(x, L)

        # Now get the results for all the applied loads on a simple span
        for load in beam.loads:
            if load.i_span != i_span:
                continue
            res += load.get_mbr_results(x, L)

        # And superimpose end displacements using Moment-Area
        h = L / self.npts
        EI = beam.mbr_EIs[i_span]

        R = integrate.cumtrapz(res.M[1:-1], dx=h, initial=0) / EI + d[1]
        D = integrate.cumtrapz(R, dx=h, initial=0) + d[0]

        res.R[1:-1] = R
        res.D[1:-1] = D

        return res


class Envelopes:
    """
    Envelopes load effects from a vector of BeamResults
    """

    def __init__(self, vResults: List[MemberResults]):
        """
        Constructs the envelope of each load effect given a vector of results for
        the beam.

        Parameters
        ----------
        vResults : List[MemberResults]
            The vector of results from each analysis that are to be enveloped.

        Returns
        -------
        None.

        """
        self.vResults = vResults
        self.x = vResults[0].results.x
        self.npts = len(self.x)
        self.nres = len(vResults)
        self.nsup = len(vResults[0].R)

        self.Vmax, self.Vmin = self._get_envelope_V()
        self.Mmax, self.Mmin = self._get_envelope_M()
        self.Rmax, self.Rmin = self._get_envelope_R()
        self.Rmaxval = self.Rmax.max(axis=1)
        self.Rminval = self.Rmin.min(axis=1)

    def _get_envelope_V(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates the envelopes for shear.

        Parameters
        ----------
        None

        Returns
        -------
        Vmax : np.ndarray
            The vector of enveloped maximum values.
        Vmin : np.ndarray
            The vector of enveloped minimum values.
        """
        Vmax = np.zeros(self.npts)
        Vmin = np.zeros(self.npts)

        for res in self.vResults:
            Vmax = np.maximum(Vmax, res.results.V)
            Vmin = np.minimum(Vmin, res.results.V)
        return (Vmax, Vmin)

    def _get_envelope_M(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates the envelopes for moment.

        Parameters
        ----------
        None

        Returns
        -------
        Mmax : np.ndarray
            The vector of enveloped maximum values.
        Mmin : np.ndarray
            The vector of enveloped minimum values.
        """
        Mmax = np.zeros(self.npts)
        Mmin = np.zeros(self.npts)

        for res in self.vResults:
            Mmax = np.maximum(Mmax, res.results.M)
            Mmin = np.minimum(Mmin, res.results.M)
        return (Mmax, Mmin)

    def _get_envelope_R(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates the envelopes for reaction. Strictly this is not an envelope but
        the history of reaction as the vehicle traverses the bridge.

        The returned matrices are of dimension `[nps,nsup]`:

            - `npts` is the number of positions the load was moved
            - `nsup` is the number of full vertical supports

        Parameters
        ----------
        None

        Returns
        -------
        Rmax : np.ndarray
            The matrix of enveloped maximum values.
        Rmin : np.ndarray
            The matrix of enveloped minimum values.
        """
        Rmax = np.zeros((self.nsup, self.nres))
        Rmin = np.zeros((self.nsup, self.nres))
        zero = np.zeros(self.nsup)

        for i, res in enumerate(self.vResults):
            Rmax[:, i] = np.maximum(zero, res.R)  # remove negatives
            Rmin[:, i] = np.minimum(zero, res.R)  # remove positives
        return (Rmax, Rmin)

    @classmethod
    def zero_like(cls, env: Envelopes) -> Envelopes:
        """
        Returns a zeroed zet of envelopes like the reference :class:`pycba.results.Envelopes`.
        This is necessary since a :class:`pycba.results.Envelopes` object stores information
        about the beam from which it came. This facilitates the creation of an
        envelope of envelopes.

        Parameters
        ----------
        env : Envelopes
            A :class:`pycba.results.Envelopes` to be used as the basis for a zeroed
            :class:`pycba.results.Envelopes` object.

        Returns
        -------
        Envelopes
            A :class:`pycba.results.Envelopes` object of zero-valued envelopes.
        """
        zero_env = deepcopy(env)
        zero_env.Vmax = np.zeros(env.npts)
        zero_env.Vmin = np.zeros(env.npts)
        zero_env.Mmax = np.zeros(env.npts)
        zero_env.Mmin = np.zeros(env.npts)
        for i in range(env.nsup):
            zero_env.Rmax[i] = np.zeros(env.nres)
            zero_env.Rmin[i] = np.zeros(env.nres)
        zero_env.Rmaxval = np.zeros(env.nsup)
        zero_env.Rminval = np.zeros(env.nsup)
        return zero_env

    def augment(self, env: Envelopes):
        """
        Augments this set of envelopes with another compatible set, making this the
        envelopes of the two sets of envelopes.

        All envelopes must be from the same :class:`pycba.bridge.BridgeAnalysis` object.

        If the envelopes have a different number of analyses (due to differing vehicle
        lengths, for example), then only the reaction extreme are retained, and not
        the entire reaction history.

        Parameters
        ----------
        env : Envelopes
            A compatible :class:`pycba.results.Envelopes` object.

        Raises
        ------
        ValueError
            All envelopes must be for the same bridge.

        Returns
        -------
        None.
        """

        if self.npts != env.npts or self.nsup != env.nsup:
            raise ValueError("Cannot augment with an inconsistent envelope")
        self.Vmax = np.maximum(self.Vmax, env.Vmax)
        self.Vmin = np.minimum(self.Vmin, env.Vmin)

        self.Mmax = np.maximum(self.Mmax, env.Mmax)
        self.Mmin = np.minimum(self.Mmin, env.Mmin)

        self.Rmaxval = np.maximum(self.Rmaxval, env.Rmaxval)
        self.Rminval = np.minimum(self.Rminval, env.Rminval)

        if self.nres == env.nres:
            self.Rmax = np.maximum(self.Rmax, env.Rmax)
            self.Rmin = np.minimum(self.Rmin, env.Rmin)
        else:
            # Ensure no misleading results returned
            self.Rmax = np.zeros((self.nsup, self.nres))
            self.Rmin = np.zeros((self.nsup, self.nres))
