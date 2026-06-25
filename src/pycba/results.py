"""
PyCBA - Beam Results module
"""

from __future__ import annotations  # https://bit.ly/3KYiL2o
from typing import Dict, List, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from .beam import Beam
from .section import SectionEI
from .types import MemberResults
from .load import LoadMaMb, LoadCNL, LoadIC
from copy import deepcopy


class BeamResults:
    """
    BeamResults Class for processing and containing the results for each member
    """

    def __init__(
        self,
        beam: Beam,
        d: np.ndarray,
        r: np.ndarray,
        npts: int = 100,
        rs: np.ndarray = None,
    ):
        """
        Initialize member results from global results

        Parameters
        ----------
        beam : Beam
            The :class:`pycba.beam.Beam` object for which the results are to be stored.
        d : np.ndarray
            The vector of nodal displacements (from the stiffness method).
        r : np.ndarray
            The vector of reactions at fixed restraints (restraint == -1).
        npts : int, optional
            The number of points along the member at which to calculate the load
            effects. The default is 100.
        rs : np.ndarray, optional
            The vector of spring forces (k_s * u_i) for spring restraints (restraint > 0).

        Returns
        -------
        None.
        """
        self.npts = npts
        self.vRes = self._member_analysis(beam, d)
        self.D = d  # nodal displacements
        self.R = r  # reactions at fixed restraints (restraint == -1)
        self.Rs = rs if rs is not None else np.array([])  # spring forces
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
            fmbr += beam.get_ref(i)
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
        EI = beam.mbr_EIs[i_span]
        etype = beam.mbr_eletype[i_span]
        mbr_GAv = getattr(beam, "mbr_GAv", [])
        GAv = mbr_GAv[i_span] if i_span < len(mbr_GAv) else None

        dx = L / self.npts
        x = np.zeros(self.npts + 3)
        x[1 : self.npts + 2] = dx * np.arange(0, self.npts + 1)
        x[self.npts + 2] = L

        # Get the results for the end moments alone
        MaMb = LoadMaMb(i_span=i_span, Ma=f[1], Mb=f[3])
        res = MaMb.get_mbr_results(x, L)

        # Now get the results for all the applied loads on a simple span.  Any
        # imposed-curvature (initial-strain) load contributes a free curvature
        # kappa_imp(x) which is accumulated here for the moment-area integration.
        Ma = 0
        Mb = 0
        kappa_imp = np.zeros_like(x)
        for load in beam._loads:
            if load.i_span != i_span:
                continue
            res += load.get_mbr_results(x, L)
            cnl = load.get_cnl(L, etype)
            Ma += cnl.Ma
            Mb += cnl.Mb
            if isinstance(load, LoadIC):
                kappa_imp += load.kappa_imp(x)

        # Total curvature along the member: flexural M(x)/EI(x) plus any free
        # (imposed) curvature.  EI(x) is evaluated point-wise for non-prismatic
        # members and is constant for the prismatic path.
        if isinstance(EI, SectionEI):
            curv = res.M / EI(x)
        else:
            curv = res.M / EI
        curv = curv + kappa_imp

        h = L / self.npts

        # Shear-deformable (Timoshenko) member: the rotation reported (and used
        # as the integration constant) is the *cross-section* rotation ``psi``;
        # the slope of the deflected shape is ``psi + gamma`` with the shear
        # strain ``gamma = V/GAv``.  For Euler–Bernoulli (``GAv is None``) the
        # shear slope is zero and ``psi`` is the slope, so the result below is
        # bit-for-bit identical to the previous behaviour.
        if GAv is not None:
            GAv_x = GAv(x[1:-1]) if isinstance(GAv, SectionEI) else GAv
            # PyCBA's shear sign is opposite to the (dw/dx = psi + gamma)
            # convention, so the shear slope enters with a leading minus.
            gamma = -res.V[1:-1] / GAv_x
        else:
            gamma = 0.0

        # Shear deformation forces the boundary-condition integration for a
        # released member (the closed-form rotation correction is
        # Euler–Bernoulli only); non-prismatic and imposed-curvature members
        # likewise recover the i-end rotation from the kinematic BC D(L) = d[2].
        use_bc = (
            isinstance(EI, SectionEI) or np.any(kappa_imp != 0.0) or GAv is not None
        )

        # Provisional cross-section rotation from the bending curvature.
        psi_prov = integrate.cumulative_trapezoid(curv[1:-1], dx=h, initial=0)

        if etype > 1 and use_bc:
            # Recover the i-end rotation from the kinematic BC D(L) = d[2],
            # valid for any EI(x), GAv(x) and curvature field.
            slope_prov = psi_prov + gamma
            D_prov = integrate.cumulative_trapezoid(slope_prov, dx=h, initial=0)
            psi_i = (d[2] - d[0] - D_prov[-1]) / L
            psi = psi_prov + psi_i
        elif etype > 1:
            # Euler–Bernoulli closed-form release correction (no shear).
            theta = (d[2] - d[0]) / L
            phi = (L / (3 * EI)) * (-(f[1] - 0.5 * f[3]) + (Ma - 0.5 * Mb))
            psi = psi_prov + (theta - phi)
        else:
            # No release: the i-end cross-section rotation is the nodal DOF.
            psi = psi_prov + d[1]

        slope = psi + gamma
        D = integrate.cumulative_trapezoid(slope, dx=h, initial=0) + d[0]

        res.R[1:-1] = psi
        res.D[1:-1] = D

        return res


class Envelopes:
    """
    Envelopes load effects from a vector of BeamResults.

    Attributes
    ----------
    x : np.ndarray
        Coordinates along the beam.
    Vmax, Vmin : np.ndarray
        Maximum and minimum shear force envelopes.
    Mmax, Mmin : np.ndarray
        Maximum and minimum bending moment envelopes.
    Vco_Mmax : np.ndarray
        Shear coincident with the moment maximum at each point.
    Vco_Mmin : np.ndarray
        Shear coincident with the moment minimum at each point.
    Mco_Vmax : np.ndarray
        Moment coincident with the shear maximum at each point.
    Mco_Vmin : np.ndarray
        Moment coincident with the shear minimum at each point.
    Rmax, Rmin : np.ndarray
        Reaction history matrices (nsup x nres).
    Rmaxval, Rminval : np.ndarray
        Maximum and minimum reaction per support.
    """

    def __init__(self, vResults: List[MemberResults]):
        """
        Constructs the envelope of each load effect given a vector of results for
        the beam. Also tracks coincident (co-existing) load effects: the value of
        the other effect (V or M) at the analysis that caused each envelope extreme.

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

        (
            self.Vmax,
            self.Vmin,
            self.Mmax,
            self.Mmin,
            self.Vco_Mmax,
            self.Vco_Mmin,
            self.Mco_Vmax,
            self.Mco_Vmin,
        ) = self._get_envelopes_VM()
        self.Rmax, self.Rmin = self._get_envelope_R()
        self.Rmaxval = self.Rmax.max(axis=1)
        self.Rminval = self.Rmin.min(axis=1)

    def _get_envelopes_VM(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Creates the envelopes for shear and moment, and tracks the coincident
        (co-existing) value of the other effect at the truck position that
        caused each extreme.

        Returns
        -------
        Vmax, Vmin : np.ndarray
            Enveloped maximum and minimum shear.
        Mmax, Mmin : np.ndarray
            Enveloped maximum and minimum moment.
        Vco_Mmax, Vco_Mmin : np.ndarray
            Shear coincident with the moment extreme at each point.
        Mco_Vmax, Mco_Vmin : np.ndarray
            Moment coincident with the shear extreme at each point.
        """
        Vmax = np.zeros(self.npts)
        Vmin = np.zeros(self.npts)
        Mmax = np.zeros(self.npts)
        Mmin = np.zeros(self.npts)

        Vco_Mmax = np.zeros(self.npts)
        Vco_Mmin = np.zeros(self.npts)
        Mco_Vmax = np.zeros(self.npts)
        Mco_Vmin = np.zeros(self.npts)

        for res in self.vResults:
            V = res.results.V
            M = res.results.M

            mask = M > Mmax
            Vco_Mmax = np.where(mask, V, Vco_Mmax)
            Mmax = np.where(mask, M, Mmax)

            mask = M < Mmin
            Vco_Mmin = np.where(mask, V, Vco_Mmin)
            Mmin = np.where(mask, M, Mmin)

            mask = V > Vmax
            Mco_Vmax = np.where(mask, M, Mco_Vmax)
            Vmax = np.where(mask, V, Vmax)

            mask = V < Vmin
            Mco_Vmin = np.where(mask, M, Mco_Vmin)
            Vmin = np.where(mask, V, Vmin)

        return (Vmax, Vmin, Mmax, Mmin, Vco_Mmax, Vco_Mmin, Mco_Vmax, Mco_Vmin)

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
        zero_env.Vco_Mmax = np.zeros(env.npts)
        zero_env.Vco_Mmin = np.zeros(env.npts)
        zero_env.Mco_Vmax = np.zeros(env.npts)
        zero_env.Mco_Vmin = np.zeros(env.npts)
        for i in range(env.nsup):
            zero_env.Rmax[i] = np.zeros(env.nres)
            zero_env.Rmin[i] = np.zeros(env.nres)
        zero_env.Rmaxval = np.zeros(env.nsup)
        zero_env.Rminval = np.zeros(env.nsup)
        return zero_env

    @classmethod
    def combine(cls, envs: Sequence[Envelopes], mode: str = "envelope") -> Envelopes:
        """
        Merge several compatible envelopes into a new :class:`Envelopes`.

        This is a one-call replacement for the
        ``zero_like`` + repeated :meth:`augment`/:meth:`sum` ritual. The input
        envelopes are left unmutated; a fresh :class:`Envelopes` is returned.

        Parameters
        ----------
        envs : Sequence[Envelopes]
            One or more compatible :class:`pycba.results.Envelopes` objects (same
            ``npts`` and ``nsup``). All must come from the same beam geometry.
        mode : str, optional
            ``"envelope"`` (default) takes the governing maxima/minima of moment
            and shear (plus coincident effects and reaction extremes) by calling
            :meth:`augment` for each input. ``"sum"`` superimposes the envelopes
            element-wise by calling :meth:`sum` for each input.

        Returns
        -------
        Envelopes
            A new merged :class:`pycba.results.Envelopes` object.

        Raises
        ------
        ValueError
            If ``envs`` is empty or ``mode`` is not ``"envelope"`` or ``"sum"``.

        Notes
        -----
        When the merged envelopes come from analyses with differing numbers of
        results (``nres``), only the reaction extremes (``Rmaxval``/``Rminval``)
        are retained; the per-analysis reaction history is zeroed. This is the
        pre-existing behaviour of :meth:`augment`/:meth:`sum`.
        """

        envs = list(envs)
        if len(envs) == 0:
            raise ValueError("combine requires at least one envelope.")
        if mode not in ("envelope", "sum"):
            raise ValueError("mode must be 'envelope' or 'sum'.")

        out = cls.zero_like(envs[0])
        for env in envs:
            if mode == "envelope":
                out.augment(env)
            else:
                out.sum(env)
        return out

    @classmethod
    def from_beam_analysis(cls, ba) -> Envelopes:
        """
        Build a single-result :class:`Envelopes` from one analysed beam.

        This is the discoverable adapter that lifts an ordinary
        :class:`pycba.analysis.BeamAnalysis` result (for example the result of a
        :meth:`pycba.load_cases.LoadCombination.analyze` call) into a first-class
        :class:`pycba.results.Envelopes`, so it can be merged with
        :meth:`combine` or the ``|``/``+`` operators and plotted.

        Parameters
        ----------
        ba : BeamAnalysis
            An analysed beam whose ``beam_results`` are available.

        Returns
        -------
        Envelopes
            A one-result :class:`pycba.results.Envelopes`.

        Raises
        ------
        ValueError
            If ``ba`` has not been analysed (``ba.beam_results`` is ``None``).
        """

        beam_results = getattr(ba, "beam_results", None)
        if beam_results is None:
            raise ValueError(
                "BeamAnalysis has no results; call analyze() before "
                "Envelopes.from_beam_analysis()."
            )
        return cls([beam_results])

    def augment(self, env: Envelopes):
        """
        Augments this set of envelopes with another compatible set, making this the
        envelopes of the two sets of envelopes. Coincident values are updated to
        match whichever envelope governs at each point.

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

        # Track where envelope values will change BEFORE updating them
        vmax_update = env.Vmax > self.Vmax
        vmin_update = env.Vmin < self.Vmin
        mmax_update = env.Mmax > self.Mmax
        mmin_update = env.Mmin < self.Mmin

        self.Vmax = np.maximum(self.Vmax, env.Vmax)
        self.Vmin = np.minimum(self.Vmin, env.Vmin)
        self.Mmax = np.maximum(self.Mmax, env.Mmax)
        self.Mmin = np.minimum(self.Mmin, env.Mmin)

        # Update coincident values where the envelope changed
        self.Vco_Mmax = np.where(mmax_update, env.Vco_Mmax, self.Vco_Mmax)
        self.Vco_Mmin = np.where(mmin_update, env.Vco_Mmin, self.Vco_Mmin)
        self.Mco_Vmax = np.where(vmax_update, env.Mco_Vmax, self.Mco_Vmax)
        self.Mco_Vmin = np.where(vmin_update, env.Mco_Vmin, self.Mco_Vmin)

        self.Rmaxval = np.maximum(self.Rmaxval, env.Rmaxval)
        self.Rminval = np.minimum(self.Rminval, env.Rminval)

        if self.nres == env.nres:
            self.Rmax = np.maximum(self.Rmax, env.Rmax)
            self.Rmin = np.minimum(self.Rmin, env.Rmin)
        else:
            # Ensure no misleading results returned
            self.Rmax = np.zeros((self.nsup, self.nres))
            self.Rmin = np.zeros((self.nsup, self.nres))

    def sum(self, env: Envelopes):
        """
        Adds another compatible set of envelopes to this one element-wise. This is
        useful for superimposing load effects from different sources, e.g. a patterned
        UDL envelope with a moving vehicle envelope.

        All envelopes must be for the same beam geometry.

        If the envelopes have a different number of analyses (due to differing vehicle
        lengths, for example), then only the reaction extremes are retained, and not
        the entire reaction history.

        Parameters
        ----------
        env : Envelopes
            A compatible :class:`pycba.results.Envelopes` object.

        Raises
        ------
        ValueError
            All envelopes must be for the same beam geometry.

        Returns
        -------
        None.
        """

        if self.npts != env.npts or self.nsup != env.nsup:
            raise ValueError("Cannot sum with an inconsistent envelope")
        self.Vmax = np.add(self.Vmax, env.Vmax)
        self.Vmin = np.add(self.Vmin, env.Vmin)

        self.Mmax = np.add(self.Mmax, env.Mmax)
        self.Mmin = np.add(self.Mmin, env.Mmin)

        self.Rmaxval = np.add(self.Rmaxval, env.Rmaxval)
        self.Rminval = np.add(self.Rminval, env.Rminval)

        if self.nres == env.nres:
            self.Rmax = np.add(self.Rmax, env.Rmax)
            self.Rmin = np.add(self.Rmin, env.Rmin)
        else:
            # Ensure no misleading results returned
            self.Rmax = np.zeros((self.nsup, self.nres))
            self.Rmin = np.zeros((self.nsup, self.nres))

    def __or__(self, other: Envelopes) -> Envelopes:
        """
        Enclosing envelope of ``self`` and ``other`` (non-mutating).

        ``a | b`` is sugar for ``Envelopes.combine([a, b], mode="envelope")`` and
        returns a new :class:`Envelopes` taking the governing maxima/minima of the
        two operands. Neither operand is mutated.

        Note that this differs from :meth:`augment`, which mutates in place and
        returns ``None``.
        """

        return Envelopes.combine([self, other], mode="envelope")

    def __add__(self, other: Envelopes) -> Envelopes:
        """
        Superposition of ``self`` and ``other`` (non-mutating).

        ``a + b`` is sugar for ``Envelopes.combine([a, b], mode="sum")`` and
        returns a new :class:`Envelopes` with the two operands summed
        element-wise. Neither operand is mutated.

        Note that this differs from :meth:`sum`, which mutates in place and
        returns ``None``.
        """

        return Envelopes.combine([self, other], mode="sum")

    def _unique_x(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the de-duplicated station coordinates and their indices."""

        return np.unique(self.x, return_index=True)

    def at(
        self,
        x: float,
        attrs: Tuple[str, ...] = ("Mmax", "Mmin", "Vmax", "Vmin"),
    ) -> Dict[str, float]:
        """
        Return envelope values at a global coordinate by interpolation.

        The station array ``self.x`` repeats span-boundary coordinates (each span
        contributes its own start and end), so this method de-duplicates the
        stations with :func:`numpy.unique` before interpolating. It replaces the
        manual ``idx = (np.abs(env.x - x)).argmin()`` lookup.

        Parameters
        ----------
        x : float
            Global coordinate measured from the left end of the beam.
        attrs : tuple of str, optional
            Envelope attribute names to evaluate. The default returns the moment
            and shear extremes.

        Returns
        -------
        dict
            Mapping from each requested attribute name to its interpolated value
            at ``x``.
        """

        unique_x, unique_index = self._unique_x()
        out: Dict[str, float] = {}
        for attr in attrs:
            values = np.asarray(getattr(self, attr))
            out[attr] = float(np.interp(float(x), unique_x, values[unique_index]))
        return out

    def per_span(self, attr: str = "Mmax", reduce: str = "auto") -> np.ndarray:
        """
        Split a station-indexed envelope array into per-span values.

        The station array concatenates each member's stations, so this method
        chunks the requested attribute using the per-member station counts read
        from ``self.vResults[0].vRes`` (each member's ``x`` length, i.e.
        ``npts + 3``) rather than a hard-coded constant. It replaces the manual
        ``env.Vmax[i * (n + 3):(i + 1) * (n + 3)]`` slicing.

        Parameters
        ----------
        attr : str, optional
            Envelope attribute name to split (default ``"Mmax"``).
        reduce : str, optional
            How to reduce each per-span chunk. ``"auto"`` (default) takes the
            maximum for ``*max`` attributes and the minimum for ``*min``
            attributes; ``"max"``/``"min"`` force the reduction; ``"none"``
            returns the list of raw chunks.

        Returns
        -------
        numpy.ndarray or list of numpy.ndarray
            One reduced value per span, or the list of raw chunks when
            ``reduce="none"``.

        Raises
        ------
        ValueError
            If ``reduce`` is not one of ``"auto"``, ``"max"``, ``"min"``,
            ``"none"``.
        """

        if reduce not in ("auto", "max", "min", "none"):
            raise ValueError("reduce must be 'auto', 'max', 'min', or 'none'.")

        values = np.asarray(getattr(self, attr))
        counts = [len(member.x) for member in self.vResults[0].vRes]

        chunks = []
        start = 0
        for count in counts:
            chunks.append(values[start : start + count])
            start += count

        if reduce == "none":
            return chunks

        if reduce == "auto":
            op = np.min if attr.endswith("min") else np.max
        else:
            op = np.max if reduce == "max" else np.min

        return np.array([op(chunk) for chunk in chunks])

    def plot(self, each=False, units=None, **kwargs):
        """
        Plots the envelopes of bending and shear.

        Parameters
        ----------
        each : Boolean
            Wether or not to show each BMD and SFD in the enveloping. The default is False
        units : str or pycba.units.UnitSystem, optional
            Display unit system for the labels (see :func:`pycba.set_units`).
        **kwargs : Dict
            Matplotlib keyword arguments for plotting.

        Returns
        -------
        None.

        """
        from .units import resolve

        us = resolve(units)
        if self.nres < 1:
            raise ValueError("No results to display")

        L = self.x[-1]

        kwargs.setdefault("figsize", (10, 6))
        fig, axs = plt.subplots(2, 1, sharex=True, **kwargs)

        ax = axs[0]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(self.x, self.Mmax, "r")
        ax.plot(self.x, self.Mmin, "b")
        ax.grid()
        ax.invert_yaxis()
        ax.set_ylabel(us.moment_axis)

        ax = axs[1]
        ax.plot([0, L], [0, 0], "k", lw=2)
        ax.plot(self.x, self.Vmax, "r")
        ax.plot(self.x, self.Vmin, "b")
        ax.grid()
        ax.set_ylabel(us.shear_axis)
        ax.set_xlabel(us.distance_axis)

        if each:
            for res in self.vResults:
                axs[0].plot(self.x, res.results.M, "r", lw=0.5)
                axs[1].plot(self.x, res.results.V, "b", lw=0.5)

        return fig, ax
