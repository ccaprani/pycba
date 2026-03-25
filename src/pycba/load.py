"""
PyCBA - Load module

The load matrix is a ``List[List]`` of load descriptors.  Each entry
describes one load; the number of columns varies by load type:

=====  ====================  ================================  ====
Type   Name                  Format                            Cols
=====  ====================  ================================  ====
1      UDL                   ``[span, 1, w]``                  3
2      Point Load            ``[span, 2, P, a]``               4
3      Partial UDL           ``[span, 3, w, a, c]``            5
4      Moment Load           ``[span, 4, M, a]``               4
5      Trapezoidal (full)    ``[span, 5, w1, w2]``             4
5      Trapezoidal (partial) ``[span, 5, w1, w2, a, c]``       6
=====  ====================  ================================  ====

The type alias `LoadMatrix` is defined as

.. autodata:: LoadMatrix

"""

from __future__ import annotations
from typing import Union, List, NamedTuple, Tuple, Optional
import numpy as np

from .types import LoadType, LoadMatrix, LoadCNL, MemberResults


class Load:
    """
    Beam load container and processor
    """

    def __init__(self, i_span: int):
        """
        Initialize the loads from the Load Matrix

        Parameters
        ----------
        i_span : int
            The index of the span (or member), 1-based
        """
        self.i_span = i_span

    def get_cnl(self, L):
        # Enforce virtual base class
        raise NotImplementedError

    def get_mbr_results(self, x, L):
        raise NotImplementedError

    def MB(self, v: np.ndarray) -> np.ndarray:
        """
        Macaulay bracket: clipping values less than zero to zero.

        Parameters
        ----------
        v : np.ndarray
            The vector to which the Macaulay Bracket will be applied
        """
        return v.clip(0.0)

    def H(self, v: np.ndarray, value: float = 0.0) -> np.ndarray:
        """
        Heaviside step function: values less than zero are clipped to zero;
        values greater than zero are clipped to unity; zeros are retained.

        Parameters
        ----------
        v : np.ndarray
            The vector to which the Heaviside function will be applied
        value : float
            The value of the Heaviside function at zero, usually 0, but sometimes
            0.5 (average of adjacent values) or 1.0.
        """
        return np.heaviside(v, value)

    def get_ref(self, L: float, eType: int) -> LoadCNL:
        """
        Returns the Released End Forces for a span of length L of element eType:
        converts the Consistent Nodal Loads of the applied loading to the correct nodal
        loading depending on the element type.

        Parameters
        ----------
        L : float
            The length of the member
        eType : int
            The member element type

        Returns
        -------
        LoadCNL
            Released End Forces for this load type: the nodal loads to be applied in
            the analysis, consistent with the element type.
        """
        cnl = self.get_cnl(L, eType)
        ref = np.zeros(4)
        fm = 6 / (4 * L)  # flexibility coeff for moment

        if eType == 2:  # DOF = moment at j node
            ref[0] = fm * cnl.Mb
            ref[1] = 0.5 * cnl.Mb
            ref[2] = -fm * cnl.Mb
            ref[3] = 1.0 * cnl.Mb
        elif eType == 3:  # DOF = moment at i node
            ref[0] = fm * cnl.Ma
            ref[1] = 1.0 * cnl.Ma
            ref[2] = -fm * cnl.Ma
            ref[3] = 0.5 * cnl.Ma
        elif eType == 4:  # keep only vertical, remove moments
            ref[0] = -(cnl.Ma + cnl.Mb) / L
            ref[1] = 1.0 * cnl.Ma
            ref[2] = (cnl.Ma + cnl.Mb) / L
            ref[3] = 1.0 * cnl.Mb
        else:
            # no nothing if it is FF
            pass
        # now superimpose the released forces
        return LoadCNL(
            Va=cnl.Va - ref[0],
            Ma=cnl.Ma - ref[1],
            Vb=cnl.Vb - ref[2],
            Mb=cnl.Mb - ref[3],
        )


class LoadUDL(Load):
    """
    Uniformly Distributed Load: CNLs and member results
    """

    def __init__(self, i_span: int, w: float):
        """
        Creates a UDL for the member

        Parameters
        ----------
        i_span : int
            The member index to which the load is applied.
        w : float
            The load magnitude.

        Returns
        -------
        None.

        """
        super().__init__(i_span)
        self.w = w

    def get_cnl(self, L: float, eType: int) -> LoadCNL:
        """
        Returns the Consistent Nodal Loads for a span of length L of element eType

        Parameters
        ----------
        L : float
            The length of the member
        eType : int
            The member element type

        Returns
        -------
        LoadCNL
            Consistent Nodal Loads for this load type
        """

        w = self.w

        cnl = LoadCNL(
            # Shears
            Va=w * L / 2.0,
            Vb=w * L / 2.0,
            # Moments
            Ma=w * L**2 / 12.0,
            Mb=-w * L**2 / 12.0,
        )
        return cnl

    def get_mbr_results(self, x: np.ndarray, L: float) -> MemberResults:
        """
        Results along the member from UDL

        Parameters
        ----------
        x : np.ndarray
            Vector of points along the length of the member
        L : float
            The length of the member

        Returns
        -------
        res : MemberResults
            A populated :class:`pycba.load.MemberResults` object
        """

        npts = len(x)
        res = MemberResults(vals=None, n=npts)
        res.x = x

        Va = self.w * L / 2
        Ra = -self.w * L**3 / 24

        res.V = Va - self.w * x
        res.M = Va * x - self.w / 2 * x**2
        res.R = (Va / 2) * x**2 - (self.w / 6) * x**3 + Ra
        res.D = (Va / 6) * x**3 - (self.w / 24) * x**4 + Ra * x

        res.V[0] = 0
        res.V[npts - 1] = 0
        res.M[0] = 0
        res.M[npts - 1] = 0

        return res


class LoadPL(Load):
    """
    Point Load class: CNLs and member results
    """

    def __init__(self, i_span: int, P: float, a: float):
        """
        Creates a Point Load for the member

        Parameters
        ----------
        i_span : int
            The member index to which the load is applied.
        P : float
            The load magnitude.
        a : float
            The load location along the member

        Returns
        -------
        None.

        """
        super().__init__(i_span)
        self.P = P
        self.a = a

    def get_cnl(self, L, eType) -> LoadCNL:
        """
        Returns the Consistent Nodal Loads for a span of length L of element eType

        Parameters
        ----------
        L : float
            The length of the member
        eType : int
            The member element type

        Returns
        -------
        LoadCNL
            Consistent Nodal Loads for this load type

        """

        P = self.P
        a = self.a
        b = max(L - a, 0)

        cnl = LoadCNL(
            # Shears
            Va=P / L**3 * (b * L**2 - a**2 * b + a * b**2),
            Vb=P / L**3 * (a * L**2 + a**2 * b - a * b**2),
            # Moments
            Ma=P * a * b**2 / L**2,
            Mb=-P * a**2 * b / L**2,
        )
        return cnl

    def get_mbr_results(self, x: np.ndarray, L: float) -> MemberResults:
        """
        Results along the member from this load

        Parameters
        ----------
        x : np.ndarray
            Vector of points along the length of the member
        L : float
            The length of the member

        Returns
        -------
        res : MemberResults
            A populated :class:`pycba.load.MemberResults` object
        """

        npts = len(x)
        res = MemberResults(vals=None, n=npts)
        res.x = x

        P = self.P
        a = self.a
        b = max(L - a, 0)

        Va = P * b / L
        Ra = P * b * (b**2 - L**2) / (6 * L)

        res.V = Va - P * self.H(x - a)
        res.M = Va * x - P * self.MB(x - a)
        res.R = (Va / 2) * x**2 - (P / 2) * self.MB(x - a) ** 2 + Ra
        res.D = (Va / 6) * x**3 - (P / 6) * self.MB(x - a) ** 3 + Ra * x

        res.V[0] = 0
        res.V[npts - 1] = 0
        res.M[0] = 0
        res.M[npts - 1] = 0

        return res


class LoadPUDL(Load):
    """
    Concrete class for Partial UDLs
    """

    def __init__(self, i_span, w, a, c):
        super().__init__(i_span)
        self.w = w
        self.a = a
        self.c = c

    def get_cnl(self, L, eType) -> LoadCNL:
        """
        Returns the Consistent Nodal Loads for a span of length L of element eType

        Parameters
        ----------
        L : float
            The length of the member
        eType : int
            The member element type

        Returns
        -------
        LoadCNL
            Consistent Nodal Loads for this load type
        """

        a = self.a
        c = self.c
        w = self.w
        # Check if on span, if not, return zeros
        if self.a > L:
            return [0.0] * 4
        # Actual cover on span
        d = L - (a + c)
        # If cover hangs off span, adjust it
        if d < 0:
            c += d
        # More useful vars
        s = a + c / 2
        t = L - s

        cnl = LoadCNL(
            # Shears
            Va=(w * c / L**3) * ((2 * s + L) * t**2 + (s - t) * c**2 / 4),
            Vb=w * c - (w * c / L**3) * ((2 * s + L) * t**2 + (s - t) * c**2 / 4),
            # Moments
            Ma=(w * c / L**2) * (s * t**2 + (s - 2 * t) * c**2 / 12),
            Mb=-(w * c / L**2) * (t * s**2 + (t - 2 * s) * c**2 / 12),
        )
        # implicit conversion to tuple in correct order
        return cnl

    def get_mbr_results(self, x: np.ndarray, L: float) -> MemberResults:
        """
        Results along the member from this load

        Parameters
        ----------
        x : np.ndarray
            Vector of points along the length of the member
        L : float
            The length of the member

        Returns
        -------
        res : MemberResults
            A populated :class:`pycba.load.MemberResults` object
        """

        npts = len(x)
        res = MemberResults(vals=None, n=npts)
        res.x = x

        a = self.a
        c = self.c
        w = self.w
        b = c + a

        Va = (L - b + c / 2) * c * w / L
        Ra = (
            -((Va / 6) * L**3 + (w / 24) * (L - b) ** 4 - (w / 24) * (L - a) ** 4) / L
        )

        res.V = Va - w * self.MB(x - a) + w * self.MB(x - b)
        res.M = (
            Va * x - (w / 2) * (self.MB(x - a)) ** 2 + (w / 2) * (self.MB(x - b)) ** 2
        )
        res.R = (
            (Va / 2) * x**2
            - (w / 6) * (self.MB(x - a)) ** 3
            + (w / 6) * (self.MB(x - b)) ** 3
            + Ra
        )
        res.D = (
            (Va / 6) * x**3
            - (w / 24) * (self.MB(x - a)) ** 4
            + (w / 24) * (self.MB(x - b)) ** 4
            + Ra * x
        )

        res.V[0] = 0
        res.V[npts - 1] = 0
        res.M[0] = 0
        res.M[npts - 1] = 0

        return res


class LoadTrapez(Load):
    """
    Trapezoidal (linearly varying) distributed load, optionally partial.

    The load varies linearly from intensity *w1* at position *a* to *w2* at
    position *a + c*.  When *a* = 0 and *c* = span length the load covers
    the full span (the default).
    """

    def __init__(
        self,
        i_span: int,
        w1: float,
        w2: float,
        a: float = 0.0,
        c: Optional[float] = None,
    ):
        """
        Creates a trapezoidal load for the member.

        Parameters
        ----------
        i_span : int
            The member index to which the load is applied.
        w1 : float
            The load intensity at position *a* (left edge of the load).
        w2 : float
            The load intensity at position *a + c* (right edge of the load).
        a : float, optional
            Distance from the left end of the span to the start of the load.
            Default is 0 (load starts at the left end).
        c : float or None, optional
            Length (cover) of the load.  ``None`` (default) means full span
            from *a* to the right end.
        """
        super().__init__(i_span)
        self.w1 = w1
        self.w2 = w2
        self.a = a
        self._c = c  # None ⇒ full span from a, resolved when L is known

    def _resolve(self, L: float):
        """Resolve c and clip to span boundaries.

        Returns (w1, w2, dw, a, c) with c > 0, or c = 0 when the load
        falls outside the span.
        """
        a = self.a
        c = self._c if self._c is not None else L - a
        w1 = self.w1
        w2 = self.w2

        if a >= L or c <= 0:
            return w1, w2, 0.0, a, 0.0

        # Clip overhang
        if a + c > L:
            c_orig = c
            c = L - a
            w2 = w1 + (w2 - w1) * c / c_orig  # interpolated at clipped end

        return w1, w2, w2 - w1, a, c

    def get_cnl(self, L: float, eType: int) -> LoadCNL:
        """
        Consistent Nodal Loads for the trapezoidal load on a fixed-fixed span.

        Derived from the fixed-end force influence integrals:

        .. math::
            M_A = \\frac{1}{L^2}\\int_a^b w(x)\\,x\\,(L-x)^2\\,dx

        Parameters
        ----------
        L : float
            The length of the member
        eType : int
            The member element type

        Returns
        -------
        LoadCNL
            Consistent Nodal Loads for this load type
        """
        w1, w2, dw, a, c = self._resolve(L)

        if c <= 0:
            return LoadCNL(Va=0.0, Vb=0.0, Ma=0.0, Mb=0.0)

        alpha = L - a  # distance: load start → right beam end
        # delta = L - a - c    # distance: load end → right beam end (not used directly)

        # --- Ma: ∫ w(x) · x · (L−x)² dx / L² ---
        # Split into UDL(w1) integral I1 and triangular(dw) integral I2.
        I1 = (
            c**4 / 4
            + (a - 2 * alpha) * c**3 / 3
            + (alpha**2 - 2 * a * alpha) * c**2 / 2
            + a * alpha**2 * c
        )
        I2 = (
            c**5 / 5
            + (a - 2 * alpha) * c**4 / 4
            + (alpha**2 - 2 * a * alpha) * c**3 / 3
            + a * alpha**2 * c**2 / 2
        )
        Ma = (w1 / L**2) * I1 + (dw / (c * L**2)) * I2

        # --- Mb: −∫ w(x) · x² · (L−x) dx / L² ---
        J1 = (
            -(c**4) / 4
            + (alpha - 2 * a) * c**3 / 3
            + (2 * a * alpha - a**2) * c**2 / 2
            + a**2 * alpha * c
        )
        J2 = (
            -(c**5) / 5
            + (alpha - 2 * a) * c**4 / 4
            + (2 * a * alpha - a**2) * c**3 / 3
            + a**2 * alpha * c**2 / 2
        )
        Mb = -((w1 / L**2) * J1 + (dw / (c * L**2)) * J2)

        # --- Va: ∫ w(x) · (L−x)² · (2x+L) dx / L³ ---
        K1 = (
            c**4 / 2
            + (a - alpha) * c**3
            - 3 * a * alpha * c**2
            + (3 * a + alpha) * alpha**2 * c
        )
        K2 = (
            2 * c**5 / 5
            + 3 * (a - alpha) * c**4 / 4
            - 2 * a * alpha * c**3
            + (3 * a + alpha) * alpha**2 * c**2 / 2
        )
        Va = (w1 / L**3) * K1 + (dw / (c * L**3)) * K2

        Vb = (w1 + w2) * c / 2 - Va

        return LoadCNL(Va=Va, Vb=Vb, Ma=Ma, Mb=Mb)

    def get_mbr_results(self, x: np.ndarray, L: float) -> MemberResults:
        """
        Simply-supported member results using Macaulay bracket integration.

        The load ``w(x) = w1 + (w2−w1)·(x−a)/c`` for ``a ≤ x ≤ a+c`` is
        integrated using Macaulay brackets at positions *a* and *b = a+c*.

        Parameters
        ----------
        x : np.ndarray
            Vector of points along the length of the member
        L : float
            The length of the member

        Returns
        -------
        res : MemberResults
            A populated :class:`pycba.load.MemberResults` object
        """
        npts = len(x)
        res = MemberResults(vals=None, n=npts)
        res.x = x

        w1, w2, dw, a, c = self._resolve(L)

        if c <= 0:
            res.V = np.zeros(npts)
            res.M = np.zeros(npts)
            res.R = np.zeros(npts)
            res.D = np.zeros(npts)
            return res

        b = a + c
        alpha = L - a
        delta = L - b

        # Simply-supported reaction at left end (moment equilibrium about B)
        Va = (
            (w1 / 2) * alpha**2
            + (dw / (6 * c)) * alpha**3
            - (w2 / 2) * delta**2
            - (dw / (6 * c)) * delta**3
        ) / L

        # Rotation integration constant (from D(0) = D(L) = 0)
        Ra = (
            -(Va / 6) * L**3
            + (w1 / 24) * alpha**4
            + (dw / (120 * c)) * alpha**5
            - (w2 / 24) * delta**4
            - (dw / (120 * c)) * delta**5
        ) / L

        # Macaulay brackets at load start and end
        MBa = self.MB(x - a)
        MBb = self.MB(x - b)

        res.V = (
            Va
            - w1 * MBa
            - (dw / (2 * c)) * MBa**2
            + w2 * MBb
            + (dw / (2 * c)) * MBb**2
        )
        res.M = (
            Va * x
            - (w1 / 2) * MBa**2
            - (dw / (6 * c)) * MBa**3
            + (w2 / 2) * MBb**2
            + (dw / (6 * c)) * MBb**3
        )
        res.R = (
            (Va / 2) * x**2
            - (w1 / 6) * MBa**3
            - (dw / (24 * c)) * MBa**4
            + (w2 / 6) * MBb**3
            + (dw / (24 * c)) * MBb**4
            + Ra
        )
        res.D = (
            (Va / 6) * x**3
            - (w1 / 24) * MBa**4
            - (dw / (120 * c)) * MBa**5
            + (w2 / 24) * MBb**4
            + (dw / (120 * c)) * MBb**5
            + Ra * x
        )

        res.V[0] = 0.0
        res.V[npts - 1] = 0.0
        res.M[0] = 0.0
        res.M[npts - 1] = 0.0

        return res


class LoadMaMb(Load):
    """
    Member end moment loads
    """

    def __init__(self, i_span, Ma, Mb):
        super().__init__(i_span)
        self.Ma = Ma
        self.Mb = Mb

    def get_cnl(self, L, eType) -> LoadCNL:
        """
        Returns the Consistent Nodal Loads for a span of length L of element eType

        Parameters
        ----------
        L : float
            The length of the member
        eType : int
            The member element type

        Returns
        -------
        LoadCNL
            Consistent Nodal Loads for this load type
        """

        Ma = self.Ma
        Mb = self.Mb

        cnl = LoadCNL(
            # Shears
            Va=(Ma + Mb) / L,
            Vb=-(Ma + Mb) / L,
            # Moments
            Ma=Ma,
            Mb=Mb,
        )
        return cnl

    def get_mbr_results(self, x: np.ndarray, L: float) -> MemberResults:
        """
        Results along the member from this load

        Parameters
        ----------
        x : np.ndarray
            Vector of points along the length of the member
        L : float
            The length of the member

        Returns
        -------
        res : MemberResults
            A populated :class:`pycba.load.MemberResults` object
        """

        npts = len(x)
        res = MemberResults(vals=None, n=npts)
        res.x = x

        Ma = self.Ma
        Mb = self.Mb

        Va = (Ma + Mb) / L
        Ra = Ma * L / 3 - Mb * L / 6

        res.V = Va * np.ones(npts)
        res.M = Va * x - Ma
        res.R = (Va / 2) * x**2 - Ma * x + Ra
        res.D = (Va / 6) * x**3 - (Ma / 2) * x**2 + Ra * x

        res.V[0] = 0.0
        res.V[npts - 1] = 0.0
        res.M[0] = 0.0
        res.M[npts - 1] = 0.0

        return res


class LoadML(Load):
    """
    Moment load applied at a along member
    """

    def __init__(self, i_span, M, a):
        super().__init__(i_span)
        self.M = M
        self.a = a

    def get_cnl(self, L, eType) -> LoadCNL:
        """
        Returns the Consistent Nodal Loads for a span of length L of element eType

        Parameters
        ----------
        L : float
            The length of the member
        eType : int
            The member element type

        Returns
        -------
        LoadCNL
            Consistent Nodal Loads for this load type
        """

        m = self.M
        a = self.a
        b = L - a

        cnl = LoadCNL(
            # Shears
            Va=6 * m * a * b / L**3,
            Vb=-6 * m * a * b / L**3,
            # Moments
            Ma=(m * b / L**2) * (2 * a - b),
            Mb=(m * a / L**2) * (2 * b - a),
        )
        return cnl

    def get_mbr_results(self, x: np.ndarray, L: float) -> MemberResults:
        """
        Results along the member from this load

        Parameters
        ----------
        x : np.ndarray
            Vector of points along the length of the member
        L : float
            The length of the member

        Returns
        -------
        res : MemberResults
            A populated :class:`pycba.load.MemberResults` object
        """

        npts = len(x)
        res = MemberResults(vals=None, n=npts)
        res.x = x

        m = self.M
        a = self.a
        b = L - a

        Va = m / L
        Ra = (m / 6) * (3 * b**2 / L - L)

        # # For moment we must insert an additional point to properly capture the step
        # idx = np.searchsorted(x, a)
        # if not np.any(np.isclose(x, a)):
        #     x = np.insert(x, idx, a)
        # x = np.insert(x, idx, a)
        # res.x = x

        res.V = Va * np.ones(npts)

        if a == 0:
            res.M = Va * x - m * self.H(x - a, 1)
        elif a == L:
            res.M = Va * x - m * self.H(x - a, 0)
        else:
            res.M = Va * x - m * self.H(x - a, 0.5)

        # res.M = np.array(
        #     [Va * xi - m if i > idx else Va * xi for i, xi in enumerate(x)]
        # )
        res.R = (Va / 2) * x**2 - m * self.MB(x - a) + Ra
        res.D = (Va / 6) * x**3 - (m / 2) * self.MB(x - a) ** 2 + Ra * x

        res.V[0] = 0.0
        res.V[npts - 1] = 0.0
        res.M[0] = 0.0
        res.M[npts - 1] = 0.0

        return res


def parse_LM(LM: LoadMatrix) -> List[Load]:
    """
    This function parses the Load Matrix and returns a list
    of Load objects

    **Note: span/member numbering converted to base-0 here**

    Parameters
    ----------
    LM : LoadMatrix
        The user-defined LoadMatrix

    Returns
    -------
    loads : List[Load]
        A list of Load objects
    """

    if not all(isinstance(load, list) for load in LM):
        raise ValueError("Load Matrix must be a list of lists")
    loads = []
    for load in LM:
        span = int(load[0] - 1)
        ltype = load[1]

        # UDL
        if ltype == 1:
            w = load[2]
            loads.append(LoadUDL(span, w))
        # Point load
        elif ltype == 2:
            P = load[2]
            a = load[3]
            loads.append(LoadPL(span, P, a))
        # Partial UDL
        elif ltype == 3:
            w = load[2]
            a = load[3]
            c = load[4]
            loads.append(LoadPUDL(span, w, a, c))
        # Moment Load
        elif ltype == 4:
            m = load[2]
            a = load[3]
            loads.append(LoadML(span, m, a))
        # Trapezoidal Load
        elif ltype == 5:
            w1 = load[2]
            w2 = load[3]
            a = load[4] if len(load) > 4 else 0.0
            c = load[5] if len(load) > 5 else None
            loads.append(LoadTrapez(span, w1, w2, a, c))
    return loads


def add_LM(LM1: LoadMatrix, LM2: LoadMatrix) -> LoadMatrix:
    """
    Adds two load matrices and returns the sum; this enables superposition

    Parameters
    ----------
    LM1 : LoadMatrix
        The first `LoadMatrix` object

    LM2 : LoadMatrix
        The second `LoadMatrix` object

    Returns
    -------
    LM : LoadMatrix
        The superimposed `LoadMatrix` object
    """

    LM = []
    for load in LM1:
        LM.append(load)
    for load in LM2:
        LM.append(load)

    return LM


def factor_LM(LM: LoadMatrix, gamma: float) -> LoadMatrix:
    """
    Applies a factor to the loads in a `LoadMatrix` object

    Parameters
    ----------
    LM : LoadMatrix
        The `LoadMatrix` object

    gamma : float
        A factor to apply to the load magnitudes

    Returns
    -------
    LM : LoadMatrix
        The factored `LoadMatrix` object
    """
    LMnew = []
    for load in LM:
        i_span = load[0]
        l_type = load[1]
        mag = gamma * load[2]
        if l_type == 1:  # UDL
            LMnew.append([i_span, l_type, mag])
        elif l_type == 2 or l_type == 4:  # PL or ML
            LMnew.append([i_span, l_type, mag, load[3]])
        elif l_type == 5:  # Trapezoidal
            new_load = [i_span, l_type, mag, gamma * load[3]]
            if len(load) > 4:
                new_load.extend(load[4:])  # a, c are not factored
            LMnew.append(new_load)
        else:  # PUDL
            LMnew.append([i_span, l_type, mag, load[3], load[4]])

    return LMnew
