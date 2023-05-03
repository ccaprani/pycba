"""
PyCBA - Load module

The load matrix represents the loads as a `List` of `Lists`.
Each list entry represents a single load and must be in the following format:

     Span No. | Load Type | Load Value | Distance a | Load Cover c

Load Types are:

    1 - **Uniformly Distributed Loads**, which only have a load value; distances `a` and `c` are set to "0".

    2 - **Point Loads**, located at `a` from the left end of the span; distances `c` is set to "0".

    3 - **Partial UDLs**, starting at `a` for a distance of `c` (i.e. the cover) where $L >= a+c$.

    4 - **Moment Load**, located at `a`; distances `c` is set to "0".

It has dimension `M` x 5, where `M` is the number of loads applied to the beam.

The type alias `LoadMatrix` is defined as

.. autodata:: LoadMatrix

"""

from __future__ import annotations
from typing import Union, List, NamedTuple, Tuple, Optional
import numpy as np

# Define a type alias
LoadMatrix = List[List[Union[int, float]]]


class LoadCNL(NamedTuple):
    """
    A typed namedtuple for Consistent Nodal Loads
    """

    Va: float
    Ma: float
    Vb: float
    Mb: float


# Would be nice to have this in results.py but it causes a circular reference
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
        """
        Construct the class with a tuple of the vectors of results along the member.

        Parameters
        ----------
        vals : Optional[Tuple[np.array, np.array, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], optional
            The tuple containing the vectors of results along the member `(x, M, V, R, D)`. The default is None.
        n : Optional[int], optional
            The length of each vector in the results tuple. The default is None.

        Raises
        ------
        ValueError
            Either of the function parameters must be provided.

        Returns
        -------
        None.
        """

        if vals is not None:
            (x, M, V, R, D) = vals
            self.n = len(x)
            self.x = x  # location of result values along member
            self.V = V  # Shear force
            self.M = M  # Bending moments
            self.R = R  # Rotations
            self.D = D  # Deflection/translation
        elif n is not None:
            self.n = n
            self._zero(n)
        else:
            raise ValueError("MemberResults requries either vals or n")

    def _zero(self, n: int):
        """
        Creates a zero arrays of results

        Parameters
        ----------
        n : int
            The number of entries for results along the member

        Returns
        -------
        None
        """

        self.x = np.zeros(n)
        self.M = np.zeros(n)
        self.V = np.zeros(n)
        self.R = np.zeros(n)
        self.D = np.zeros(n)

    def __add__(self, o: MemberResults):
        """
        Overload addition of :class:`pycba.load.MemberResults` objects to superimpose
        load effects.

        Parameters
        ----------
        o : MemberResults
            The other set of results for the member to be added to the current set

        Raises
        ------
        ValueError
            The results must be for the same member.

        Returns
        -------
        MemberResults
            An object containing the superimposed set of :class:`pycba.load.MemberResults`
        """

        # Test they are the same member
        np.testing.assert_equal(
            self.x, o.x, err_msg="Cannot superimpose results of different members"
        )

        # Do not superimpose distance
        x = self.x

        # Superimpose load effects
        M = self.M + o.M
        V = self.V + o.V
        R = self.R + o.V
        D = self.D + o.D

        return MemberResults(vals=(x, M, V, R, D))

    def apply_EI(self, EI: float):
        """
        Factors results by flexural rigidity after numerical integration of the
        bending moments for the displacements (rotations and translations).

        Parameters
        ----------
        EI : float
            The flexural rigidity

        Returns
        -------
        None
        """
        self.R /= EI
        self.D /= EI


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

    def get_cnl(self, L, eType):
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

    def released_end_forces(self, cnl: LoadCNL, L: float, eType: int) -> LoadCNL:
        """
        The released end forces for each element type: converts the Consistent Nodal
        Loads of the applied loading to the correct nodal loading depending on the
        element type.

        Parameters
        ----------
        cnl : LoadCNL
            The nodal loading statically consistent with the externally applied loads.
        L : float
            The length of the member.
        eType : int
            The element type.

        Returns
        -------
        LoadCNL
            The nodal loads to be applied in the analysis, consistent with the element
            type.

        """

        ref = np.zeros(4)
        fm = 6 / (4 * L)  # flexibility coeff for moment

        if eType == 2:  # DOF = moment at j node
            ref[0] = fm * cnl.Mb
            ref[1] = 0.5 * cnl.Mb
            ref[2] = -fm * cnl.Mb
            ref[3] = 1.0 * cnl.Mb
        elif eType == 3:  # DOF = moment at i node
            ref[0] = fm * cnl.Ma
            ref[1] = 0.5 * cnl.Ma
            ref[2] = -fm * cnl.Ma
            ref[3] = 1.0 * cnl.Ma
        elif eType == 4:  # keep only vertical, remove moments
            ref[0] = 0
            ref[1] = 1.0 * cnl.Va
            ref[2] = 0
            ref[3] = 1.0 * cnl.Va
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

        return self.released_end_forces(cnl, L, eType)

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
        # implicit conversion to tuple in correct order
        return self.released_end_forces(cnl, L, eType)

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
        return self.released_end_forces(cnl, L, eType)

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
        # implicit conversion to tuple in correct order
        return self.released_end_forces(cnl, L, eType)

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
        # implicit conversion to tuple in correct order
        return self.released_end_forces(cnl, L, eType)

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
        span = load[0] - 1
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
    return loads
