Beam Configuration
==================

The beam configuration input data is to be in the following format:

Member Lengths (`L`)
--------------------

A vector of the lengths of each member.

**Dimension**: `N` x 1, where `N` is the number of members.

**Units**: m

Flexural Rigidity (`EI`)
------------------------

The flexural rigidity for each member.

A member may be **prismatic** (constant `EI`, given as a scalar) or
**non-prismatic** (variable `EI`). A non-prismatic member is defined with a
`pycba.SectionEI` object built from one or more contiguous **segments** that
together describe `EI(x)` over the span. `x` is the **span-local physical
coordinate** (0 at the span start, the real span length at the end). Segments
are added head-to-tail with `add_segment(seg_type, x, ei, degree=None)`
(chainable), or in one line by passing a list of specs to the constructor:

.. code-block:: python

   import pycba as cba
   # Straight haunch -> flat soffit -> straight haunch over a 12 m span:
   sec = (cba.SectionEI()
          .add_segment("linear", [0.0, 3.0], [3.0e5, 1.2e5])
          .add_segment("const",  [3.0, 9.0], 1.2e5)
          .add_segment("linear", [9.0, 12.0], [1.2e5, 3.0e5]))

The `seg_type` is one of:

- `"const"` — constant `EI` over `[x0, x1]`; `x = [x0, x1]`, `ei` scalar.
- `"linear"` — one linear piece; `x = [x0, x1]`, `ei = [ei0, ei1]`.
- `"pwl"` — piecewise-linear; `x = [x0, …, xn]` (n ≥ 2 stations), `ei` of the
  same length, giving `n − 1` linear pieces with kinks at the interior
  stations.
- `"poly"` — one polynomial piece; `ei` is either sample values (a polynomial
  of order `degree`, default `len(x) − 1`, is fitted) or a `callable`
  `ei(x_local)` evaluated in the span-local coordinate (e.g. a parabolic soffit,
  `EI ∝ depth³`).

Segments must be **contiguous** (each new segment's `x[0]` equals the running
end; the first starts at 0) — gaps/overlaps raise an error — and the total
coverage must equal the span length. A coincident `x` carrying a different `ei`
across a join is an allowed **step** (discontinuous `EI`).

Non-prismatic members are analysed exactly by flexibility (force-method)
integration of the element stiffness, evaluated **piece-by-piece between
consecutive breakpoints** (segment joins and `pwl` kinks) so that kinks and
steps are captured exactly; a single `const` segment reproduces the closed-form
prismatic element to machine precision. Scalar and `SectionEI` members may be
freely mixed in the same beam.

**Dimension**: `N` x 1 (one scalar or `SectionEI` per span; a single value is
applied to all spans)

**Units**: kNm2

Restraints (`R`)
----------------

A vector of restraints for each node, as defined by the ends of each member.
Each node has 2 degrees of freedom, vertical deflection and rotation, in that order, for each node.
Restraint for a degree of freedom is indicated by a "-1" value.
Unrestrained degrees of freedom are indicated by a "0" value.
Supports with a stiffness (kN/m or kNm/rad) are indicated by a positive value of the stiffness, `k`: i.e. "+k"

**Dimension**: 2`N` x 1

**Units**: kN/m or kNm/rad or None

Load Matrix (`LM`)
------------------

A `List` of `Lists` representing the applied loads.
Each entry is a single load descriptor whose length depends on the load type:

.. list-table::
   :header-rows: 1
   :widths: 8 24 34 10

   * - Type
     - Name
     - Format
     - Columns
   * - 1
     - UDL
     - ``[span, 1, w]``
     - 3
   * - 2
     - Point Load
     - ``[span, 2, P, a]``
     - 4
   * - 3
     - Partial UDL
     - ``[span, 3, w, a, c]``
     - 5
   * - 4
     - Moment Load
     - ``[span, 4, M, a]``
     - 4
   * - 5
     - Trapezoidal (full)
     - ``[span, 5, w1, w2]``
     - 4
   * - 5
     - Trapezoidal (partial)
     - ``[span, 5, w1, w2, a, c]``
     - 6
   * - 6
     - Imposed curvature
     - ``[span, 6, k0, k1, ...]``
     - 3+

Load Types:

    1 - **Uniformly Distributed Loads**, which only have a load value.

    2 - **Point Loads**, located at `a` from the left end of the span.

    3 - **Partial UDLs**, starting at `a` for a distance of `c` (i.e. the cover) where :math:`L \geq a + c`.

    4 - **Moment Load**, located at `a`.

    5 - **Trapezoidal Load**, linearly varying from `w1` to `w2`.
        Full span: `[span, 5, w1, w2]` — `w1` at the left end, `w2` at the right end.
        Partial:   `[span, 5, w1, w2, a, c]` — `w1` at position `a`, `w2` at position `a + c`, where :math:`L \geq a + c`.

    6 - **Imposed Curvature** (initial-strain) load, applying a free curvature
        field :math:`\kappa(x) = k_0 + k_1 x + \dots` along the member. On a
        statically-determinate span it induces no internal forces, only a free
        deflected shape; on a restrained or continuous structure its restraint
        generates real moments and reactions. This is the mechanism for
        applying creep, shrinkage and thermal curvatures (e.g. for
        prestressed-concrete time-dependent analysis).

**Dimension**: `M` rows (one per applied load), with 3 or more columns per row depending on load type.

**Units**: kN, kN/m, and metres.

Prescribed Displacements (`D`)
------------------------------

An optional vector of prescribed nodal displacements (settlements), one entry per degree of freedom.
Use `None` for DOFs where the displacement is unknown (the default), and a float for DOFs whose displacement is known (e.g. a support settlement).

- Fixed supports (`R = -1`) default to zero displacement unless overridden by `D`.
- Spring supports (`R > 0`) can also have a prescribed displacement; in that case the spring force is `k_s × δ` and is reported in `beam_results.Rs`.
- **Constraint**: a DOF cannot simultaneously have a spring (`R > 0`), a prescribed displacement (`D[i] ≠ None`), *and* a non-zero consistent nodal load — this combination is physically inconsistent and `analyze()` will raise a `ValueError`.

**Dimension**: 2`N+2` x 1 (same length as `R`)

**Units**: m (vertical DOFs), rad (rotational DOFs)

Element Types (`eleType`)
-------------------------

Each member can be one of several element types, depending on the presence of hinges in the beam.

**Note that at a hinge, only one of the members meeting at that node should have a pinned end.**

The element types are given by an index:

    1 - fixed-fixed
    
    2 - fixed-pinned
    
    3 - pinned-fixed
    
    4 - pinned-pinned

**Dimension**: `N` x 1

**Units**: N/A


