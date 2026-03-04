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
Currently, each member is considered as prismatic.

**Dimension**: `N` x 1

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

A matrix representing the loads (i.e. a `List` of `Lists`). 
Each entry represents a single load and must be in the following format:

     Span No. | Load Type | Load Value | Distance a | Load Cover c
     
Load Types are: 

    1 - **Uniformly Distributed Loads**, which only have a load value; set distance `a` to "0".
    
    2 - **Point Loads**, located at `a` from the left end of the span.
    
    3 - **Partial UDLs**, starting at `a` for a distance of `c` (i.e. the cover) where $L >= a+c$.
    
    4 - **Moment Load**, located at `a`.
    
**Dimension**: `M` x 5, where `M` is the number of loads applied.

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


