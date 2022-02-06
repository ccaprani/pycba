# PyCBA - Python Continuous Beam Analysis

![PyCBA logo](https://raw.githubusercontent.com/ccaprani/pycba/main/docs/source/images/pycba_logo.png)

`PyCBA` is a python implementation of the Continuous Beam Analysis program, [originally coded in Matlab](http://www.colincaprani.com/programming/matlab/) and subsequently [ported to C++](http://cbeam.sourceforge.net/) (by Pierrot).

`PyCBA` is for fast linear elastic analysis of general beam configurations. It uses the matrix stiffness method to determine the displacements at each node. These are then used to determine the member end forces. Exact expressions are then used to determine the distribution of shear, moment, and rotation along each member. Cumulative trapezoidal integration is then used to determine the rotations and deflections along each member. The program features:

- Multiple load types: point load; uniformly distributed load; patch load, and; moment load;
- Spring supports, both vertical and rotational, enabling it to be used as part of a subframe analysis;
- Results are output at 100 (user can change) positions along each span, enable accurate deflection estimation.

One of the main functions of `PyCBA` is that the basic analysis engine forms the basis for higher-level analysis. Current `PyCBA` includes modules for:

- Influence line generation
- Moving load analysis for bridges, targeted at bridge access assessments
