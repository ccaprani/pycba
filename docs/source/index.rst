.. figure:: images/pycba_logo.png
   :alt: PyCBA logo
   :align: center
   :scale: 50

|Code style: black|
|License: MIT|
.. |version|
|Github issues|
|Github pull requests|
|PyPI|
..    |GitHub Workflow Deploy|
..    |GitHub Workflow Build|
..    |GitHub Workflow Status|
..    |GitHub commit activity|
..    |GitHub last commit|
..    |Contributor Covenant|
|codecov|

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg 
   :target: https://github.com/psf/black

.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg 
   :target: https://opensource.org/licenses/MIT

.. 
   |version| image:: https://img.shields.io/github/downloads/ccaprani/pycba/total?label=version

.. |GitHub issues| image:: https://img.shields.io/github/issues/ccaprani/ospgrillage?logoColor=yellowgreen

.. |GitHub pull requests| image:: https://img.shields.io/github/issues-pr/ccaprani/ospgrillage?color=yellowgreen

.. |PyPI| image:: https://img.shields.io/pypi/v/ospgrillage

.. |GitHub Workflow Deploy| image:: https://img.shields.io/github/workflow/status/ccaprani/pycba/Build%20and%20deploy

.. |GitHub Workflow Build| image:: https://img.shields.io/github/workflow/status/ccaprani/pycba/Deploy%20to%20GitHub%20Pages?label=gh%20page%20build

.. |GitHub Workflow Status| image:: https://img.shields.io/github/workflow/status/ccaprani/pycba/Tests?label=Tests

.. |GitHub commit activity| image:: https://img.shields.io/github/commit-activity/m/ccaprani/ospgrillage

.. |GitHub last commit| image:: https://img.shields.io/github/last-commit/ccaprani/ospgrillage?color=ff69b4

.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg 
   :target: code_of_conduct.md

.. |codecov| image:: https://codecov.io/gh/ccaprani/pycba/branch/main/graph/badge.svg?token=dUTOmPBnyP 
   :target: https://codecov.io/gh/ccaprani/pycba

Welcome to PyCBA's documentation!
=================================

`PyCBA` is a python implementation of the Continuous Beam Analysis program, `originally coded in Matlab <http://www.colincaprani.com/programming/matlab/>`_ and subsequently `ported to C++ <http://cbeam.sourceforge.net/>`_ (by Pierrot).

`PyCBA` is for fast linear elastic analysis of general beam configurations.
It uses the matrix stiffness method to determine the displacements at each node.
These are then used to determine the member end forces.
Exact expressions are then used to determine the distribution of shear, moment, and rotation along each member.
Cumulative trapezoidal integration is then used to determine the rotations and deflections along each member.
The program features:

-    Multiple load types: point load; uniformly distributed load; patch load, and; moment load;
-    Spring supports, both vertical and rotational, enabling it to be used as part of a subframe analysis;
-    Results are output at 100 (user can change) positions along each span, enable accurate deflection estimation.

One of the main functions of `PyCBA` is that the basic analysis engine forms the basis for higher-level analysis. 
Current `PyCBA` includes modules for:

- Influence line generation

- Moving load analysis for bridges, targeted at bridge access assessments 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials
   general
   api
   theory
..
   inf_lines 
   bridges
   ModuleDoc
   ChangeLog

Related Packages
================
- `anastruct <https://github.com/ritchie46/anaStruct>`_ is an analysis package for 2D frames and trusses with both linear and non-linear analysis capability.
- `sectionproperties <https://github.com/robbievanleeuwen/section-properties>`_ is a package for the analysis of cross-sectional geometric properties and stress distributions.
- `ospgrillage <https://github.com/ccaprani/ospgrillage>`_ is a bridge deck grillage analysis package which is a pre-processor for `OpenSeesPy <https://github.com/zhuminjie/OpenSeesPy>`_, a python wrapper for the general finite element analysis framework `OpenSees <https://github.com/OpenSees/OpenSees>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
