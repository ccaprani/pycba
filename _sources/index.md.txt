```{image} images/pycba_logo.png
:alt: PyCBA logo
:align: center
:scale: 50
```

[![PyPI](https://img.shields.io/pypi/v/pycba.svg?color=blue)](https://pypi.org/project/pycba/)
![Python versions](https://img.shields.io/pypi/pyversions/pycba.svg)
[![Tests](https://github.com/ccaprani/pycba/actions/workflows/pytest.yml/badge.svg)](https://github.com/ccaprani/pycba/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/ccaprani/pycba/branch/main/graph/badge.svg?token=dUTOmPBnyP)](https://codecov.io/gh/ccaprani/pycba)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Welcome to PyCBA's documentation!

`PyCBA` is a focused, dependable **1-D continuous-beam engine** built on the matrix (direct)
stiffness method — fast, exact, and a pleasure to use, with light dependencies (numpy / scipy /
matplotlib). It is deliberately *not* a general 2-D/3-D FE package; instead its analysis core
powers a surprisingly broad toolkit for the **design, assessment and teaching** of buildings and
bridges.

**Analysis.** Continuous beams of any number of spans with pin / roller / fixed / spring and
rotational-spring supports, prescribed settlements and internal hinges; an element library of
Euler–Bernoulli, **Timoshenko** (shear-deformable), **non-prismatic** (variable `EI`) and
**Winkler-foundation** members; **nonlinear** elasto-plastic analysis to collapse; and
**free-vibration (modal)** analysis.

**Loads & bridges.** UDL, point, partial, trapezoidal, moment and **imposed-curvature**
(creep / shrinkage / thermal) loads, load cases and combinations; **moving-load bridge
assessment** with influence lines, envelopes, coincident effects and shear points; and built-in
**code load models, road & rail, from six nations** (HL-93, LM1 / LM71, HB, CL-625, JTG,
Cooper E, M1600 / 300LA, T44 / MS18).

**Visualisation.** Beam and load schematics (matplotlib *and* publication-quality TikZ /
`stanli`), shaded result diagrams, reaction / coincident-effect / mode-shape /
collapse-mechanism / vehicle plots, an interactive **Plotly** backend, and selectable display
unit systems.

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
tutorials
general
api
theory
references
```

## Related Packages

- [anastruct](https://github.com/ritchie46/anaStruct) is an analysis package for 2D frames and trusses with both linear and non-linear analysis capability.
- [sectionproperties](https://github.com/robbievanleeuwen/section-properties) is a package for the analysis of cross-sectional geometric properties and stress distributions.
- [ospgrillage](https://github.com/ccaprani/ospgrillage) is a bridge deck grillage analysis package which is a pre-processor for [OpenSeesPy](https://github.com/zhuminjie/OpenSeesPy), a python wrapper for the general finite element analysis framework [OpenSees](https://github.com/OpenSees/OpenSees).

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
