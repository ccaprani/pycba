<p align="center">
  <img src="https://raw.githubusercontent.com/ccaprani/pycba/main/docs/source/images/pycba_logo.png" alt="PyCBA logo" width="320">
</p>

<h1 align="center">PyCBA — Python Continuous Beam Analysis</h1>

<p align="center">
  <em>Fast, accurate continuous-beam analysis for the design, assessment and teaching of buildings and bridges —<br>
  from linear statics to plastic collapse, modal dynamics and moving-load bridge assessment.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/pycba/"><img src="https://img.shields.io/pypi/v/pycba.svg?color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pycba/"><img src="https://img.shields.io/pypi/pyversions/pycba.svg" alt="Python versions"></a>
  <a href="https://github.com/ccaprani/pycba/actions/workflows/pytest.yml"><img src="https://github.com/ccaprani/pycba/actions/workflows/pytest.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/ccaprani/pycba"><img src="https://codecov.io/gh/ccaprani/pycba/branch/main/graph/badge.svg?token=dUTOmPBnyP" alt="codecov"></a>
  <a href="https://ccaprani.github.io/pycba/"><img src="https://img.shields.io/badge/docs-ccaprani.github.io%2Fpycba-blue" alt="Documentation"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ccaprani/pycba/main/docs/source/images/hero_results.png" alt="PyCBA result diagrams" width="640">
</p>

`PyCBA` is a focused, dependable **1-D continuous-beam engine** built on the matrix (direct)
stiffness method. It is deliberately *not* a general 2-D/3-D FE package — instead it does one
thing extremely well, with a clean API, light dependencies (**numpy / scipy / matplotlib**),
and an analysis core that powers a surprisingly broad toolkit.

```python
import pycba as cba

# A two-span continuous beam: 10 m + 12 m, EI = 30 000 kN·m², pinned supports
beam = cba.BeamAnalysis(L=[10, 12], EI=30_000, supports=["pin", "pin", "pin"])
beam.add_udl(i_member=1, w=20)         # 20 kN/m on span 1
beam.add_pl(i_member=2, p=50, a=6)     # 50 kN point load at mid-span 2
beam.analyze()

beam.plot_results()                    # bending moment, shear, deflection + reactions
print(beam.at(5.0))                    # -> {'M': 162.5, 'V': -17.5, 'R': 0.0, 'D': -0.05}
```

## ✨ What it does

**Analysis**
- Continuous beams of any number of spans, with **pin / roller / fixed / vertical-spring / rotational-spring** supports, **prescribed settlements**, and internal **hinges** — plus pre-solve **mechanism detection**.
- An **element library**: Euler–Bernoulli, **Timoshenko** (shear-deformable), **non-prismatic** (variable `EI`), and **beam-on-Winkler-foundation**.
- **Nonlinear** elasto-plastic analysis **to collapse** (plastic hinges, mechanism detection, collapse-mechanism plots).
- **Free-vibration (modal)** analysis — natural frequencies, periods and mode shapes.

**Loads & bridges**
- UDL, point, partial, **trapezoidal**, moment and **imposed-curvature** (creep / shrinkage / thermal) loads; **load cases, combinations** and patterned UDLs.
- **Moving-load bridge assessment** — influence lines, envelopes, **coincident effects**, lane UDLs and **shear points / critical shear**.
- **Code load models — road & rail, from six nations** — AASHTO **HL-93**, Eurocode **LM1 / LM71**, BS 5400 / CS 454 **HB**, CSA **CL-625**, China **JTG**, AREA **Cooper E**, AS 5100 **M1600 / S1600 / 300LA**, NAASRA **T44 / MS18** — organised by region.
- A **post-tensioning** preprocessor for prestress.

**Visualisation**
- Beam & load **schematics** in matplotlib *and* publication-quality **TikZ / `stanli`**.
- **Shaded** bending-moment / shear / deflection diagrams, **reaction** plots, **coincident-effects**, **mode-shape**, **collapse-mechanism** and **vehicle** plots.
- An **interactive Plotly** backend, and selectable **display unit systems** (SI / US / N·mm).

**Ergonomics**
- Point queries `at(x)`, exports `to_dataframe()` / `to_csv()`, a friendly `supports=` API, and clear errors — all validated against closed-form solutions, with 370+ tests.

## 📦 Installation

```bash
pip install pycba                 # core
pip install "pycba[plotly]"       # + interactive plots
```

Requires Python 3.9+.

## 📚 Documentation & tutorials

Full documentation, a Theoretical Basis, and a dozen worked-example notebooks (bridges, foundations,
modal, non-prismatic, nonlinear collapse, creep/shrinkage/thermal, the vehicle library, and more) are at
**[ccaprani.github.io/pycba](https://ccaprani.github.io/pycba/)**.

## 🌱 Origins

`PyCBA` began life as a Python port of Colin Caprani's MATLAB
[Continuous Beam Analysis](http://www.colincaprani.com/programming/matlab/) program (later
[ported to C++](http://cbeam.sourceforge.net/) by Pierrot). It has since grown well beyond that
starting point into the toolkit above — but the spirit is unchanged: fast, exact, and a pleasure to use.

## 📄 License

Released under the **Apache 2.0** license. Contributions and issues are welcome on
[GitHub](https://github.com/ccaprani/pycba).
