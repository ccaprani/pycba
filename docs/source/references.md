(references)=

# References

The source texts and papers underpinning the formulations used in `PyCBA` are
collected below: the matrix (direct) stiffness method and Euler–Bernoulli beam
element; the consistent nodal (fixed-end) load formulae; the non-prismatic
(variable-$EI$) element and imposed-curvature load; and the nonlinear
(Generalized Clough) plastic-hinge analysis. Each entry is hyperlinked to a DOI
or a stable URL, and is cross-referenced from the [Theoretical Basis](theory.md)
page by the labelled anchors shown.

## Matrix (direct) stiffness method and beam elements

(ref-mcguire-2000)=
**McGuire, W., Gallagher, R.H. & Ziemian, R.D. (2000).** *Matrix Structural
Analysis*, 2nd edn. John Wiley & Sons, New York. The companion *MASTAN2*
software and the full text are freely available at
<https://www.mastan2.com/textbook.html>. The standard graduate reference for the
direct stiffness method, element stiffness matrices, and global assembly.

(ref-przemieniecki-1968)=
**Przemieniecki, J.S. (1968).** *Theory of Matrix Structural Analysis*.
McGraw-Hill, New York. (Dover reprint, 1985, ISBN 978-0486649481.) Catalogue
record: <https://search.worldcat.org/title/911747>. The classic derivation of
element stiffness and consistent (work-equivalent) nodal load matrices from
energy principles.

(ref-weaver-gere-1990)=
**Weaver, W. & Gere, J.M. (1990).** *Matrix Analysis of Framed Structures*, 3rd
edn. Springer, New York.
[DOI: 10.1007/978-1-4684-7491-2](https://doi.org/10.1007/978-1-4684-7491-2).
Detailed treatment of beam and frame elements, fixed-end actions, support
displacements and the handling of member releases.

(ref-ghali-neville-brown-2017)=
**Ghali, A., Neville, A.M. & Brown, T.G. (2017).** *Structural Analysis: A
Unified Classical and Matrix Approach*, 7th edn. CRC Press, Boca Raton.
[DOI: 10.1201/9781315275215](https://doi.org/10.1201/9781315275215). Unified
classical (flexibility/slope-deflection) and matrix treatment, including
support settlement and temperature/curvature effects in indeterminate beams.

(ref-hibbeler-2017)=
**Hibbeler, R.C. (2017).** *Structural Analysis*, 10th edn. Pearson, Hoboken.
Catalogue record: <https://search.worldcat.org/title/953777155>. An accessible
undergraduate development of the stiffness method for beams, with extensive
worked examples of fixed-end moments and member releases.

(ref-timoshenko-young-1965)=
**Timoshenko, S.P. & Young, D.H. (1965).** *Theory of Structures*, 2nd edn.
McGraw-Hill, New York. Catalogue record:
<https://search.worldcat.org/title/1473595>. Classical reference for the
slope-deflection and force methods that underlie the released-element
formulation.

(ref-cook-2001)=
**Cook, R.D., Malkus, D.S., Plesha, M.E. & Witt, R.J. (2001).** *Concepts and
Applications of Finite Element Analysis*, 4th edn. John Wiley & Sons, New York.
Catalogue record: <https://search.worldcat.org/title/45286597>. Source for
consistent nodal loads, static condensation, and the imposition of boundary
conditions by direct elimination.

(ref-felippa-iffem)=
**Felippa, C.A. (2004).** *Introduction to Finite Element Methods* (lecture
notes), University of Colorado Boulder. Freely available at
<https://quickfield.com/advanced/felippa_introduction_to_FEM.pdf>. A clear,
openly-available reference for element stiffness, assembly, consistent nodal
loads and constraint handling.

(ref-caprani-sa4)=
**Caprani, C.C.** *Structural Analysis IV — The Matrix Stiffness Method*
(course notes). Available at
<http://www.colincaprani.com/structural-engineering/courses/structural-analysis-iv>.
The lecture notes from which the original CBA program was developed, covering the
beam element, assembly, and member-release condensation used in `PyCBA`.

## Consistent nodal loads (fixed-end forces)

(ref-roark-2020)=
**Budynas, R.G. & Sadegh, A.M. (2020).** *Roark's Formulas for Stress and
Strain*, 9th edn. McGraw-Hill, New York. Catalogue record:
<https://search.worldcat.org/title/1145991188>. Closed-form reactions, fixed-end
moments and deflections for point, uniform, partial and trapezoidal loadings.

(ref-aisc-manual)=
**American Institute of Steel Construction (2017).** *Steel Construction Manual*,
15th edn, "Beam Diagrams and Formulas". AISC, Chicago.
<https://www.aisc.org/publications/steel-construction-manual-resources/>.
Tabulated fixed-end moments and shears for the standard load types implemented in
`PyCBA`'s load module.

## Non-prismatic (variable-$EI$) elements and imposed curvature

(ref-ghali-2002)=
**Ghali, A., Favre, R. & Elbadry, M. (2002).** *Concrete Structures: Stresses
and Deformations*, 3rd edn. Spon Press, London.
[DOI: 10.1201/9781482271782](https://doi.org/10.1201/9781482271782). Chapter 13
develops the flexibility-integrated stiffness of a variable-rigidity member,
$[S^{*}] = [f^{*}]^{-1}$; Section 13.7 gives the corresponding imposed-curvature
(creep, shrinkage and thermal) fixed-end forces in continuous members.

(ref-hulse-mosley-1986)=
**Hulse, R. & Mosley, W.H. (1986).** *Reinforced Concrete Design by Computer*.
Macmillan Education, London.
[DOI: 10.1007/978-1-349-18496-4](https://doi.org/10.1007/978-1-349-18496-4).
Section 2.6 builds a pure-flexural haunched-beam element by Simpson-rule
flexibility integration, giving the stiffness and carry-over factors directly.

(ref-pca-frame-constants)=
**Portland Cement Association (1958).** *Handbook of Frame Constants: Beam
Factors and Moment Coefficients for Members of Variable Section*. PCA, Skokie,
IL. Catalogue record: <https://search.worldcat.org/title/3992779>.
Tabulated stiffness factors, carry-over factors and fixed-end-moment coefficients
for haunched and tapered (non-prismatic) members.

(ref-elbadry-ghali-1989)=
**Elbadry, M.M. & Ghali, A. (1989).** "Serviceability design of continuous
prestressed concrete structures." *PCI Journal* **34**(1), 54–91.
[DOI: 10.15554/pcij.01011989.54.91](https://doi.org/10.15554/pcij.01011989.54.91).
Restraint of creep, shrinkage and thermal (imposed) curvatures in continuous
members — the engineering context for `PyCBA`'s imposed-curvature load.

(ref-ghali-favre-1994)=
**Ghali, A. & Favre, R. (1994).** *Concrete Structures: Stresses and
Deformations*, 2nd edn. E & FN Spon, London. Catalogue record:
<https://search.worldcat.org/title/30894947>. Earlier edition of the
imposed-curvature / variable-rigidity formulation refined in the 3rd edition.

## Numerical integration

(ref-stroud-secrest-1966)=
**Stroud, A.H. & Secrest, D. (1966).** *Gaussian Quadrature Formulas*.
Prentice-Hall, Englewood Cliffs, NJ. Catalogue record:
<https://search.worldcat.org/title/1527634>. Reference for the Gauss–Legendre
rules used to evaluate the non-prismatic element flexibility integrals exactly
(for polynomial $EI$) or to machine precision.

(ref-scipy-2020)=
**Virtanen, P. et al. (2020).** "SciPy 1.0: fundamental algorithms for
scientific computing in Python." *Nature Methods* **17**, 261–272.
[DOI: 10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
Provides the cumulative-trapezoidal and Simpson integration
(`scipy.integrate.cumulative_trapezoid`, `scipy.integrate.simpson`) used for
member deflection recovery and the flexibility/curvature integrals.

(ref-harris-numpy-2020)=
**Harris, C.R. et al. (2020).** "Array programming with NumPy." *Nature*
**585**, 357–362.
[DOI: 10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
The linear-algebra back end (`numpy.linalg.solve`, `numpy.linalg.eigvalsh`,
`numpy.polynomial.legendre.leggauss`) used to solve the assembled system, run the
stability eigenvalue test, and generate the Gauss nodes.

## Stability / mechanism detection

(ref-golub-vanloan-2013)=
**Golub, G.H. & Van Loan, C.F. (2013).** *Matrix Computations*, 4th edn. Johns
Hopkins University Press, Baltimore. Catalogue record:
<https://search.worldcat.org/title/824733531>. Source for the symmetric
eigenvalue problem, condition number, and the singularity criterion used in the
free-DOF mechanism check.

(ref-bathe-2014)=
**Bathe, K.-J. (2014).** *Finite Element Procedures*, 2nd edn. Prentice Hall /
K.-J. Bathe, Watertown, MA. Freely available at
<https://web.mit.edu/kjb/www/Books/FEP_2nd_Edition_4th_Printing.pdf>. Establishes
that the free (unconstrained) stiffness partition of a stable structure is
symmetric positive-definite, and that a mechanism renders it singular.

## Nonlinear analysis and plastic-hinge models

(ref-clough-johnston-1966)=
**Clough, R.W. & Johnston, S.B. (1966).** "Effect of stiffness degradation on
earthquake ductility requirements." *Proceedings of the Japan Earthquake
Engineering Symposium*, Tokyo, pp. 227–232. Report record:
<https://search.worldcat.org/title/30950305>. The stiffness-degradation
("Clough") model underlying `PyCBA`'s concentrated-plasticity element.

(ref-li-li-2007)=
**Li, G.-Q. & Li, J.-J. (2007).** *Advanced Analysis and Design of Steel
Frames*. John Wiley & Sons, Chichester.
[DOI: 10.1002/9780470059715](https://doi.org/10.1002/9780470059715). Chapter 4
gives the interpolated element-stiffness degradation between fixed-fixed,
pinned-fixed and fixed-pinned matrices used in the nonlinear engine.

(ref-neal-1977)=
**Neal, B.G. (1977).** *The Plastic Methods of Structural Analysis*, 3rd edn.
Chapman & Hall, London.
[DOI: 10.1007/978-94-009-5764-6](https://doi.org/10.1007/978-94-009-5764-6).
Foundations of plastic-hinge theory, collapse mechanisms and the upper/lower
bound theorems.

## Plastic analysis (virtual work / collapse load factors)

(ref-baker-heyman-1956)=
**Baker, J.F., Horne, M.R. & Heyman, J. (1956).** *The Steel Skeleton, Volume 2:
Plastic Behaviour and Design*. Cambridge University Press, Cambridge. Catalogue
record: <https://search.worldcat.org/title/575393>. Foundational treatment of
plastic collapse of steel frames.

(ref-heyman-1971)=
**Heyman, J. (1971).** *Plastic Design of Frames, Volume 1: Fundamentals*.
Cambridge University Press, Cambridge.
[DOI: 10.1017/CBO9781139106740](https://doi.org/10.1017/CBO9781139106740).
Fundamentals of the plastic theorems used to interpret the collapse load factors.

(ref-horne-1979)=
**Horne, M.R. (1979).** *Plastic Theory of Structures*, 2nd edn. Pergamon Press,
Oxford. Catalogue record: <https://search.worldcat.org/title/4136860>. Concise
development of plastic-hinge collapse analysis.

## Bridge loading and moving-load analysis

(ref-mccarthy-2012)=
**McCarthy, L.A. (2012).** *Probabilistic Analysis of Indeterminate Highway
Bridges Considering Material Nonlinearity*. MPhil Thesis, Dublin Institute of
Technology.
[DOI: 10.21427/D7C30J](https://doi.org/10.21427/D7C30J). Application of the
nonlinear continuous-beam analysis to indeterminate highway bridges under moving
loads.

(ref-caprani-2006)=
**Caprani, C.C. (2006).** *Probabilistic Analysis of Highway Bridge Traffic
Loading*. PhD Thesis, University College Dublin. Available at
<http://www.colincaprani.com/files/Caprani%20PhD%20Thesis.pdf>. Background to the
influence-line and moving-load machinery in `PyCBA`.

## Prestressed concrete and equivalent loads

(ref-gilbert-mickleborough-ranzi-2017)=
**Gilbert, R.I., Mickleborough, N.C. & Ranzi, G. (2017).** *Design of
Prestressed Concrete to AS3600-2009*, 2nd edn. CRC Press, Boca Raton. ISBN
978-1466572690.
[Publisher record](https://www.routledge.com/Design-of-Prestressed-Concrete-to-AS3600-2009/Gilbert-Mickleborough-Ranzi/p/book/9781466572690).
The standard Australian text on prestressed concrete; **Example 11.1**
(continuous beam) derives the equivalent loads and the total, primary and
secondary moments induced by prestress — the worked example reproduced by the
`pycba.prestress` preprocessor.

(ref-ptdesigner-2000)=
**Structural Data Inc. (2000).** *PT Designer — Post-Tensioning Design and
Analysis Programs: Theory Manual.*
[PDF](https://secure.skghoshassociates.com/product/PT/download/TheoryManual.pdf).
Chapters 5–6 define the 12-profile tendon library (shared with RAPT) and the
equivalent-load (balanced-load) formulae implemented in `pycba.prestress`.
