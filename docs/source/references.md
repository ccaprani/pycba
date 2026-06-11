(references)=

# References

The source texts for the formulations used in `PyCBA` are listed below — the
matrix stiffness method, the non-prismatic element and imposed-curvature load,
and the nonlinear (Generalized Clough) analysis.

## Matrix stiffness method

- Weaver, W. & Gere, J.M. (1990). *Matrix Analysis of Framed Structures*, 3rd ed. Springer.
- Ghali, A., Neville, A.M. & Brown, T.G. (2017). *Structural Analysis: A Unified Classical and Matrix Approach*, 7th ed. CRC Press.
- Timoshenko, S.P. & Young, D.H. (1965). *Theory of Structures*, 2nd ed. McGraw-Hill.
- McGuire, W., Gallagher, R.H. & Ziemian, R.D. (2000). *Matrix Structural Analysis*, 2nd ed. John Wiley & Sons.

## Non-prismatic elements and imposed curvature

(ref-ghali-2002)=
**Ghali, A., Favre, R. and Elbadry, M. (2002).** *Concrete Structures: Stresses
and Deformations*, 3rd edn, Spon Press, London.
Chapter 13 develops the flexibility-integrated stiffness of a variable-rigidity
member, $[S^{*}] = [f^{*}]^{-1}$; Section 13.7 gives the corresponding
imposed-curvature (thermal) fixed-end forces.

(ref-hulse-mosley-1986)=
**Hulse, R. and Mosley, W. H. (1986).** *Reinforced Concrete Design by
Computer*, Macmillan, London.
Section 2.6 builds a pure-flexural haunched-beam element by Simpson-rule
flexibility integration, giving the stiffness and carry-over factors directly.

(ref-pca-frame-constants)=
**Portland Cement Association.** *Handbook of Frame Constants*, PCA, Skokie, IL.
Tabulated stiffness factors, carry-over factors and fixed-end-moment
coefficients for haunched and tapered (non-prismatic) members.

## Nonlinear analysis and plastic hinge models

- Clough, R.W. & Johnston, S.B. (1966). "Effect of stiffness degradation on earthquake ductility requirements." *Proc. Japan Earthquake Engineering Symposium*, Tokyo, pp. 227–232.
- Li, G.Q. & Li, J.J. (2007). *Advanced Analysis and Design of Steel Frames*. John Wiley & Sons, Chapter 4.
- Neal, B.G. (1977). *The Plastic Methods of Structural Analysis*, 3rd ed. Chapman & Hall.

## Plastic analysis (virtual work / collapse load factors)

- Baker, J., Horne, M.R. & Heyman, J. (1956). *The Steel Skeleton*, Vol. 2: Plastic Behaviour and Design. Cambridge University Press.
- Heyman, J. (1971). *Plastic Design of Frames*, Vol. 1: Fundamentals. Cambridge University Press.
- Horne, M.R. (1979). *Plastic Design of Low-Rise Frames*. Granada.

## Bridge loading and moving load analysis

- McCarthy, L.A. (2012). *Probabilistic Analysis of Indeterminate Highway Bridges Considering Material Nonlinearity*. MPhil Thesis, Dublin Institute of Technology.
- Caprani, C.C. (2006). "Probabilistic Analysis of Highway Bridge Traffic Loading." PhD Thesis, University College Dublin.
