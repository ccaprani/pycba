(theory)=

# Theoretical Basis

`PyCBA` analyses continuous beams by the **matrix (direct) stiffness
method** — the standard displacement-based formulation of linear structural
analysis
([McGuire, Gallagher & Ziemian, 2000](ref-mcguire-2000);
[Przemieniecki, 1968](ref-przemieniecki-1968);
[Weaver & Gere, 1990](ref-weaver-gere-1990)).
The structure is idealised as an assembly of two-node Euler–Bernoulli beam
elements; the element stiffness relations are assembled into a global system,
the supports are imposed, the system is solved for the nodal displacements, and
the support reactions and member load effects are recovered. This page derives
each step, in the order the implementation performs it, so that every equation
corresponds directly to the code in
{class}`~pycba.analysis.BeamAnalysis`, {class}`~pycba.beam.Beam`,
{mod}`pycba.load` and {class}`~pycba.section.SectionEI`.

The overall procedure is:

1. **Elements** — each span is a two-node Euler–Bernoulli beam element whose
   $4\times4$ stiffness matrix depends on its end releases and, for a
   non-prismatic member, on the variation of $EI$ along its length.
2. **Global stiffness matrix** — the element matrices are assembled by the
   direct stiffness method into the global matrix $\mathbf{K}$, with elastic
   spring supports added on the diagonal.
3. **Loads** — span loads are converted to consistent (equivalent) nodal forces
   that assemble into the force vector $\mathbf{f}$.
4. **Stability check** — the free-DOF partition of $\mathbf{K}$ is tested for a
   mechanism before the system is solved.
5. **Boundary conditions** — fixed supports, prescribed settlements and spring
   supports are imposed.
6. **Solution and recovery** — the system $\mathbf{K}\,\mathbf{d}=\mathbf{f}$ is
   solved for the nodal displacements $\mathbf{d}$, from which the **reactions**
   and the **member results** (bending moment, shear, rotation and deflection
   along each span) are recovered.

```{note}
By default `PyCBA` uses **Euler–Bernoulli** (engineer's) beam theory: plane
sections remain plane and normal to the neutral axis, so **shear deformation is
neglected**. A member may instead be made a **Timoshenko** (shear-deformable)
element by supplying a finite transverse shear rigidity `GAv` (see
[Timoshenko elements](theory-timoshenko) below); this is opt-in per member and
the Euler–Bernoulli path is otherwise unchanged. Axial deformation is omitted in
both cases — the elements carry transverse (bending and, optionally, shear)
action only, which is the appropriate idealisation for a continuous beam.
```

## Sign conventions and degrees of freedom

`PyCBA` is **unit-agnostic**: the solver performs no unit conversions, and any
internally consistent set of units may be used (see the {doc}`general` page).
The sign conventions used throughout are:

* Vertical displacements $v$ and vertical forces (reactions) are **positive
  upward**.
* Rotations $\theta$ and moments are **positive counter-clockwise**.
* A **downward** applied load (UDL $w$, point load $P$) is entered with a
  **positive** magnitude; a positive applied moment is counter-clockwise.
* A support **settlement** is a prescribed (negative = downward) displacement.

Each node carries two degrees of freedom (DOF) — a transverse displacement $v$
and a rotation $\theta$ — so a single element with end nodes $i$ and $j$ has the
four local DOF, ordered

$$
\mathbf{d}^{(e)} =
\begin{bmatrix} v_i & \theta_i & v_j & \theta_j \end{bmatrix}^{\mathsf T},
$$

and a conjugate force vector $\mathbf{f}^{(e)} = [\,V_i,\;M_i,\;V_j,\;M_j\,]^{\mathsf T}$
of end shears and moments. This is the classical Hermitian beam element of the
matrix stiffness method
([McGuire, Gallagher & Ziemian, 2000](ref-mcguire-2000);
[Weaver & Gere, 1990](ref-weaver-gere-1990)).

## Elements

### The prismatic Euler–Bernoulli beam element

For a prismatic member of constant flexural rigidity $EI$ and length $L$,
Euler–Bernoulli theory relates the transverse displacement $v(x)$ to the bending
moment by $EI\,v''(x) = M(x)$, with no distributed load on the element interior
giving $v(x)$ as a cubic. Using the cubic Hermite shape functions
$\mathbf{N}(x)$ that interpolate the four end DOF, the element stiffness matrix
follows from the strain-energy (virtual-work) integral

$$
\mathbf{k}^{(e)} = \int_0^L EI\,\mathbf{B}(x)^{\mathsf T}\mathbf{B}(x)\,dx,
\qquad
\mathbf{B}(x) = \mathbf{N}''(x),
$$

which, for constant $EI$, evaluates to the well-known $4\times4$ matrix
([Przemieniecki, 1968](ref-przemieniecki-1968);
[Cook et al., 2001](ref-cook-2001)). In `PyCBA`'s DOF order and
sign convention (implemented in {meth}`pycba.beam.Beam.k_FF`):

$$
\mathbf{k}_{\text{FF}} = \frac{EI}{L^3}
\begin{bmatrix}
12 & 6L & -12 & 6L \\
6L & 4L^2 & -6L & 2L^2 \\
-12 & -6L & 12 & -6L \\
6L & 2L^2 & -6L & 4L^2
\end{bmatrix}.
$$

This is the **fixed–fixed** (Type 1, the default) element: both ends transmit
moment. The matrix is symmetric and positive semi-definite — it has a
two-dimensional null space corresponding to the rigid-body translation and
rotation of the unconstrained element, which is removed once the element is
assembled and supports are applied.

### Element types and moment releases

A continuous beam may contain internal **hinges** (moment releases). `PyCBA`
encodes this per span through the `eletype` index, which controls which end(s)
of the element carry moment:

| `eletype` | Name | Released end(s) | Method |
| --- | --- | --- | --- |
| 1 | Fixed–Fixed (FF) | none (default) | {meth}`~pycba.beam.Beam.k_FF` |
| 2 | Fixed–Pinned (FP) | right end ($j$) | {meth}`~pycba.beam.Beam.k_FP` |
| 3 | Pinned–Fixed (PF) | left end ($i$) | {meth}`~pycba.beam.Beam.k_PF` |
| 4 | Pinned–Pinned (PP) | both ends | {meth}`~pycba.beam.Beam.k_PP` |

A moment release is a known **zero internal moment** at that end. The
corresponding rotational DOF is therefore not connected to the element stiffness
and is **statically condensed** out: setting the released end moment to zero and
eliminating its rotation gives the reduced element stiffness. For the prismatic
element these condensations have the closed forms below (with the released
rotational rows and columns zeroed).

**Type 2 — Fixed–Pinned** (pin at the $j$-end), {meth}`~pycba.beam.Beam.k_FP`:

$$
\mathbf{k}_{\text{FP}} = \frac{3EI}{L^3}
\begin{bmatrix}
1 & L & -1 & 0 \\
L & L^2 & -L & 0 \\
-1 & -L & 1 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}.
$$

**Type 3 — Pinned–Fixed** (pin at the $i$-end), {meth}`~pycba.beam.Beam.k_PF`:

$$
\mathbf{k}_{\text{PF}} = \frac{3EI}{L^3}
\begin{bmatrix}
1 & 0 & -1 & L \\
0 & 0 & 0 & 0 \\
-1 & 0 & 1 & -L \\
L & 0 & -L & L^2
\end{bmatrix}.
$$

**Type 4 — Pinned–Pinned** (pin at both ends), {meth}`~pycba.beam.Beam.k_PP`:
with no moment transmitted at either end the element has no transverse bending
stiffness, so

$$
\mathbf{k}_{\text{PP}} = \mathbf{0}_{4\times4}.
$$

```{warning}
A moment release models an internal hinge *within* the elements, not a support
condition. At any joint where a hinge is required, **exactly one** of the two
members meeting there should carry the pin; if both adjacent ends are released
the rotation at that node is unrestrained by either element and the assembled
stiffness becomes singular — a mechanism. `PyCBA` detects this before solving
(see [Stability check](theory-stability)).
```

### Non-prismatic (variable-$EI$) elements

A *non-prismatic* member has a flexural rigidity $EI(x)$ that varies along its
length (a haunch, a taper, or a stepped section). `PyCBA` represents the
variation with a {class}`~pycba.section.SectionEI` object built from contiguous
**segments** (`const`, `linear`, `pwl`, `poly`) describing $EI(x)$ in the
span-local coordinate $x$, and analyses the member with a flexibility-integrated
element ({meth}`pycba.beam.Beam.k_nonprismatic`). This follows the classical
**force (flexibility) method** for variable-rigidity members
([Ghali, Favre & Elbadry, 2002, Ch. 13](ref-ghali-2002);
[Hulse & Mosley, 1986, §2.6](ref-hulse-mosley-1986); and the
idiom tabulated in the PCA
[*Handbook of Frame Constants*](ref-pca-frame-constants)).

Release the span to a simply-supported beam and apply unit moments at each end.
The resulting (linear) unit-moment diagrams are

$$
m_i(x) = 1 - \frac{x}{L}, \qquad m_j(x) = -\frac{x}{L},
$$

where the sign of $m_j$ carries the counter-clockwise-positive nodal-moment
convention used throughout (so that the formulation reproduces
$\mathbf{k}_{\text{FF}}$ exactly in the constant-$EI$ limit). By the unit-load
theorem the $2\times2$ end-rotation **flexibility** about the two end DOF is

$$
\mathbf{F} =
\begin{bmatrix} f_{ii} & f_{ij} \\ f_{ij} & f_{jj} \end{bmatrix},
\qquad
f_{pq} = \int_0^L \frac{m_p(x)\,m_q(x)}{EI(x)}\,dx,
$$

and the end **moment–rotation stiffness** is its inverse,

$$
\mathbf{K}_\theta = \mathbf{F}^{-1}
$$

(implemented in {meth}`pycba.beam.Beam.k_theta`). The diagonal terms of
$\mathbf{K}_\theta$ are the rotational *stiffness factors* (end moment per unit
rotation, far end fixed), and the off-diagonal term yields the *carry-over
factors* — exactly the quantities tabulated for haunched members in the PCA
handbook.

The full $4\times4$ element stiffness is recovered by expanding the two end
rotations to the four nodal DOF, removing the rigid-body chord rotation
$\psi = (v_j - v_i)/L$. With the kinematic transformation
$[\theta_i,\;\theta_j]^{\mathsf T} = \mathbf{T}\,\mathbf{d}^{(e)}$ where

$$
\mathbf{T} =
\begin{bmatrix}
1/L & 1 & -1/L & 0 \\
1/L & 0 & -1/L & 1
\end{bmatrix},
$$

the element stiffness is

$$
\mathbf{k}^{(e)} = \mathbf{T}^{\mathsf T}\,\mathbf{K}_\theta\,\mathbf{T}.
$$

The end shears emerge automatically as $(M_i + M_j)/L$ from this transformation,
matching `PyCBA`'s DOF order and sign convention. Moment releases (types 2–4)
are then imposed by the **same static condensation** of the released rotational
DOF used for the prismatic element (the Schur complement
$\mathbf{k}_{kk} - \mathbf{k}_{kr}\,\mathbf{k}_{rr}^{-1}\,\mathbf{k}_{rk}$ over
the released rows/columns; {meth}`pycba.beam.Beam._condense`).

The flexibility integrals are evaluated by **breakpoint-aware Gauss–Legendre
quadrature** ([Stroud & Secrest, 1966](ref-stroud-secrest-1966);
`numpy.polynomial.legendre.leggauss`,
[Harris et al., 2020](ref-harris-numpy-2020)), summed
piece-by-piece *between consecutive breakpoints* (segment joins and `pwl`
kinks). Splitting at the breakpoints makes a slope change (haunch → flat) or a
step (a discontinuous $EI$) **exact** rather than smeared by a single global
quadrature. On a constant piece the integrand $m_p m_q / EI$ is a pure quadratic
polynomial and a 2-point Gauss rule is exact, so a single `const` segment (the
prismatic limit) reproduces the closed-form $\mathbf{k}_{\text{FF}}$,
$\mathbf{k}_{\text{FP}}$, $\mathbf{k}_{\text{PF}}$ and $\mathbf{k}_{\text{PP}}$
above to machine precision. Scalar (prismatic) and `SectionEI` (non-prismatic)
members may be freely mixed in one beam.

(theory-timoshenko)=

### Timoshenko (shear-deformable) elements

A **Timoshenko** member augments the bending deformation with a transverse
**shear deformation** ([Timoshenko, 1921](ref-timoshenko-1921)): the cross
section rotates by $\psi(x)$, which is no longer constrained to remain normal to
the deflected axis, and the difference between the axis slope and the section
rotation is the shear strain

$$
\gamma = \frac{dw}{dx} - \psi = \frac{V}{GA_v},
$$

where $GA_v$ is the transverse **shear rigidity** ($A_v = kA$ the shear area,
with $k$ the cross-section shear coefficient,
[Cowper, 1966](ref-cowper-1966)). A member becomes a Timoshenko element purely
by being given a finite `GAv` (a scalar, or a {class}`~pycba.section.SectionEI`
for a variable $GA_v(x)$); the nodal DOF are unchanged — two per node,
$[v,\ \theta]$, with $\theta$ now the **section rotation** $\psi$ — so assembly,
supports, reactions and the plotting layer are all inherited unchanged.

For a prismatic member the element is parameterised by the dimensionless shear
parameter

$$
\Phi = \frac{12\,EI}{GA_v\,L^2},
$$

giving the locking-free two-node stiffness
([Friedman & Kosmatka, 1993](ref-friedman-kosmatka-1993);
[Przemieniecki, 1968](ref-przemieniecki-1968);
{meth}`pycba.beam.Beam.k_FF_timo`)

$$
\mathbf{k}_{\text{FF}} = \frac{EI}{(1+\Phi)L^3}
\begin{bmatrix}
12 & 6L & -12 & 6L \\
6L & (4+\Phi)L^2 & -6L & (2-\Phi)L^2 \\
-12 & -6L & 12 & -6L \\
6L & (2-\Phi)L^2 & -6L & (4+\Phi)L^2
\end{bmatrix}.
$$

As $GA_v \to \infty$ (slender member, $\Phi \to 0$) this reduces **exactly** to
the Euler–Bernoulli $\mathbf{k}_{\text{FF}}$ above, and member releases are
imposed by the same static condensation used everywhere else.

The element follows the *same flexibility derivation* as the non-prismatic
member, with one addition: applying unit end moments to the released span also
produces a constant shear $v = -1/L$ (the support reactions), so the shear
strain energy adds the term

$$
\frac{1}{L^2}\int_0^L \frac{dx}{GA_v(x)}
$$

to **every** entry of the end-rotation flexibility $\mathbf{F}$ before inverting
to $\mathbf{K}_\theta$ ({meth}`pycba.beam.Beam._timo_flexibility`). For a
constant $EI$/$GA_v$ this reproduces the closed-form matrix above to machine
precision; the same single path
({meth}`pycba.beam.Beam.k_timoshenko`) therefore covers prismatic,
non-prismatic, and variable-shear members. Because this shares the non-prismatic
machinery, **no per-load fixed-end-force formulae are rewritten**: for a
prismatic member the shear contribution to the released-span end rotations
integrates to zero, so the Timoshenko fixed-end moments are the exact closed-form
transform of the Euler–Bernoulli values,

$$
\begin{bmatrix} M_a \\ M_b \end{bmatrix}_{\!T}
= \frac{1}{2(1+\Phi)}
\begin{bmatrix} 2+\Phi & -\Phi \\ -\Phi & 2+\Phi \end{bmatrix}
\begin{bmatrix} M_a \\ M_b \end{bmatrix}_{\!EB}
$$

({meth}`pycba.beam.Beam._ref_timoshenko`); for a variable $EI$/$GA_v$ the end
rotations (including the shear term) are obtained by the same breakpoint-aware
integration. A symmetric load on a prismatic member therefore keeps its
Euler–Bernoulli fixed-end moments, while an unsymmetric load — or a continuous
beam — redistributes according to $\Phi$.

In the member-results recovery the reported rotation is the **section rotation**
$\psi$ (continuous with the nodal DOF), and the deflection integrates the axis
slope $\psi + \gamma$, i.e. the bending deflection plus the shear contribution
$\int V/GA_v$; for $GA_v \to \infty$ the shear slope vanishes and the result is
identical to Euler–Bernoulli.

```{note}
For slender members (span/depth $\gtrsim 20$) the shear deflection is typically
under a couple of percent; the Timoshenko option matters for deep or short
members (transfer beams, deep voided slabs, short spans). Shear-deformable
elements are not currently combined with the nonlinear (plastic-hinge) engine,
which remains Euler–Bernoulli.
```

### Beam on an elastic (Winkler) foundation

A member given a finite foundation modulus $k_f$ (the modulus of subgrade
reaction per unit length of beam) rests on a continuous Winkler foundation that
resists deflection with a distributed reaction $q(x) = -k_f\,v(x)$. (Throughout,
*member* and *span* are used interchangeably for the element between two nodes.)
The governing equation becomes

$$ EI\,\frac{\mathrm{d}^4 v}{\mathrm{d}x^4} + k_f\,v = w(x), $$

whose homogeneous solutions decay over the characteristic length
$\lambda = (4EI/k_f)^{1/4}$.

Rather than introduce the exact (hyperbolic) foundation element and re-derive
the fixed-end forces for every load type, PyCBA models the foundation member as a
**statically-condensed super-element** (the same internal-meshing idea used by
the [nonlinear analysis](theory-nonlinear)). The member is meshed into $n$
ordinary Euler–Bernoulli sub-elements; each receives the standard *consistent*
foundation stiffness

$$ \mathbf{k}_f^{(e)} = \frac{k_f\,h}{420}
\begin{bmatrix}
156 & 22h & 54 & -13h \\
22h & 4h^2 & 13h & -3h^2 \\
54 & 13h & 156 & -22h \\
-13h & -3h^2 & -22h & 4h^2
\end{bmatrix}, $$

formed from the same cubic Hermite shape functions as the element stiffness
($h = L/n$ the sub-element length). The internal nodes are removed by static
condensation, so the member still presents a two-node $4\times4$ stiffness and a
condensed fixed-end-force vector to the global assembly — reactions, plotting
and influence lines are inherited unchanged. Member results are recovered by
reconstructing the internal sub-element displacements and concatenating each
sub-element's exact Euler–Bernoulli diagrams; accuracy improves with mesh
refinement, and the mesh defaults to several sub-elements per characteristic
length $\lambda$. The implementation reproduces the analytic infinite-beam
deflection $P\beta/2k_f$ and moment $P/4\beta$ (with $\beta = 1/\lambda$) under a
point load from [Hetényi (1946)](ref-hetenyi-1946), and is used to validate it.

The Winkler model is linear and *bidirectional*: where a member lifts, the
springs resist by pulling down, so the foundation can carry apparent tension. A
real soil cannot, and a railway run-on slab in fact spans a settlement trough
behind the abutment rather than a continuous bed
([O'Brien, Keogh & O'Connor 2014](ref-obrien-keogh-oconnor-2014), §4.5); the
linear bed therefore over-estimates effects in any uplift zones. The
[foundation tutorial](notebooks/foundation.ipynb) shows a worked railway-bridge
example with ballasted approaches under a moving load.

```{note}
The foundation super-element currently supports prismatic, fixed-fixed members
without shear flexibility (`GAv`), carrying UDL, point and partial-UDL loads;
other combinations raise a clear error.
```

## Global stiffness matrix

For an $N$-span beam there are $N+1$ nodes and hence $2(N+1)$ degrees of
freedom. The global DOF vector is ordered node by node:

$$
\mathbf{d} =
\begin{bmatrix}
v_0 & \theta_0 & v_1 & \theta_1 & \cdots & v_N & \theta_N
\end{bmatrix}^{\mathsf T}.
$$

The unrestricted global stiffness matrix $\mathbf{K}$, of size
$2(N+1)\times2(N+1)$, is assembled by the **direct stiffness method**
({meth}`pycba.analysis.BeamAnalysis._assemble`). For span $m$ (zero-indexed),
the left node is global node $m$ and the right node is global node $m+1$, so the
element DOF map to the global indices $[\,2m,\;2m+1,\;2m+2,\;2m+3\,]$. The
$4\times4$ element stiffness $\mathbf{k}^{(e)}$ is overlapped (scatter-added)
into $\mathbf{K}$:

$$
\mathbf{K}_{[2m:2m+4],\,[2m:2m+4]}
\;\mathrel{+}=\;
\mathbf{k}^{(e)}.
$$

Because adjacent spans share a node, the $2\times2$ blocks of neighbouring
elements overlap on the shared DOF; this overlap is precisely what enforces
**displacement compatibility** (a single $v$ and $\theta$ per node) and **nodal
equilibrium** between the spans. The assembled $\mathbf{K}$ is symmetric, banded
and — before supports are applied — positive *semi*-definite (it retains the
global rigid-body modes).

### Spring supports

An elastic support at DOF $i$ with stiffness $k_s > 0$ (a vertical translational
spring on a $v$-DOF, or a rotational spring on a $\theta$-DOF) contributes a
restoring force $k_s u_i$ at that DOF. It is added directly to the diagonal of
the assembled matrix, *before* boundary conditions:

$$
K_{ii} \;\leftarrow\; K_{ii} + k_s.
$$

A spring DOF then remains a **free unknown** in the linear system — no row or
column elimination is performed for it, unlike a rigid support. Adding the
spring to the unrestricted matrix here is what later allows the spring force to
be recovered correctly during reaction recovery. This makes `PyCBA` usable as a
sub-frame analysis tool, where springs model the rotational/translational
restraint offered by members not explicitly modelled.

## Loads

Span loads are not applied at the nodes directly. Each load on a member is first
converted to its **consistent (equivalent) nodal forces** — the fixed-end forces
the load would produce with both member ends clamped — which then assemble into
the global force vector $\mathbf{f}$. This is the work-equivalent load lumping of
the displacement method
([Przemieniecki, 1968](ref-przemieniecki-1968);
[Cook et al., 2001](ref-cook-2001);
[Felippa, 2004](ref-felippa-iffem)).

### Consistent nodal (fixed-end) forces

For a load on a fixed–fixed span, the consistent nodal load vector is the set of
end shears $V_a, V_b$ and end moments $M_a, M_b$ that hold both ends clamped.
For a transverse distributed load $w(x)$ these follow from the fixed-end
influence integrals

$$
M_a = \frac{1}{L^2}\int_0^L w(x)\,x\,(L-x)^2\,dx,
\qquad
M_b = -\frac{1}{L^2}\int_0^L w(x)\,x^2\,(L-x)\,dx,
$$

with the end shears following from vertical and moment equilibrium of the
clamped span. `PyCBA` implements the closed-form results of these integrals (and
their point-load and applied-moment analogues) for each supported load type in
{mod}`pycba.load` — the standard fixed-end actions tabulated in, e.g.,
[Roark's Formulas](ref-roark-2020) and the
[AISC beam diagrams](ref-aisc-manual). The supported types and the
fixed-end actions on a fixed–fixed span are:

| Type | Load | Class | Fixed-end actions ($M_a$, $M_b$) |
| --- | --- | --- | --- |
| 1 | UDL $w$ | {class}`~pycba.load.LoadUDL` | $M_a = \dfrac{wL^2}{12}$, $\;M_b = -\dfrac{wL^2}{12}$ |
| 2 | Point load $P$ at $a$ (with $b=L-a$) | {class}`~pycba.load.LoadPL` | $M_a = \dfrac{P a b^2}{L^2}$, $\;M_b = -\dfrac{P a^2 b}{L^2}$ |
| 3 | Partial UDL $w$ over $[a, a+c]$ | {class}`~pycba.load.LoadPUDL` | influence integrals over the loaded length |
| 4 | Moment $M$ at $a$ (with $b=L-a$) | {class}`~pycba.load.LoadML` | $M_a = \dfrac{Mb}{L^2}(2a - b)$, $\;M_b = \dfrac{Ma}{L^2}(2b - a)$ |
| 5 | Trapezoidal $w_1 \to w_2$ over $[a, a+c]$ | {class}`~pycba.load.LoadTrapez` | analytic evaluation of the influence integrals |
| 6 | Imposed curvature $\kappa(x)$ | {class}`~pycba.load.LoadIC` | derived below |

The partial-UDL and trapezoidal loads are clipped to the span: any portion that
extends beyond the member end is silently ignored. The full load-matrix format
and column conventions are given on the {doc}`general` page.

### Released end forces (member releases)

When the member is *not* fixed–fixed, the consistent nodal forces above must be
adjusted so that the moment at any **released** end is zero — otherwise the
clamped-end moment would be applied to a hinge. These adjusted forces are the
**released end forces** (`get_ref`). They are obtained by superimposing onto the
fixed–fixed consistent nodal loads a correction that exactly cancels the moment
at the released DOF, distributing its effect to the retained DOF — the
load-vector counterpart of the static condensation applied to the element
stiffness ({meth}`pycba.load.Load.get_ref`; for a non-prismatic member the
equivalent flexibility-based reduction is
{meth}`pycba.beam.Beam._ref_nonprismatic`). For example, releasing the $j$-end
(Type 2) carries the clamped moment $M_b$ back to the $i$-end and into the end
shears via the $3EI/L$ stiffness of the released element, giving the
simply-supported-equivalent fixed-end actions.

### Assembly of the load vector

The released end forces of every span are accumulated into the global force
vector $\mathbf{f}$ ({meth}`pycba.analysis.BeamAnalysis._forces`). The span
contribution is **subtracted**:

$$
\mathbf{f}_{[2m:2m+4]} \;\mathrel{-}=\; \mathbf{f}^{(e)}_{\text{ref}},
$$

because the equivalent nodal loads applied to the structure are equal and
opposite to the fixed-end *reactions* the clamped member exerts on the nodes.
The resulting $\mathbf{f}$ is the right-hand side of $\mathbf{K}\,\mathbf{d} =
\mathbf{f}$.

### Imposed-curvature (initial-strain) loads

An **imposed-curvature** (or initial-strain) load applies a *free*, stress-free
curvature field along a member,

$$
\kappa_{\text{imp}}(x) = \kappa_0 + \kappa_1 x + \kappa_2 x^2 + \dots,
$$

specified by its polynomial coefficients (load Type 6, added with
{meth}`pycba.analysis.BeamAnalysis.add_ic`). It is the mechanism by which
`PyCBA` (and downstream time-dependent tools) apply **creep, shrinkage and
thermal** curvatures to a continuous beam
([Ghali, Favre & Elbadry, 2002, §13.7](ref-ghali-2002);
[Elbadry & Ghali, 1989](ref-elbadry-ghali-1989)). The
implementation is {class}`pycba.load.LoadIC`.

On a statically-determinate (simply-supported) span an imposed curvature induces
**no internal forces** — only a free deflected shape (the curvature is taken up
freely, e.g. a midspan deflection $\kappa L^2/8$ for a uniform $\kappa$). On a
restrained or continuous structure, however, the restraint of the free curvature
generates **real bending moments and reactions**, exactly as for a temperature
gradient or differential settlement.

The fixed-end forces use the same flexibility integration as the non-prismatic
element. The primary (simply-supported) end rotations produced by the free
curvature are

$$
\boldsymbol{\theta}_0 =
\begin{bmatrix}
\displaystyle\int_0^L m_i(x)\,\kappa_{\text{imp}}(x)\,dx \\[2mm]
\displaystyle\int_0^L m_j(x)\,\kappa_{\text{imp}}(x)\,dx
\end{bmatrix},
\qquad
m_i = 1 - \frac{x}{L}, \;\; m_j = -\frac{x}{L},
$$

(the unit-moment diagrams), and the fixed-end moments follow from the
moment–rotation stiffness,

$$
\begin{bmatrix} M_a \\ M_b \end{bmatrix}
= \mathbf{K}_\theta\,\boldsymbol{\theta}_0 ,
$$

with the balancing end shears $V_a = (M_a + M_b)/L = -V_b$ from equilibrium of
the resulting couple (no transverse load is present). For a scalar $EI$ the
closed-form $\mathbf{K}_\theta = \dfrac{2EI}{L}\begin{bmatrix} 2 & 1 \\ 1 & 2
\end{bmatrix}$ is used; for a {class}`~pycba.section.SectionEI` member the
flexibility-integrated $\mathbf{K}_\theta$ is used and the rotation integrals are
split at the section breakpoints, so an imposed curvature on a non-prismatic
member honours every $EI$ kink/step exactly.

Two reference cases fix ideas (both reproduced by `PyCBA` to machine precision):

* a **uniform** curvature $\kappa$ on a prismatic **fixed–fixed** member gives a
  constant restraint moment $M = EI\kappa$ throughout;
* two equal continuous spans, each with uniform $\kappa$, give an
  interior-support moment $-1.5\,EI\kappa$, fully self-equilibrating (zero net
  reaction).

These are the classic imposed-deformation restraint results of
[Ghali, Favre & Elbadry (2002, §13.7)](ref-ghali-2002).

(theory-stability)=

## Stability / mechanism detection

If the restraints leave the structure under-supported, or an internal hinge is
over-released, the structure is a **mechanism**: the stiffness matrix is singular
and the solution is meaningless (enormous, spurious displacements). Before
solving, {meth}`pycba.analysis.BeamAnalysis.analyze` performs a stability check
({meth}`~pycba.analysis.BeamAnalysis._check_stability`) and raises a clear
`ValueError` rather than returning nonsense.

The check operates on the **free-DOF partition** of the *unrestricted* global
stiffness matrix (including any spring terms). Let $\mathcal{F}$ be the set of
DOF that are neither fully fixed ($R_i < 0$) nor carry a prescribed displacement;
spring DOF remain free and contribute their stiffness. The partition
$\mathbf{K}_{\mathcal{F}\mathcal{F}}$ governs the unknown displacements. (The
free partition of the *unrestricted* matrix is used deliberately: direct
elimination places $1.0$ on the diagonal of constrained DOF, which would pollute
the condition number of the reduced system.)

For a **stable** linear-elastic structure $\mathbf{K}_{\mathcal{F}\mathcal{F}}$
is symmetric positive-definite
([Bathe, 2014](ref-bathe-2014)); a mechanism makes it singular —
its smallest eigenvalue collapses to zero relative to the largest. `PyCBA`
forms the symmetric eigenvalues $\lambda$ of
$\mathbf{K}_{\mathcal{F}\mathcal{F}}$ (`numpy.linalg.eigvalsh`) and compares the
**reciprocal condition number**

$$
\frac{\min_k |\lambda_k|}{\max_k |\lambda_k|}
$$

against a floor of $10^{-12}$
([Golub & Van Loan, 2013](ref-golub-vanloan-2013)). A value below
the floor (or a zero maximum eigenvalue) signals a mechanism. The floor is
chosen so that a genuine mechanism — whose null eigenvalue sits near machine
epsilon ($\sim 10^{-16}$) — is separated with margin from a legitimately flexible
but stable structure.

The check runs **at most once per structure**: its result is cached against the
beam's `structure_version` and re-evaluated only if the geometry, rigidity,
restraints or prescribed displacements change. It therefore adds no cost to
looped analyses that vary only the loads (e.g. a moving-load run). The same logic
is exposed without solving via {meth}`pycba.analysis.BeamAnalysis.is_stable`
(returns a `bool`), and can be skipped with `analyze(check_stability=False)` for
an intentionally near-singular model. As a final safeguard, the linear solve
itself ({meth}`~pycba.analysis.BeamAnalysis._solver`) traps a singular matrix and
raises the same instability error.

## Boundary conditions

With $\mathbf{K}$ and $\mathbf{f}$ assembled, the governing system is

$$
\mathbf{K}\,\mathbf{d} = \mathbf{f}.
$$

Boundary conditions are imposed by the **direct elimination method**
({meth}`pycba.analysis.BeamAnalysis._apply_bc`;
[Cook et al., 2001](ref-cook-2001)). A DOF $i$ is eliminated
whenever its displacement is known — either because an explicit value
$\bar{d}_i$ has been prescribed in the displacement vector $\mathbf{D}$, or
because the DOF is fully fixed ($R_i = -1$) with no override, in which case
$\bar{d}_i = 0$. For each such DOF:

1. Transfer the constraint's contribution to the right-hand side of every other
   equation, by subtracting the full $i$-th column scaled by $\bar{d}_i$:

   $$
   f_j \;\leftarrow\; f_j - K_{ji}\,\bar{d}_i \qquad \forall\, j.
   $$

2. Zero the $i$-th row and column:

   $$
   K_{ij} = K_{ji} = 0 \qquad \forall\, j.
   $$

3. Set the diagonal to unity and the right-hand side to the prescribed value:

   $$
   K_{ii} = 1, \qquad f_i = \bar{d}_i.
   $$

This preserves symmetry and reproduces the prescribed displacement exactly in the
solution ($d_i = \bar{d}_i$), while correctly transferring the constraint
reaction into the remaining free equations.

* A **fixed support** is the special case $\bar{d}_i = 0$.
* A **support settlement** is modelled simply by providing a non-zero
  $\bar{d}_i$ at the support DOF (negative = downward). Prescribed displacements
  may also be applied to otherwise-free DOF.
* **Spring DOF** ($R_i > 0$) without a prescribed displacement are *not*
  eliminated — their stiffness is already on the diagonal and they remain free
  unknowns.

```{note}
A DOF cannot simultaneously carry a spring ($R_i > 0$), a prescribed displacement
*and* a non-zero consistent nodal load: the elimination would set
$f_i = \bar{d}_i$ and silently discard the load. `PyCBA` validates against this
inconsistent combination up front ({meth}`pycba.analysis.BeamAnalysis._validate`)
and raises a `ValueError`.
```

## Solution and reaction recovery

The restricted system is solved for the nodal displacements with a direct dense
solver, $\mathbf{d} = \mathbf{K}^{-1}\mathbf{f}$ (`numpy.linalg.solve`,
[Harris et al., 2020](ref-harris-numpy-2020)).

Support reactions are recovered using the **unrestricted** stiffness matrix
$\mathbf{K}_U$ (assembled before boundary conditions, including spring terms) and
the original force vector $\mathbf{f}_U$
({meth}`pycba.analysis.BeamAnalysis._reactions`). The nodal residual is

$$
\mathbf{r}^{*} = \mathbf{K}_U\,\mathbf{d} - \mathbf{f}_U.
$$

At a **fully-fixed** DOF $i$ ($R_i = -1$) the displacement is zero (or the
prescribed settlement) and the residual equals the support reaction directly:

$$
R_i = r^{*}_i,
$$

returned in `beam_results.R`, with the upward-positive sign convention.

For a **spring** DOF the residual contains structural coupling from neighbouring
DOF and any applied nodal load, so it is *not* the spring force alone. The spring
force is therefore computed explicitly from the spring displacement and reported
separately in `beam_results.Rs`:

$$
F_s^{(i)} = -k_s\,u_i,
$$

with the sign chosen so that an upward spring reaction is positive when the
displacement is downward (negative).

## Member results: moment, shear, rotation and deflection

Once the nodal displacements are known, the load effects are recovered along each
member at $n_{\text{pts}}$ stations (default 100;
{meth}`pycba.results.BeamResults._member_values`). The **bending moment** and
**shear** distributions are obtained from *exact* analytical expressions:
superposing the member-end-moment effect (from the solved nodal moments) with the
simply-supported member results of each applied load (the closed-form `get_mbr_results`
of each load class in {mod}`pycba.load`). These are exact because the
inter-nodal load distributions are known in closed form.

The **rotation** and **deflection** are then obtained by integrating the
curvature, because the cubic Hermite shape functions are *not* valid in the
presence of inter-nodal loading. The total curvature is the flexural curvature
plus any free (imposed) curvature:

$$
\kappa(x) = \frac{M(x)}{EI(x)} + \kappa_{\text{imp}}(x),
$$

with $EI(x)$ evaluated point-wise for a non-prismatic member (constant for the
prismatic path). The rotation and deflection follow from successive integration:

$$
\theta(x) = \theta_0 + \int_0^x \kappa(s)\,ds,
\qquad
\delta(x) = \delta_0 + \int_0^x \theta(s)\,ds,
$$

evaluated numerically by the cumulative trapezoidal rule

$$
\int_0^x y(s)\,ds \;\approx\; \sum_{k=1}^{n} \frac{y_{k-1} + y_k}{2}\,\Delta x,
$$

using `scipy.integrate.cumulative_trapezoid`
([Virtanen et al., 2020](ref-scipy-2020)) over the member stations.

The integration constants $\theta_0$ and $\delta_0$ are the rotation and
deflection at the $i$-node. The deflection constant is the nodal DOF,
$\delta_0 = v_i$. For a **fixed–fixed** element the start rotation is also the
nodal DOF, $\theta_0 = \theta_i$. For an element with a **release** (types 2–4)
the rotation at a pinned $i$-end is not a primary unknown and must be recovered.
In the prismatic case `PyCBA` uses the slope-deflection relation

$$
\theta_i = \frac{\delta_j - \delta_i}{L}
- \frac{L}{3EI}\Big(-\,\mathrm{FEM}_i + \tfrac{1}{2}\,\mathrm{FEM}_j
+ M_i - \tfrac{1}{2}\,M_j\Big),
$$

where the $\mathrm{FEM}$ are the consistent nodal-load moments due to the
inter-nodal loads and the $M_i, M_j$ are the solved member-end moments. For a
non-prismatic member, or whenever an imposed curvature is present, $\theta_0$ is
instead recovered from the kinematic boundary condition $\delta(L) = v_j$, which
is valid for any $EI(x)$ and curvature field.

For a **Timoshenko** member the integrated quantity $\theta(x)$ is the section
rotation $\psi$ (from the bending curvature, as above), and the deflection
integrates the **axis slope** $\psi + \gamma$ with the shear strain
$\gamma = V/GA_v$, so $\delta$ carries the additional shear deflection
$\int V/GA_v$. Because the closed-form release correction is Euler–Bernoulli
only, a released Timoshenko member uses the same kinematic boundary-condition
recovery as the non-prismatic case. For $GA_v \to \infty$ the shear slope
vanishes and the recovery is identical to Euler–Bernoulli.

## Linear superposition, load cases and patterning

Because the analysis is **linear elastic**, responses to different load
arrangements superpose. `PyCBA` exposes this directly: load matrices can be added
and factored ({func}`pycba.load.add_LM`, {func}`pycba.load.factor_LM`), and the
higher-level `LoadCases` / `LoadCombination` workflow assembles a response matrix
of basis cases and forms arbitrary weighted combinations and envelopes (see the
{doc}`general` page).

### Segmented load patterning

For a distributed live load that may act on selected parts of the beam, a
continuous UDL is discretised into short partial-UDL **segments**, each analysed
as a basis `LoadCase` over the whole beam. For a chosen target effect (e.g.
bending moment at an internal support, at target coordinate $x_t$) the basis
analyses give a response vector

$$
\mathbf{r}(x_t) = \big[\,r_1(x_t),\; r_2(x_t),\; \dots,\; r_n(x_t)\,\big],
$$

where $r_i(x_t)$ is the contribution of segment $i$. A loading arrangement is a
`LoadCombination` with $\{0,1\}$ factors $\alpha_i$ on the basis cases. When the
segments may be selected **independently**, the adverse arrangement is a simple
sign selection — include every segment whose contribution worsens the target
effect:

$$
\alpha_i =
\begin{cases}
1, & r_i(x_t) < 0 \quad \text{(for a minimum / hogging effect)} \\
0, & \text{otherwise,}
\end{cases}
$$

with the inequality reversed for a maximum effect. This produces one physical
loading arrangement for that target.

If a loading rule instead requires a single **contiguous** loaded length, the
ordered segment responses are searched for the contiguous block with the largest
adverse sum — a Kadane maximum-subarray step, with recurrence

$$
b_i = \max(r_i,\; b_{i-1} + r_i),
\qquad
B_i = \max(B_{i-1},\; b_i)
$$

(sign-reversed for a minimum effect). The indices of the governing block define
the non-zero factors of the `LoadCombination`. A target combination such as
"hogging at support 1" is therefore *one* arrangement, whereas a response
envelope over many target stations is an envelope of (possibly different)
arrangements.

## Free-vibration (modal) analysis

Beyond static analysis, `PyCBA` provides the natural frequencies and mode shapes
of the beam. A **consistent mass matrix** is assembled alongside the stiffness
matrix; for a prismatic Euler–Bernoulli element of mass per unit length
$\bar m$ it is, from the same cubic Hermite shape functions used for the
stiffness,

$$ \mathbf{m}^{(e)} = \frac{\bar m\,L}{420}
\begin{bmatrix}
156 & 22L & 54 & -13L \\
22L & 4L^2 & 13L & -3L^2 \\
54 & 13L & 156 & -22L \\
-13L & -3L^2 & -22L & 4L^2
\end{bmatrix}. $$

The natural circular frequencies $\omega$ and mode shapes
$\boldsymbol\phi$ are the solutions of the generalized eigenproblem

$$ \mathbf{K}\,\boldsymbol\phi = \omega^2\,\mathbf{M}\,\boldsymbol\phi $$

on the free (unrestrained) degrees of freedom, with elastic spring supports
contributing to $\mathbf{K}$. Because a single element per span resolves only
the first mode or two, each span is refined into several Euler–Bernoulli
sub-elements for the eigenanalysis; the lowest frequencies then match the
classical analytic results (simply-supported $\omega_n = (n\pi/L)^2\sqrt{EI/\bar m}$,
cantilever, fixed-fixed) to well under a percent.

```{note}
Modal analysis currently supports prismatic, fixed-fixed spans without shear
flexibility (`GAv`); other combinations raise a clear error.
```

(theory-nonlinear)=

## Nonlinear analysis — Generalized Clough model

Beyond linear elastic analysis, `PyCBA` provides an incremental **nonlinear
(elasto-plastic)** analysis that tracks plastic-hinge formation and moment
redistribution up to collapse. It uses **concentrated plasticity**: nonlinear
behaviour is localised at element ends (potential hinge locations), while the
element interior remains elastic
([Clough & Johnston, 1966](ref-clough-johnston-1966);
[Li & Li, 2007, Ch. 4](ref-li-li-2007);
[Neal, 1977](ref-neal-1977)).

### Concentrated plasticity via stiffness degradation

The Generalized Clough model
([Clough & Johnston, 1966](ref-clough-johnston-1966)) introduces a
stiffness-reduction parameter $R$ at each element end, varying between $R = 1$
(fully elastic) and $R = q$ (fully plastic hinge), where $q$ is the
strain-hardening ratio ($q = 0$ for elastic–perfectly-plastic). The transition is
governed by the normalised moment ratio $\gamma = |M|/M_p$:

$$
R =
\begin{cases}
1, & \gamma \le \gamma_y, \\[4pt]
1 - \dfrac{\gamma - \gamma_y}{1 - \gamma_y}\,(1 - q), & \gamma_y < \gamma < 1, \\[6pt]
q, & \gamma \ge 1,
\end{cases}
$$

where $\gamma_y = M_y/M_p$ is the normalised yield moment. This produces a
bilinear moment–rotation response at the section. On unloading ($\gamma$
decreasing below the historical peak), $R$ resets to $1$ — the Clough
"origin-oriented" unloading rule.

### Element stiffness with degradation

For an element with left- and right-end parameters $R_1$ and $R_2$, the element
stiffness is interpolated between the fixed–fixed ($\mathbf{k}_e$),
pinned–fixed ($\mathbf{k}_1$) and fixed–pinned ($\mathbf{k}_2$) matrices
([Li & Li, 2007, Ch. 4](ref-li-li-2007)). If $R_1 \ge R_2$,

$$
\mathbf{k} = R_2\,\mathbf{k}_e + (R_1 - R_2)\,\mathbf{k}_2,
$$

and if $R_1 < R_2$,

$$
\mathbf{k} = R_1\,\mathbf{k}_e + (R_2 - R_1)\,\mathbf{k}_1.
$$

The stiffness degrades smoothly as either end yields, and reduces to the elastic
matrix when $R_1 = R_2 = 1$.

### Hinge ownership

At a node shared by two elements, each plastic hinge is *owned* by a single
element end — the one that reached plasticity first — while the adjacent element
retains $R = 1$ at that node. If both ends at a node were simultaneously degraded
the global stiffness would become singular prematurely, terminating the analysis
before the true collapse mechanism forms. For an interior node $j$ the hinge is
assigned to element $j-1$ (its right end), which keeps the global stiffness
non-singular during the progressive formation of hinges.

### Incremental analysis (static)

The proportional-load analysis proceeds incrementally:

1. **Initialise:** all $R = 1$, load factor $\lambda = 0$, moments
   $\mathbf{M} = \mathbf{0}$.
2. **Increment:** choose a step $\Delta\lambda$ (adaptive — smaller when any $R$
   is low, i.e. near plasticity).
3. **Solve** the current tangent system
   $\mathbf{K}\,\Delta\mathbf{u} = \Delta\lambda\,\mathbf{f}_{\text{ref}}$, where
   $\mathbf{f}_{\text{ref}}$ is the reference load vector.
4. **Update moments:** $\mathbf{M} \leftarrow \mathbf{M} + \Delta\mathbf{M}$.
5. **Update $R$:** recompute $\gamma$ at each node; degrade $R$ if $\gamma$
   exceeds its historical peak, or reset $R = 1$ on unloading.
6. **Check for collapse** when a new hinge forms (see below).
7. **Repeat** until collapse or $\lambda_{\max}$.

### Moving-load analysis

For a vehicle traversing the beam the load position changes at each step, and the
load must be transferred from the old position to the new one without spurious
plastic deformation. `PyCBA` uses paired **elastic-unload / nonlinear-reload**
sub-increments at each position: unload the previous position's load fraction
using the *elastic* stiffness (unloading is always elastic in the Clough model),
then reload the new position's load fraction using the *current* (degraded)
stiffness. Each position step is divided into $n_{\text{sub}}$ sub-increments for
accuracy. This correctly accumulates plastic damage as the vehicle traverses
([McCarthy, 2012](ref-mccarthy-2012);
[Caprani, 2006](ref-caprani-2006)).

### Collapse detection

A collapse mechanism requires enough plastic hinges to render the structure
kinematically unstable. `PyCBA` detects this with a **rank test**: nearby hinged
nodes are first clustered (within $2 h_{\min}$, where $h_{\min}$ is the smallest
element length) to absorb mesh discretisation; at each clustered hinge $R = 0$ is
set at *both* element ends, fully releasing the node as in a true mechanism; the
test stiffness $\mathbf{K}_{\text{test}}$ is assembled with the boundary
conditions applied; and if
$\operatorname{rank}(\mathbf{K}_{\text{test}}) < n_{\text{dof}}$ the structure is
a mechanism and collapse has occurred. This direct kinematic test is more robust
than monitoring determinant magnitude or displacement growth.

### Mesh considerations

The continuous beam is internally meshed into short elements of a target length
(`mesh_size`). Point loads at mesh nodes are represented exactly; loads between
nodes are distributed to adjacent nodes by Hermite interpolation, and UDL is
lumped to nodal forces. Plastic hinges can form only at mesh nodes, so a hinge
whose theoretical location (e.g. $x^{*} = 0.414\,L$ for a propped cantilever
under UDL) falls between nodes "snaps" to the nearest node — a discretisation
error that is *not* monotonic with refinement. Point-load problems whose load and
hinge locations coincide with nodes converge rapidly; UDL problems typically show
errors of a few percent, reducible by placing nodes near the expected hinge
locations.

---

The nonlinear analysis is demonstrated in the
[nonlinear tutorial](notebooks/nonlinear.ipynb); worked linear examples are in
the {doc}`tutorials`, and all the source texts cited above — for the matrix
stiffness method, the consistent nodal loads, the non-prismatic and
imposed-curvature formulations, the stability check, and the plastic-hinge
model — are collected with hyperlinks on the [References](references.md) page.
