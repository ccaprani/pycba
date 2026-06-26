# Changelog

## Unreleased

### Features
- **Free-vibration (modal) analysis**: `BeamAnalysis.modal(mass, n_modes=...)` returns the natural frequencies and mode shapes, assembling a consistent mass matrix alongside the stiffness matrix and solving the generalized eigenproblem `K φ = ω² M φ`. Each span is refined into Euler–Bernoulli sub-elements so the modes are accurate (validated against the classical simply-supported, cantilever and fixed-fixed frequencies to <0.1%). Returns a `ModalResults` (`omega`, `f`, `periods`, mode shapes, `plot()`); elastic spring supports are included. Prismatic, fixed-fixed spans without `GAv`.
- **Individual result-diagram plots**: `BeamAnalysis.plot_bmd`, `plot_sfd` and `plot_dsd` draw a single bending-moment, shear-force or deflection diagram (the bending moment sagging-positive, with the y-axis inverted, matching `plot_results`). Each accepts an existing `ax`, so two analyses can be overlaid for comparison, and matplotlib `**kwargs` (`color`, `ls`, `label`, …).
- **Timoshenko (shear-deformable) beam elements**: a member becomes a shear-deformable Timoshenko element when given a finite transverse shear rigidity `GAv` (a scalar, or a `SectionEI` for a variable `GAv(x)`); with `GAv=None` (the default) the member stays on the exact Euler–Bernoulli path, bit-for-bit unchanged. Shear deformation flows through the element stiffness, the fixed-end forces and the displacement recovery, for prismatic and non-prismatic members and all four release types, routed through the existing flexibility integrator (no per-load closed forms rewritten). Two DOF per node is preserved, so the `supports=` API, reactions, plotting and influence lines are inherited unchanged. `GAv` is accepted by `Beam`, `Beam.add_span`/`add_member`, `BeamAnalysis` and `BridgeAnalysis.add_bridge`. Nonlinear (plastic-hinge) analysis remains Euler–Bernoulli.

### Documentation
- Add a free-vibration (modal analysis) tutorial notebook.
- Add a Timoshenko beam-element tutorial notebook, a Theoretical Basis section on the shear-deformable element, and literature references ([Timoshenko, 1921](https://doi.org/10.1080/14786442108636264); [Cowper, 1966](https://doi.org/10.1115/1.3625046); [Friedman & Kosmatka, 1993](https://doi.org/10.1016/0045-7949(93)90243-7)).

## 0.8.0 — 2026-06-11

### Features
- **Nonlinear (elasto-plastic) beam analysis** (#140): `NonlinearBeamAnalysis`, an incremental analysis using the Generalized Clough concentrated-plasticity model. Tracks plastic-hinge formation and moment redistribution to collapse for both proportional static loading and moving vehicle loads, with rank-test mechanism detection. Prismatic members only — a non-prismatic (`SectionEI`) member raises a clear `TypeError`.
- **Non-prismatic (variable-EI) elements** (#139): `SectionEI`, a member whose flexural rigidity varies along its length, built from `const` / `linear` / `pwl` / `poly` segments and analysed exactly by piece-by-piece flexibility integration. Scalar and `SectionEI` members can be mixed in one beam.
- **Imposed-curvature (initial-strain) loads** (#139): load type 6 / `BeamAnalysis.add_ic`, a free curvature field for modelling creep, shrinkage and thermal effects.
- Add coincident load effects to `Envelopes` (#122). New attributes `Vco_Mmax`, `Vco_Mmin`, `Mco_Vmax`, `Mco_Vmin` track the co-existing value of the other effect (V or M) at the truck position that caused each envelope extreme. `critical_values()` output now includes `"Vco"` and `"Mco"` keys. Coincident values are preserved through `augment()` and `zero_like()`.
- Add trapezoidal (linearly varying) distributed load type (#101). Load type 5 supports both full-span `[span, 5, w1, w2]` and partial coverage `[span, 5, w1, w2, a, c]` where w1/w2 are intensities at positions a and a+c respectively. Also adds `BeamAnalysis.add_trap()` convenience method with optional `a` and `c` parameters.
- Add `pos_start` and `pos_end` parameters to `BridgeAnalysis.run_vehicle()` to restrict the vehicle traverse range (#53). Useful for transverse deck analyses where the vehicle is confined to specific lanes.
- Add `Envelopes.sum()` method for element-wise addition of compatible envelopes (#92). This enables superimposing load effects from different sources, e.g. a patterned UDL envelope with a moving vehicle envelope.
- Fix `InfluenceLines.get_il()` raising `IndexError` when the point of interest does not fall exactly on the result grid (#89). This occurred with longer spans where the default 100-point discretization produced a grid spacing that could not match arbitrary poi values. The fix uses per-member result lookup, avoiding both the overly tight floating-point tolerance and the zero-valued padding indices at span boundaries.

### Documentation
- Migrated the documentation to Markdown (MyST); reorganised and expanded the Theoretical Basis page (elements, loads, nonlinear analysis); added tutorials for non-prismatic elements and imposed curvature (#139, #140).
