# Changelog

## Unreleased

### Features
- Add coincident load effects to `Envelopes` (#122). New attributes `Vco_Mmax`, `Vco_Mmin`, `Mco_Vmax`, `Mco_Vmin` track the co-existing value of the other effect (V or M) at the truck position that caused each envelope extreme. `critical_values()` output now includes `"Vco"` and `"Mco"` keys. Coincident values are preserved through `augment()` and `zero_like()`.
- Add trapezoidal (linearly varying) distributed load type (#101). Load type 5 supports both full-span `[span, 5, w1, w2]` and partial coverage `[span, 5, w1, w2, a, c]` where w1/w2 are intensities at positions a and a+c respectively. Also adds `BeamAnalysis.add_trap()` convenience method with optional `a` and `c` parameters.
- Add `pos_start` and `pos_end` parameters to `BridgeAnalysis.run_vehicle()` to restrict the vehicle traverse range (#53). Useful for transverse deck analyses where the vehicle is confined to specific lanes.
- Add `Envelopes.sum()` method for element-wise addition of compatible envelopes (#92). This enables superimposing load effects from different sources, e.g. a patterned UDL envelope with a moving vehicle envelope.
- Fix `InfluenceLines.get_il()` raising `IndexError` when the point of interest does not fall exactly on the result grid (#89). This occurred with longer spans where the default 100-point discretization produced a grid spacing that could not match arbitrary poi values. The fix uses per-member result lookup, avoiding both the overly tight floating-point tolerance and the zero-valued padding indices at span boundaries.
