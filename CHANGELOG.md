# Changelog

## Unreleased

### Bug Fixes
- Fix `InfluenceLines.get_il()` raising `IndexError` when the point of interest does not fall exactly on the result grid (#89). This occurred with longer spans where the default 100-point discretization produced a grid spacing that could not match arbitrary poi values. The fix uses per-member result lookup, avoiding both the overly tight floating-point tolerance and the zero-valued padding indices at span boundaries.
