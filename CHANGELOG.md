# Changelog

## Unreleased

### Features
- Add trapezoidal (linearly varying) distributed load type (#101). Load type 5 supports both full-span `[span, 5, w1, w2]` and partial coverage `[span, 5, w1, w2, a, c]` where w1/w2 are intensities at positions a and a+c respectively. Also adds `BeamAnalysis.add_trap()` convenience method with optional `a` and `c` parameters.
