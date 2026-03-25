# Changelog

## Unreleased

### Features
- Add coincident load effects to `Envelopes` (#122). New attributes `Vco_Mmax`, `Vco_Mmin`, `Mco_Vmax`, `Mco_Vmin` track the co-existing value of the other effect (V or M) at the truck position that caused each envelope extreme. `critical_values()` output now includes `"Vco"` and `"Mco"` keys. Coincident values are preserved through `augment()` and `zero_like()`.
