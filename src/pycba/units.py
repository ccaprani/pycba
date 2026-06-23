"""
PyCBA - Display unit systems.

PyCBA's solver is **unit-agnostic**: it performs no unit conversions, so any
internally consistent set of units (e.g. kN/m/kNm, or N/mm/Nmm) may be used as
long as all inputs share the same system.  This module does **not** change that
contract - it only governs how plots are *labelled* and how deflections are
*scaled for display*.

A :class:`UnitSystem` is a small, immutable bundle of label strings (force,
length, moment, distributed load) plus a deflection display scale/label.  Pick
one of the named presets (``"SI"``, ``"SI-N-mm"``, ``"US-ft"``, ``"US-in"``,
``"none"``), set it globally with :func:`set_units`, or pass ``units=`` to any
plotting method to override it for a single figure.  Build a custom
:class:`UnitSystem` for anything else.

The default is SI with kN and m (``"SI"``), matching PyCBA's historical plot
labels, so existing scripts and figures are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class UnitSystem:
    """
    A display unit system: axis-label units and the deflection display scale.

    Holds only label strings and a deflection display factor - it is purely a
    presentation aid and never affects the analysis (PyCBA does no unit
    conversion).

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"SI (kN, m)"``).
    force : str
        Force unit label, e.g. ``"kN"`` (used for shear, reactions, loads).
    length : str
        Length unit label, e.g. ``"m"`` (used for the distance axis).
    moment : str
        Moment unit label, e.g. ``"kNm"`` (conventions vary, so it is given
        explicitly rather than composed from force x length).
    distributed : str
        Distributed-load unit label, e.g. ``"kN/m"``.
    disp_label : str
        Unit shown on the deflection axis, e.g. ``"mm"``.
    disp_scale : float
        Factor the native deflection is multiplied by for display.  For SI
        (kN, m) this is ``1000`` (m shown as mm); for a system whose length
        unit already matches the deflection unit it is ``1``.
    """

    name: str
    force: str = ""
    length: str = ""
    moment: str = ""
    distributed: str = ""
    disp_label: str = ""
    disp_scale: float = 1.0

    # --- label helpers ------------------------------------------------- #
    @staticmethod
    def _suffix(unit: str) -> str:
        return f" ({unit})" if unit else ""

    @property
    def moment_axis(self) -> str:
        return "Bending Moment" + self._suffix(self.moment)

    @property
    def shear_axis(self) -> str:
        return "Shear Force" + self._suffix(self.force)

    @property
    def deflection_axis(self) -> str:
        return "Deflection" + self._suffix(self.disp_label)

    @property
    def distance_axis(self) -> str:
        return "Distance along beam" + self._suffix(self.length)

    def length_axis(self, what: str = "Distance along beam") -> str:
        """Axis label for a length quantity, e.g. ``length_axis("Position")``."""
        return what + self._suffix(self.length)

    # --- inline value formatting (load annotations) -------------------- #
    @staticmethod
    def _val(value: str, unit: str) -> str:
        return f"{value} {unit}" if unit else value

    def fmt_force(self, value: float) -> str:
        return self._val(f"{value:g}", self.force)

    def fmt_moment(self, value: float) -> str:
        return self._val(f"{value:g}", self.moment)

    def fmt_distributed(self, w0: float, w1: Optional[float] = None) -> str:
        """Format a (possibly varying) distributed load, e.g. ``5→8 kN/m``."""
        if w1 is None or w1 == w0:
            body = f"{w0:g}"
        else:
            body = f"{w0:g}→{w1:g}"
        return self._val(body, self.distributed)


# --------------------------------------------------------------------------- #
# Named presets
# --------------------------------------------------------------------------- #
SI = UnitSystem(
    name="SI (kN, m)",
    force="kN",
    length="m",
    moment="kNm",
    distributed="kN/m",
    disp_label="mm",
    disp_scale=1000.0,
)

SI_N_MM = UnitSystem(
    name="SI (N, mm)",
    force="N",
    length="mm",
    moment="N·mm",
    distributed="N/mm",
    disp_label="mm",
    disp_scale=1.0,
)

US_KIP_FT = UnitSystem(
    name="US (kip, ft)",
    force="kip",
    length="ft",
    moment="kip·ft",
    distributed="kip/ft",
    disp_label="in",
    disp_scale=12.0,
)

US_KIP_IN = UnitSystem(
    name="US (kip, in)",
    force="kip",
    length="in",
    moment="kip·in",
    distributed="kip/in",
    disp_label="in",
    disp_scale=1.0,
)

NONE = UnitSystem(name="dimensionless")

# Friendly aliases -> preset (keys are normalised: lower-case, spaces/underscores
# collapsed to hyphens).  EU and AUS practice both use SI kN-m.
_REGISTRY = {
    "si": SI,
    "kn-m": SI,
    "knm": SI,
    "eu": SI,
    "aus": SI,
    "metric": SI,
    "si-n-mm": SI_N_MM,
    "n-mm": SI_N_MM,
    "nmm": SI_N_MM,
    "us": US_KIP_FT,
    "us-ft": US_KIP_FT,
    "kip-ft": US_KIP_FT,
    "us-customary": US_KIP_FT,
    "us-in": US_KIP_IN,
    "kip-in": US_KIP_IN,
    "none": NONE,
    "dimensionless": NONE,
    "": NONE,
}

_default: UnitSystem = SI


def _normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def set_units(units: Union[str, UnitSystem]) -> UnitSystem:
    """
    Set the global default display unit system used by plots.

    Parameters
    ----------
    units : str or UnitSystem
        A preset name (e.g. ``"SI"``, ``"US-ft"``, ``"N-mm"``, ``"none"``) or a
        :class:`UnitSystem` instance.

    Returns
    -------
    UnitSystem
        The resolved unit system now in effect.
    """
    global _default
    _default = resolve(units)
    return _default


def get_units() -> UnitSystem:
    """Return the current global default display unit system."""
    return _default


def resolve(units: Optional[Union[str, UnitSystem]] = None) -> UnitSystem:
    """
    Resolve a units argument to a :class:`UnitSystem`.

    ``None`` returns the global default (see :func:`set_units`); a string is
    looked up among the presets (case-insensitive); a :class:`UnitSystem` is
    returned unchanged.

    Raises
    ------
    KeyError
        If ``units`` is an unknown preset name.
    """
    if units is None:
        return _default
    if isinstance(units, UnitSystem):
        return units
    key = _normalise(units)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown unit system {units!r}. Known presets: "
            + ", ".join(sorted({k for k in _REGISTRY if k}))
        )
