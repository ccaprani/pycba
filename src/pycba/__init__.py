"""
PyCBA - Continuous Beam Analysis in Python
"""

__version__ = "0.8.0"

from .analysis import BeamAnalysis
from .beam import Beam
from .section import SectionEI
from .types import MemberType
from .load import (
    LoadCNL,
    MemberResults,
    LoadMatrix,
    LoadType,
    parse_LM,
    add_LM,
    factor_LM,
)
from .results import BeamResults, Envelopes
from .render import BeamPlotter
from .units import UnitSystem, set_units, get_units
from .inf_lines import InfluenceLines
from .utils import parse_beam_string
from .bridge import BridgeAnalysis
from .vehicle import Vehicle, make_train, VehicleLibrary
from .load_cases import (
    LoadCase,
    LoadCombination,
    LoadCases,
    build_pycba_model,
    analyse_load_case,
    analyze_load_case,
    collect_response_matrix,
    additive_envelope,
    make_patterned_udl,
    make_span_udl_cases,
    plot_response_envelope,
    plot_load_patterns,
)
from .pattern import LoadPattern
from .nonlinear import NonlinearBeamAnalysis, NonlinearResult, HingeEvent
