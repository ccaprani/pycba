"""
PyCBA - Continuous Beam Analysis in Python
"""

__version__ = "0.6.0"

from .analysis import BeamAnalysis
from .beam import Beam
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
from .inf_lines import InfluenceLines
from .utils import parse_beam_string
from .bridge import BridgeAnalysis
from .vehicle import Vehicle, make_train, VehicleLibrary
from .pattern import LoadPattern
from .nonlinear import NonlinearBeamAnalysis, NonlinearResult, HingeEvent
