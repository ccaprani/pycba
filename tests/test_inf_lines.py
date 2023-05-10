# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pycba as cba


def test_basic_il():
    """
    A basic IL check
    """
    L = [5, 5, 10]
    EI = 30 * 600e7 * np.ones(len(L)) * 1e-6
    eType = [2, 1, 1]
    R = [-1, -1, 0, 0, -1, 0, -1, 0]

    ils = cba.InfluenceLines(L, EI, R, eType)
    ils.create_ils(step=0.5)

    (x, y) = ils.get_il(15.0, "V")
    assert y == pytest.approx(
        [
            0.0,
            0.0018125,
            0.007,
            0.0151875,
            0.026,
            0.0390625,
            0.054,
            0.0704375,
            0.088,
            0.1063125,
            0.125,
            0.1186875,
            0.112,
            0.1045625,
            0.096,
            0.0859375,
            0.074,
            0.0598125,
            0.043,
            0.0231875,
            0.0,
            -0.02684375,
            -0.05725,
            -0.09103125,
            -0.128,
            -0.16796875,
            -0.21075,
            -0.25615625,
            -0.304,
            -0.35409375,
            0.59375,
            0.53971875,
            0.484,
            0.42678125,
            0.36825,
            0.30859375,
            0.248,
            0.18665625,
            0.12475,
            0.06246875,
            0.0,
        ]
    )

    (x, y) = ils.get_il(0, "R")
    assert y == pytest.approx(
        [
            1.0,
            0.996375,
            0.986,
            0.969625,
            0.948,
            0.921875,
            0.892,
            0.859125,
            0.824,
            0.787375,
            0.75,
            0.662625,
            0.576,
            0.490875,
            0.408,
            0.328125,
            0.252,
            0.180375,
            0.114,
            0.053625,
            0.0,
            -0.0463125,
            -0.0855,
            -0.1179375,
            -0.144,
            -0.1640625,
            -0.1785,
            -0.1876875,
            -0.192,
            -0.1918125,
            -0.1875,
            -0.1794375,
            -0.168,
            -0.1535625,
            -0.1365,
            -0.1171875,
            -0.096,
            -0.0733125,
            -0.0495,
            -0.0249375,
            0.0,
        ]
    )

    (x, y) = ils.get_il(7.5, "M")
    assert y == pytest.approx(
        [
            0.0,
            -0.0090625,
            -0.035,
            -0.0759375,
            -0.13,
            -0.1953125,
            -0.27,
            -0.3521875,
            -0.44,
            -0.5315625,
            -0.625,
            -0.3434375,
            -0.06,
            0.2271875,
            0.52,
            0.8203125,
            0.63,
            0.4509375,
            0.285,
            0.1340625,
            0.0,
            -0.11578125,
            -0.21375,
            -0.29484375,
            -0.36,
            -0.41015625,
            -0.44625,
            -0.46921875,
            -0.48,
            -0.47953125,
            -0.46875,
            -0.44859375,
            -0.42,
            -0.38390625,
            -0.34125,
            -0.29296875,
            -0.24,
            -0.18328125,
            -0.12375,
            -0.06234375,
            0.0,
        ]
    )


def test_parse_beam_notation():
    beam_str = "E20F"
    (L, EI, R, eType) = cba.parse_beam_string(beam_str)

    assert L == [20.0]
    assert EI == np.array([30e4])
    assert R == [-1, -1, 0, 0]
    assert eType == [1]

    beam_str = "E30R30H30R30E"
    (L, EI, R, eType) = cba.parse_beam_string(beam_str)

    assert L == [30.0, 30.0, 30.0, 30.0]
    assert EI == pytest.approx([300000.0, 300000.0, 300000.0, 300000.0])
    assert R == [-1, -1, -1, 0, 0, 0, -1, 0, -1, -1]
    assert eType == [1, 2, 1, 1]


def test_distcretization():
    """
    Confirm that poi rounding to find the closest idx for ILs works
    """

    beam_str = "P7R7R"
    (L, EI, R, eType) = cba.parse_beam_string(beam_str)

    ils = cba.InfluenceLines(L, EI, R, eType)
    ils.create_ils(step=0.1)
    (x, y) = ils.get_il(7.0, "M")
    assert np.linalg.norm(y) >= 5.7
