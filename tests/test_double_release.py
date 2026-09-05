"""Release regressions anchored to statics, not agreement between code paths."""
import numpy as np
import pytest
import pycba as cba
from pycba.load import LoadPL, LoadUDL, LoadPUDL
from pycba.section import SectionEI


@pytest.mark.parametrize("length", [3.0, 10.0, 27.0])
@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("magnitude", [100.0, -40.0])
@pytest.mark.parametrize("path", ["prismatic", "nonprismatic", "timoshenko"])
def test_double_release_point_load_statics(length, fraction, magnitude, path):
    beam = cba.Beam()
    load = LoadPL(0, magnitude, fraction * length)
    if path == "prismatic":
        actual = load.get_ref(length, 4)
    elif path == "nonprismatic":
        section = SectionEI([("linear", [0.0, length], [1e5, 3e5])])
        actual = beam._ref_nonprismatic(load, section, length, 4)
    else:
        actual = beam._ref_timoshenko(load, 1e5, 2.5e4, length, 4)
    expected = [magnitude * (1 - fraction), 0.0, magnitude * fraction, 0.0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("etype", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "load",
    [
        LoadPL(0, 100.0, 2.0),
        LoadPL(0, 100.0, 8.0),
        LoadUDL(0, 20.0),
        LoadPUDL(0, 20.0, 1.0, 3.0),
    ],
)
def test_release_vector_matches_matrix_condensation(etype, load):
    # Independent test-only elimination of rotations from the unreleased matrix.
    k = cba.Beam().k_FF(1e5, 10.0)
    f = np.array(load.get_cnl(10.0, 1))
    released = {1: [], 2: [3], 3: [1], 4: [1, 3]}[etype]
    expected = f.copy()
    if released:
        retained = [i for i in range(4) if i not in released]
        expected[retained] -= k[np.ix_(retained, released)] @ np.linalg.solve(
            k[np.ix_(released, released)], f[released]
        )
        expected[released] = 0.0
    np.testing.assert_allclose(
        load.get_ref(10.0, etype), expected, rtol=1e-10, atol=1e-8
    )


@pytest.mark.parametrize("path", ["prismatic", "nonprismatic", "timoshenko"])
@pytest.mark.parametrize("a", [2.0, 5.0, 8.0])
def test_suspended_span_global_equilibrium(path, a):
    lengths = [3.0, 10.0, 7.0]
    ei = [2e5, 1e5, 4e5]
    kwargs = {}
    if path == "nonprismatic":
        ei[1] = SectionEI([("linear", [0.0, 10.0], [1e5, 3e5])])
    elif path == "timoshenko":
        kwargs["GAv"] = 2.5e4
    analysis = cba.BeamAnalysis(
        lengths,
        ei,
        [-1, -1, 0, 0, 0, 0, -1, -1],
        [[2, 2, 100.0, a, 0]],
        eletype=[1, 4, 1],
        **kwargs,
    )
    analysis.analyze()
    va, vb = 100.0 * (1 - a / 10.0), 100.0 * a / 10.0
    actual = analysis.beam_results.R
    np.testing.assert_allclose(
        actual, [va, 3.0 * va, vb, -7.0 * vb], rtol=1e-10, atol=1e-8
    )
    ra, ma, rb, mb = actual
    assert ra + rb == pytest.approx(100.0, abs=1e-8)
    assert ma + mb + 20.0 * rb == pytest.approx(100.0 * (3.0 + a), abs=1e-8)
