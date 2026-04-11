"""
Validation of nonlinear module against McCarthy (2012) Table 4.1.

Three benchmark examples using a 457x152x74 UB continuous beam
(two equal spans of 12 m, three pin supports).

Section properties:
    Mp = 432 kNm (mean plastic moment capacity)
    EI ≈ 67,035 kNm² (E = 205 GPa, I = 32,700 cm⁴)
    q  = 0 (elastic-perfectly-plastic for validation)
"""

import sys
sys.path.insert(0, "/home/ccaprani/projects/pycba/src")

from pycba.nonlinear import NonlinearBeamAnalysis


def test_example1():
    """
    Example 1: Single point load P = 100 kN at midspan of span 1 (x = 6 m).

    Theoretical collapse: λ = 3Mp / (6P) = 3×432 / 600 = 2.16
    McCarthy NFEA result: 2.156
    """
    nba = NonlinearBeamAnalysis(
        L=[12, 12],
        EI=67035.0,
        R=[-1, 0, -1, 0, -1, 0],
        Mp=432.0,
        My=376.0,
        q=0.0,
        mesh_size=1.0,
    )
    result = nba.analyze(LM=[[1, 2, 100, 6]], lambda_max=5.0)

    lambda_theory = 3 * 432 / (6 * 100)
    print(f"Example 1: Point load at midspan")
    print(f"  Theoretical λ = {lambda_theory:.3f}")
    print(f"  NFEA λ         = {result.collapse_lambda:.3f}")
    print(f"  Collapsed       = {result.collapsed}")
    print(f"  Hinge events:")
    for h in result.hinge_events:
        if h.event_type == "plastic_hinge":
            print(f"    {h.event_type} at x={h.location:.1f}m, λ={h.load_factor:.3f}")
    print()
    return result


def test_example2():
    """
    Example 2: UDL w = 20 kN/m on span 1.

    Theoretical collapse: λ = 2.414Mp / (29.82w) = 2.414×432 / (29.82×20) ≈ 1.749
    McCarthy NFEA result: 1.722 (mesh effect — hinge at 0.414L ≈ 4.97m
                                 doesn't coincide with 1m mesh node)
    """
    nba = NonlinearBeamAnalysis(
        L=[12, 12],
        EI=67035.0,
        R=[-1, 0, -1, 0, -1, 0],
        Mp=432.0,
        My=376.0,
        q=0.0,
        mesh_size=1.0,
    )
    result = nba.analyze(LM=[[1, 1, 20]], lambda_max=5.0)

    lambda_theory = 2.414 * 432 / (29.82 * 20)
    print(f"Example 2: UDL on span 1")
    print(f"  Theoretical λ = {lambda_theory:.3f}")
    print(f"  NFEA λ         = {result.collapse_lambda:.3f}")
    print(f"  Collapsed       = {result.collapsed}")
    print(f"  Hinge events:")
    for h in result.hinge_events:
        if h.event_type == "plastic_hinge":
            print(f"    {h.event_type} at x={h.location:.1f}m, λ={h.load_factor:.3f}")
    print()
    return result


def test_example3():
    """
    Example 3: Two point loads, P1=100 kN at 4m, P2=75 kN at 8m.

    Theoretical collapse: λ = 2Mp / (4P1 + 2P2) = 864 / 550 ≈ 1.571
    McCarthy NFEA result: 1.568
    """
    nba = NonlinearBeamAnalysis(
        L=[12, 12],
        EI=67035.0,
        R=[-1, 0, -1, 0, -1, 0],
        Mp=432.0,
        My=376.0,
        q=0.0,
        mesh_size=1.0,
    )
    result = nba.analyze(
        LM=[[1, 2, 100, 4], [1, 2, 75, 8]],
        lambda_max=5.0,
    )

    lambda_theory = 2 * 432 / (4 * 100 + 2 * 75)
    print(f"Example 3: Two point loads")
    print(f"  Theoretical λ = {lambda_theory:.3f}")
    print(f"  NFEA λ         = {result.collapse_lambda:.3f}")
    print(f"  Collapsed       = {result.collapsed}")
    print(f"  Hinge events:")
    for h in result.hinge_events:
        if h.event_type == "plastic_hinge":
            print(f"    {h.event_type} at x={h.location:.1f}m, λ={h.load_factor:.3f}")
    print()
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("McCarthy (2012) Table 4.1 — NFEA Validation")
    print("=" * 60)
    print()
    test_example1()
    test_example2()
    test_example3()
