"""
Moving load validation against McCarthy (2012) Section 7.3.1.

2span-30 bridge (two 15 m spans, three pin supports).
Single point load traversing at 1 m/s, step = 0.1 m.
Mp = 2694 kNm (Table 6.5).

Expected from Table 7.2:
  - Plastic hinges form at x = 6 m and x = 15 m
  - Collapse at approximately t = 5.93 s (load at ~5.9 m)

Cross section: 250 mm slab on steel plate girders
  (300x25 flanges, 20x730 web, 2650 mm spacing).
EI_composite ~ 1.87e6 kNm^2 (estimated).
"""

import sys
sys.path.insert(0, "/home/ccaprani/projects/pycba/src")

import numpy as np
from pycba.nonlinear import NonlinearBeamAnalysis


def find_collapse_load():
    """
    Binary search for the load magnitude that causes collapse
    during a single traverse of the 2span-30 bridge.
    """
    EI = 1_870_000.0  # kNm^2 (composite section estimate)
    Mp = 2694.0
    My = Mp / 1.15    # ~ 2343 kNm

    P_low, P_high = 500.0, 5000.0

    for _ in range(20):
        P_mid = (P_low + P_high) / 2
        nba = NonlinearBeamAnalysis(
            L=[15, 15], EI=EI,
            R=[-1, 0, -1, 0, -1, 0],
            Mp=Mp, My=My, q=0.0, mesh_size=0.5,
        )
        result = nba.analyze_moving(P=P_mid, step=0.1, n_sub=10)
        if result.collapsed:
            P_high = P_mid
        else:
            P_low = P_mid
        print(f"  P = {P_mid:8.1f} kN  collapsed={result.collapsed}"
              f"  x_collapse={result.collapse_lambda:.1f} m")

    return P_high


def test_moving_load():
    """
    Run the moving load analysis at the collapse load and report
    the hinge formation sequence.
    """
    print("=" * 60)
    print("McCarthy (2012) Section 7.3.1 — Moving Load Validation")
    print("2span-30: two 15 m spans, Mp = 2694 kNm")
    print("=" * 60)

    # First, find the collapse load
    print("\n--- Finding collapse load by bisection ---")
    P_collapse = find_collapse_load()
    print(f"\nCollapse load P ≈ {P_collapse:.1f} kN")

    # Compare with theoretical static collapse at midspan (x = 7.5):
    # lambda = 3*Mp / (7.5*P) -> P_static = 3*Mp/7.5 = 1077.6 kN (for lambda=1)
    # But for load at x=6: P_static = 7*Mp/18 = 1048 kN
    # For moving load, collapse P should be similar
    P_static_midspan = 3 * 2694 / 7.5
    print(f"Theoretical static collapse (at midspan): {P_static_midspan:.1f} kN")

    # Run detailed analysis at collapse load
    print(f"\n--- Detailed analysis at P = {P_collapse:.1f} kN ---")
    nba = NonlinearBeamAnalysis(
        L=[15, 15], EI=1_870_000.0,
        R=[-1, 0, -1, 0, -1, 0],
        Mp=2694.0, My=2694.0 / 1.15, q=0.0, mesh_size=0.5,
    )
    result = nba.analyze_moving(
        P=P_collapse, step=0.1, n_sub=10, record_every=5,
    )

    print(f"\nCollapsed: {result.collapsed}")
    print(f"Load position at collapse: x = {result.collapse_lambda:.1f} m")
    print(f"\nHinge events (plastic hinges):")
    for h in result.hinge_events:
        if h.event_type == "plastic_hinge":
            print(f"  x_load = {h.load_factor:5.1f} m  →  hinge at x = {h.location:.1f} m")

    print(f"\nYield events (initial yield):")
    for h in result.hinge_events:
        if h.event_type == "initial_yield":
            print(f"  x_load = {h.load_factor:5.1f} m  →  yield at x = {h.location:.1f} m")

    return result


if __name__ == "__main__":
    test_moving_load()
