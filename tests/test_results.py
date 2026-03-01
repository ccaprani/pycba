import pytest
import pycba

def test_sum_compatible_envelopes():
    """Test summing two compatible envelopes"""
    # Create the beam configuration
    L = [30,30,30]
    EI = 30 * 10e9 * 1e-6
    R = [-1,0,-1,0,-1,0,-1,0]

    bridge = pycba.BeamAnalysis(L, EI, R)
    
    # Specify the uniform distributed loads for self-weight and permanent loads
    LMg = [[1,1,12.5,0,0],
            [2,1,12.5,0,0],
            [3,1,12.5,0,0]]
    gammag_max = 1.35
    gammag_min = 1.0

    LMp = [[1,1,50,0,0],
            [2,1,50,0,0],
            [3,1,50,0,0]]
    gammap_max = 1.5
    gammap_min = 1.0

    # Compose the matrix of self-weight + permanent loads
    LMd = []
    for count, load in enumerate(LMg):
        sum_load = LMg[count][2] * gammag_max + LMp[count][2] * gammap_max
        span_vector = [LMg[count][0], LMg[count][1], sum_load, LMg[count][3], LMg[count][4]]
        LMd.append(span_vector)
    gammad_max = 1.0
    gammad_min = 1.0

    # Define the matrix of udl variable loads
    LMq = [[1,1,13.5,0,0],
            [2,1,13.5,0,0],
            [3,1,13.5,0,0]]
    gammaq_max = 1.50
    gammaq_min = 0

    # Create the load pattern for dead and live loads
    loadpattern = pycba.LoadPattern(bridge)
    loadpattern.set_dead_loads(LMd, gammad_max, gammad_min)
    loadpattern.set_live_loads(LMq, gammaq_max, gammaq_min)
    
    # Define the first envelope to sum - Uniform distributed loads
    env_udl = loadpattern.analyze()

    # Define vehicles and pycba bridge analysis
    vehicle_weights = [600, 600]
    vehicle_spacing = [1.20]
    vehicle_weights_fact = []
    gammaq_max_traffic = 1.35
    for load in vehicle_weights:
        factored = load * gammaq_max_traffic
        vehicle_weights_fact.append(factored)
    
    vehicle = pycba.Vehicle(vehicle_spacing, vehicle_weights)
    bridge_analysis = pycba.BridgeAnalysis(bridge, vehicle)
    
    # Define the second envelope to sum - Vehicle loads
    env_veh = bridge_analysis.run_vehicle(0.1)

    # Sum the envelopes
    env_sum = pycba.Envelopes.zero_like(env_udl)
    env_sum.sum(env_udl)
    env_sum.sum(env_veh)

    # Extrapolate the results for checks
    x_axis = env_sum.x

    # Single envelopes results:
    env_udl_mmax = env_udl.Mmax
    env_udl_mmin = env_udl.Mmin

    env_veh_mmax = env_veh.Mmax
    env_veh_mmin = env_veh.Mmin

    results_env_udl_max = env_udl_mmax[(0 <= x_axis) & (x_axis <= 10)]
    results_env_udl_min = env_udl_mmin[(0 <= x_axis) & (x_axis <= 10)]

    results_env_veh_max = env_veh_mmax[(0 <= x_axis) & (x_axis <= 10)]
    results_env_veh_min = env_veh_mmin[(0 <= x_axis) & (x_axis <= 10)]

    # Sum of envelopes results:
    env_sum_mmax = env_sum.Mmax
    env_sum_mmin = env_sum.Mmin

    results_env_sum_max = env_sum_mmax[(0 <= x_axis) & (x_axis <= 10)]
    results_env_sum_min = env_sum_mmin[(0 <= x_axis) & (x_axis <= 10)]

    assert results_env_sum_max.max() == (results_env_udl_max.max()+results_env_veh_max.max())
    assert results_env_sum_min.min() == (results_env_udl_min.min()+results_env_veh_min.min())

def test_sum_incompatible_envelopes():
    """Test error when summing incompatible envelopes"""
    # Create the first beam configuration
    L1 = [30, 30, 30]
    EI1 = 30 * 10e9 * 1e-6
    R1 = [-1, 0, -1, 0, -1, 0, -1, 0]
    bridge1 = pycba.BeamAnalysis(L1, EI1, R1)

    # Create the second beam configuration with different length
    L2 = [20, 20]  # Different span configuration
    EI2 = 30 * 10e9 * 1e-6
    R2 = [-1, 0, -1, 0, -1, 0]  # Different number of supports
    bridge2 = pycba.BeamAnalysis(L2, EI2, R2)

    # Define loads for first envelope
    LMq1 = [[1, 1, 13.5, 0, 0],
            [2, 1, 13.5, 0, 0],
            [3, 1, 13.5, 0, 0]]
    LMg1 = [[1, 1, 12.5, 0, 0],  # Added dead loads
            [2, 1, 12.5, 0, 0],
            [3, 1, 12.5, 0, 0]]
    gammaq_max1 = 1.50
    gammaq_min1 = 0
    gammag_max1 = 1.35
    gammag_min1 = 1.0

    # Define loads for second envelope
    LMq2 = [[1, 1, 13.5, 0, 0],
            [2, 1, 13.5, 0, 0]]
    LMg2 = [[1, 1, 12.5, 0, 0],  # Added dead loads
            [2, 1, 12.5, 0, 0]]
    gammaq_max2 = 1.50
    gammaq_min2 = 0
    gammag_max2 = 1.35
    gammag_min2 = 1.0

    # Create load patterns and analyze
    loadpattern1 = pycba.LoadPattern(bridge1)
    loadpattern1.set_dead_loads(LMg1, gammag_max1, gammag_min1)  # Set dead loads
    loadpattern1.set_live_loads(LMq1, gammaq_max1, gammaq_min1)
    env1 = loadpattern1.analyze()

    loadpattern2 = pycba.LoadPattern(bridge2)
    loadpattern2.set_dead_loads(LMg2, gammag_max2, gammag_min2)  # Set dead loads
    loadpattern2.set_live_loads(LMq2, gammaq_max2, gammaq_min2)
    env2 = loadpattern2.analyze()

    # Create zero envelope for summing
    env_sum = pycba.Envelopes.zero_like(env1)

    # Try to sum incompatible envelopes
    # This should raise a ValueError because the envelopes have different geometries
    with pytest.raises(ValueError) as excinfo:
        env_sum.sum(env2)
    
    assert "Cannot sum with an inconsistent envelope" in str(excinfo.value)
