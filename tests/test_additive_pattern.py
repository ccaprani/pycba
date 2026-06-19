import numpy as np
import pycba as cba


def test_two_span_additive_envelope_equals_sum_of_two_cases():
    beam = cba.BeamAnalysis(
        L=[10.0, 10.0],
        EI=30e6,
        R=[-1, 0, -1, 0, -1, 0],
    )
    load_cases = cba.make_span_udl_cases(beam, w=10.0)

    x, B = cba.collect_response_matrix(beam, load_cases, response="M")
    env_neg, env_pos, idx_neg, idx_pos = cba.additive_envelope(B, n_combine=2)

    assert len(x) == B.shape[1]
    assert B.shape[0] == 2
    assert np.allclose(env_neg, B[0, :] + B[1, :])
    assert np.allclose(env_pos, B[0, :] + B[1, :])
    assert idx_neg.shape == (2, B.shape[1])
    assert idx_pos.shape == (2, B.shape[1])


def test_three_span_additive_envelope_is_pointwise_two_worst_values():
    beam = cba.BeamAnalysis(
        L=[6.0, 4.0, 6.0],
        EI=30e6,
        R=[-1, 0, -1, 0, -1, 0, -1, 0],
    )
    load_cases = cba.make_span_udl_cases(beam, w=10.0)

    _, B = cba.collect_response_matrix(beam, load_cases, response="M")
    env_neg, env_pos, idx_neg, idx_pos = cba.additive_envelope(B, n_combine=2)

    check_stations = [10, B.shape[1] // 3, B.shape[1] // 2, B.shape[1] - 10]
    for k in check_stations:
        values = B[:, k]
        order = np.argsort(values)
        assert env_neg[k] == np.sum(values[order[:2]])
        assert env_pos[k] == np.sum(values[order[-2:]])
        assert np.array_equal(idx_neg[:, k], order[:2])
        assert np.array_equal(idx_pos[:, k], order[-2:])

    neg_governing_sets = {tuple(sorted(idx_neg[:, k])) for k in range(B.shape[1])}
    pos_governing_sets = {tuple(sorted(idx_pos[:, k])) for k in range(B.shape[1])}
    assert len(neg_governing_sets) > 1
    assert len(pos_governing_sets) > 1


def test_additive_patterning_accepts_non_simple_support_layout():
    beam = cba.BeamAnalysis(
        L=[8.0, 5.0, 7.0],
        EI=30e6,
        R=[-1, -1, -1, 0, -1, 0, -1, 0],
    )
    load_cases = cba.make_span_udl_cases(beam, w=8.0)

    model = cba.build_pycba_model(beam, load_cases[0])
    assert model.beam.no_spans == 3

    x, B = cba.collect_response_matrix(beam, load_cases, response="M")
    env_neg, env_pos, idx_neg, idx_pos = cba.additive_envelope(B, n_combine=2)

    assert B.shape == (3, len(x))
    assert env_neg.shape == env_pos.shape == x.shape
    assert idx_neg.shape == idx_pos.shape == (2, len(x))
    assert np.any(np.abs(B) > 0.0)


def test_collect_response_matrix_accepts_raw_load_matrices():
    beam = cba.BeamAnalysis(
        L=[10.0, 10.0],
        EI=30e6,
        R=[-1, 0, -1, 0, -1, 0],
    )
    raw_load_cases = [
        [[1, 1, 10.0]],
        [[2, 1, 10.0]],
    ]

    x, B = cba.collect_response_matrix(beam, raw_load_cases, response="M")

    assert B.shape == (2, len(x))


def test_load_cases_collection_combines_with_arbitrary_factors():
    beam = cba.BeamAnalysis(
        L=[10.0, 10.0],
        EI=30e6,
        R=[-1, 0, -1, 0, -1, 0],
    )
    load_cases = cba.LoadCases(beam)
    load_cases.add("G", [[1, 1, 5.0], [2, 1, 5.0]])
    load_cases.add("Q1", [[1, 1, 10.0]])
    load_cases.add("Q2", [[2, 1, 10.0]])

    x, B = load_cases.response_matrix(response="M")
    _, y_vec = load_cases.combine([1.2, 1.5, 0.0], response="M")
    _, y_map = load_cases.combine({"G": 1.2, "Q1": 1.5}, response="M")
    combined_loads = load_cases.combined_loads({"G": 1.2, "Q1": 1.5})
    combined_analysis = load_cases.analyze_combination({"G": 1.2, "Q1": 1.5})
    _, y_rows = load_cases.combine(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.2, 1.5, 0.0],
            ]
        ),
        response="M",
    )

    assert B.shape == (3, len(x))
    assert np.allclose(y_vec, 1.2 * B[0, :] + 1.5 * B[1, :])
    assert np.allclose(y_map, y_vec)
    assert combined_loads == [[1, 1, 6.0], [2, 1, 6.0], [1, 1, 15.0]]
    assert np.allclose(combined_analysis.beam_results.results.M, y_vec)
    assert y_rows.shape == (2, len(x))
    assert np.allclose(y_rows[0, :], B[0, :])
    assert np.allclose(y_rows[1, :], y_vec)


def test_load_case_and_collection_have_high_level_load_builders():
    beam = cba.BeamAnalysis(
        L=[10.0, 10.0],
        EI=30e6,
        R=[-1, 0, -1, 0, -1, 0],
    )
    load_cases = cba.LoadCases(beam)

    g = cba.LoadCase("G").add_udl(1, 5.0).add_udl(2, 5.0)
    load_cases.append(g)
    load_cases.add_pl("Q", 1, 20.0, 4.0)
    load_cases.add_pudl("Q", 2, 8.0, 1.0, 3.0)

    assert load_cases.names == ("G", "Q")
    assert load_cases[0].loads == [[1, 1, 5.0], [2, 1, 5.0]]
    assert load_cases[1].loads == [[1, 2, 20.0, 4.0], [2, 3, 8.0, 1.0, 3.0]]

    x, B = load_cases.response_matrix(response="M")
    assert B.shape == (2, len(x))


def test_load_pattern_accepts_load_case_inputs():
    beam = cba.BeamAnalysis(
        L=[10.0, 10.0],
        EI=30e6,
        R=[-1, 0, -1, 0, -1, 0],
    )
    load_cases = cba.LoadCases(beam)
    dead = load_cases.add_case("G").add_udl(1, 5.0).add_udl(2, 5.0)
    live = load_cases.add_case("Q").add_udl(1, 10.0).add_udl(2, 10.0)

    lp = cba.LoadPattern(beam)
    lp.set_dead_loads(dead, gamma_max=1.35, gamma_min=0.9)
    lp.set_live_loads(live, gamma_max=1.5, gamma_min=0.0)

    pattern_cases = lp.to_load_cases()
    env = lp.analyze(npts=20)

    assert isinstance(pattern_cases, cba.LoadCases)
    assert pattern_cases.names == (
        "Max hogging spans 1-2",
        "Max odd spans",
        "Max even spans",
        "All spans max",
    )
    assert pattern_cases.case("Max even spans").loads == [
        [1, 1, 4.5],
        [2, 1, 6.75],
        [1, 1, 0.0],
        [2, 1, 15.0],
    ]
    assert lp.LMg == dead.loads
    assert lp.LMq == live.loads
    assert env is not None


def test_load_pattern_plots_return_figures_without_showing(monkeypatch):
    def fail_show():
        raise AssertionError("plot function should respect show=False")

    monkeypatch.setattr(cba.pattern.plt, "show", fail_show)

    beam = cba.BeamAnalysis(
        L=[5.0, 7.0, 4.0],
        EI=30e6,
        R=[-1, 0, -1, 0, -1, 0, -1, 0],
    )
    load_cases = cba.make_span_udl_cases(beam, w=5.0)
    x, B = cba.collect_response_matrix(beam, load_cases, response="M")
    env_neg, env_pos, _, _ = cba.additive_envelope(B, n_combine=2)

    fig1, ax1 = cba.plot_response_envelope(
        x, B, env_neg, env_pos, beam, load_cases, show=False
    )
    fig2, ax2 = cba.plot_load_patterns(beam, load_cases, show=False)

    assert fig1 is not None
    assert ax1 is not None
    assert fig2 is not None
    assert ax2 is not None

    cba.pattern.plt.close(fig1)
    cba.pattern.plt.close(fig2)
