import math

import numpy as np
import pytest

from gmm_toolkit import kernels

rng = np.random.default_rng(0)
K, D, M = 3, 5, 7


@pytest.fixture
def params():
    """One random query word and M random candidates, float64 so comparisons can be tight."""
    return dict(
        q_mu=rng.normal(size=(K, D)), q_var=rng.uniform(0.1, 2.0, size=(K, D)),
        c_mu=rng.normal(size=(M, K, D)), c_var=rng.uniform(0.1, 2.0, size=(M, K, D)),
        w_q=rng.dirichlet(np.ones(K)), w_c=rng.dirichlet(np.ones(K), size=M),
    )


# --- Reference implementations: plain loops over the formulas, one component pair at a time ---

def ref_log_overlap(mu1, var1, mu2, var2):
    """log N(mu1; mu2, var1 + var2), written as a product of 1D Gaussian densities."""
    s = var1 + var2
    return np.sum(-0.5 * np.log(2 * math.pi * s) - (mu1 - mu2) ** 2 / (2 * s))


def ref_kl(mu_p, var_p, mu_q, var_q):
    """KL(P ‖ Q) for diagonal Gaussians, one dimension at a time."""
    return sum(0.5 * (math.log(vq / vp) + (vp + (mp - mq) ** 2) / vq - 1)
               for mp, vp, mq, vq in zip(mu_p, var_p, mu_q, var_q))


def ref_grid(fn, q_mu, q_var, c_mu, c_var):
    return np.array([[[fn(q_mu[k], q_var[k], c_mu[i, m], c_var[i, m]) for m in range(K)]
                      for k in range(K)] for i in range(M)])


# --- unit_rows / grids ---

def test_unit_rows_normalizes_and_keeps_zero_vectors():
    mu = np.array([[3.0, 4.0], [0.0, 0.0]])
    np.testing.assert_allclose(kernels.unit_rows(mu), [[0.6, 0.8], [0.0, 0.0]])


def test_cosine_grid_matches_loop(params):
    q_unit, c_unit = kernels.unit_rows(params["q_mu"]), kernels.unit_rows(params["c_mu"])
    expected = ref_grid(lambda a, _va, b, _vb: a @ b / (np.linalg.norm(a) * np.linalg.norm(b)),
                        params["q_mu"], params["q_var"], params["c_mu"], params["c_var"])
    np.testing.assert_allclose(kernels.cosine_grid(q_unit, c_unit), expected)


def test_euclidean_grid_matches_loop(params):
    expected = ref_grid(lambda a, _va, b, _vb: np.linalg.norm(a - b),
                        params["q_mu"], params["q_var"], params["c_mu"], params["c_var"])
    np.testing.assert_allclose(kernels.euclidean_grid(params["q_mu"], params["c_mu"]), expected)


def test_log_overlap_grid_matches_loop(params):
    p = params
    expected = ref_grid(ref_log_overlap, p["q_mu"], p["q_var"], p["c_mu"], p["c_var"])
    np.testing.assert_allclose(kernels.log_overlap_grid(p["q_mu"], p["q_var"], p["c_mu"], p["c_var"]), expected)


def test_log_overlap_hand_value():
    # Two 1D unit-variance-sum Gaussians at the same mean: log N(0; 0, 1) = -0.5 * log(2π)
    grid = kernels.log_overlap_grid(np.zeros((1, 1)), np.full((1, 1), 0.5), np.zeros((1, 1, 1)), np.full((1, 1, 1), 0.5))
    assert grid[0, 0, 0] == pytest.approx(-0.5 * math.log(2 * math.pi))


def test_kl_grid_matches_loop_in_both_directions(params):
    p = params
    forward = ref_grid(ref_kl, p["q_mu"], p["q_var"], p["c_mu"], p["c_var"])
    backward = ref_grid(lambda mq, vq, mc, vc: ref_kl(mc, vc, mq, vq), p["q_mu"], p["q_var"], p["c_mu"], p["c_var"])
    np.testing.assert_allclose(kernels.kl_grid(p["q_mu"], p["q_var"], p["c_mu"], p["c_var"]), forward)
    np.testing.assert_allclose(kernels.kl_grid(p["q_mu"], p["q_var"], p["c_mu"], p["c_var"], reverse=True), backward)


def test_kl_hand_values():
    # KL(N(0,1) ‖ N(1,2)) = 0.5 * (log 2 + (1 + 1) / 2 - 1) = 0.5 * log 2, and KL of a Gaussian with itself is 0
    q_mu, q_var = np.array([[0.0], [1.0]]), np.array([[1.0], [2.0]])
    c_mu, c_var = q_mu[None], q_var[None]
    grid = kernels.kl_grid(q_mu, q_var, c_mu, c_var)
    assert grid[0, 0, 1] == pytest.approx(0.5 * math.log(2))
    assert grid[0, 0, 0] == pytest.approx(0.0)
    assert grid[0, 1, 1] == pytest.approx(0.0)


def test_chunking_gives_same_result(params, monkeypatch):
    p = params
    whole = kernels.log_overlap_grid(p["q_mu"], p["q_var"], p["c_mu"], p["c_var"])
    monkeypatch.setattr(kernels, "_MAX_CHUNK_ELEMENTS", K * K * D * 2)  # 2 candidates per chunk -> 4 uneven chunks
    np.testing.assert_allclose(kernels.log_overlap_grid(p["q_mu"], p["q_var"], p["c_mu"], p["c_var"]), whole)


# --- best_pair ---

# One candidate, K=2: grid[k, m] = query sense k vs candidate sense m
HAND_GRID = np.array([[[0.1, 0.9],
                       [0.4, 0.2]]])


@pytest.mark.parametrize("higher, sense_a, sense_b, value, a, b", [
    (True, None, None, 0.9, 0, 1),
    (False, None, None, 0.1, 0, 0),
    (True, 1, None, 0.4, 1, 0),
    (True, None, 0, 0.4, 1, 0),
    (False, None, 1, 0.2, 1, 1),
    (True, 1, 1, 0.2, 1, 1),
])
def test_best_pair_respects_direction_and_fixed_senses(higher, sense_a, sense_b, value, a, b):
    result = kernels.best_pair(HAND_GRID, higher, sense_a, sense_b)
    assert result.values[0] == pytest.approx(value)
    assert (result.sense_a[0], result.sense_b[0]) == (a, b)


@pytest.mark.parametrize("bad", [-1, 2, 1.0, "0"])
def test_invalid_sense_is_rejected(bad):
    with pytest.raises(ValueError, match="0-indexed"):
        kernels.best_pair(HAND_GRID, True, sense_a=bad)


# --- energy ---

def test_energy_unfixed_matches_direct_formula(params):
    p = params
    lo = kernels.log_overlap_grid(p["q_mu"], p["q_var"], p["c_mu"], p["c_var"])
    expected = [math.log(sum(p["w_q"][k] * p["w_c"][i, m] * math.exp(lo[i, k, m]) for k in range(K) for m in range(K)))
                for i in range(M)]
    for weighted in (False, True):  # The flag makes no difference when nothing is fixed
        np.testing.assert_allclose(kernels.energy(lo, p["w_q"], p["w_c"], weight_fixed_senses=weighted), expected)


def test_energy_both_fixed(params):
    p = params
    lo = kernels.log_overlap_grid(p["q_mu"], p["q_var"], p["c_mu"], p["c_var"])
    np.testing.assert_allclose(kernels.energy(lo, p["w_q"], p["w_c"], 2, 0), lo[:, 2, 0])
    np.testing.assert_allclose(kernels.energy(lo, p["w_q"], p["w_c"], 2, 0, weight_fixed_senses=True),
                               np.log(p["w_q"][2]) + np.log(p["w_c"][:, 0]) + lo[:, 2, 0])


def test_energy_one_fixed(params):
    p = params
    lo = kernels.log_overlap_grid(p["q_mu"], p["q_var"], p["c_mu"], p["c_var"])
    # Flat: query sense 1 as a standalone Gaussian against each candidate's whole mixture
    flat = [math.log(sum(p["w_c"][i, m] * math.exp(lo[i, 1, m]) for m in range(K))) for i in range(M)]
    np.testing.assert_allclose(kernels.energy(lo, p["w_q"], p["w_c"], sense_a=1), flat)
    # Weighted: the same, scaled by query sense 1's weight
    np.testing.assert_allclose(kernels.energy(lo, p["w_q"], p["w_c"], sense_a=1, weight_fixed_senses=True),
                               np.log(p["w_q"][1]) + np.array(flat))


def test_energy_weighted_contributions_add_up_to_full_energy(params):
    # With weight_fixed_senses=True, every pair's energy is its share of the full energy
    p = params
    lo = kernels.log_overlap_grid(p["q_mu"], p["q_var"], p["c_mu"], p["c_var"])
    pairs = [kernels.energy(lo, p["w_q"], p["w_c"], k, m, weight_fixed_senses=True) for k in range(K) for m in range(K)]
    np.testing.assert_allclose(np.log(np.sum(np.exp(pairs), axis=0)), kernels.energy(lo, p["w_q"], p["w_c"]))


def test_energy_does_not_underflow():
    # exp(-1000) is 0 in float64, so a naive log(Σ w w exp(lo)) would give -inf here
    lo = np.full((1, 2, 2), -1000.0)
    w = np.array([0.5, 0.5])
    assert kernels.energy(lo, w, w[None]) == pytest.approx([-1000.0])
