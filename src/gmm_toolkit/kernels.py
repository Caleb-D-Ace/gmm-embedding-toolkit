"""
kernels holds the pure math: NumPy functions that take model arrays and return scores, with no word strings,
vocab, or sorting involved. A phase 2 torch backend would be a second module with these same signatures.

Every *_grid function compares ONE query word against M candidate words and returns a grid of shape (M, K, K):
grid[i, k, m] scores query sense k against candidate i's sense m. Comparing two words is just M = 1.
best_pair() and energy() then collapse each (K, K) grid into one score per candidate, honoring any fixed senses,
and both report which sense pair the score came from.

Shapes used throughout:
    q_mu, q_var, q_unit:  (K, D)     the query word's components
    c_mu, c_var, c_unit:  (M, K, D)  the candidates' components
    w_q: (K,)   w_c: (M, K)          mixture weights
"""
import math
from typing import Callable, NamedTuple, Optional

import numpy as np

# Cap on elements in any (chunk, K, K, D) intermediate, about 16MB at float32. Grids that need the full
# per-dimension difference are computed over the candidates in chunks of this size, so scoring a word against
# the whole vocabulary doesn't allocate a (vocab_size, K, K, D) array all at once.
_MAX_CHUNK_ELEMENTS = 2 ** 22


class Reduced(NamedTuple):
    """Per-candidate result of best_pair() or energy(). Each field has shape (M,)."""
    values: np.ndarray   # One score per candidate
    sense_a: np.ndarray  # The query sense of the pair the score came from
    sense_b: np.ndarray  # The candidate sense of that pair


def unit_rows(mu: np.ndarray) -> np.ndarray:
    """Scale every component mean (along the last axis) to unit length. A zero vector stays zero, so its cosine is 0."""
    norms = np.linalg.norm(mu, axis=-1, keepdims=True)
    return np.divide(mu, norms, out=np.zeros_like(mu), where=norms > 0)


def cosine_grid(q_unit: np.ndarray, c_unit: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every query/candidate component pair. Inputs must already be unit_rows()-normalized,
    which is what lets the model precompute them once at load time.

    Higher is more similar.
    """
    # (M, K, D) @ (D, K) -> (M, K_candidate, K_query); swap the last two axes to put the query sense first.
    # One batched matmul, so this is cheap enough to never need chunking.
    return np.matmul(c_unit, q_unit.T).transpose(0, 2, 1)


def euclidean_grid(q_mu: np.ndarray, c_mu: np.ndarray) -> np.ndarray:
    """
    Euclidean distance between every query/candidate component mean pair. Lower is more similar.

    Computed from the explicit difference rather than the faster ||a||² + ||b||² - 2a·b expansion, which loses
    most of its precision in float32 exactly when two means are close, i.e. for the pairs that matter most.
    """
    def block(c_mu):
        diff = q_mu[None, :, None, :] - c_mu[:, None, :, :]  # (m, K, K, D)
        return np.sqrt(np.sum(diff ** 2, axis=-1))
    return _chunked(block, c_mu)


def log_overlap_grid(q_mu: np.ndarray, q_var: np.ndarray, c_mu: np.ndarray, c_var: np.ndarray) -> np.ndarray:
    """
    Log expected likelihood kernel between every query/candidate component pair. Higher is more similar.
    A direct NumPy port of GMMWordEmbedding.log_overlap in the training repo.

    Math:
        log_Δ(f_k, g_m) = -0.5 * Σ_d [ log(var1 + var2) + (mu1 - mu2)^2 / (var1 + var2) ] - (D / 2) * log(2π)
    """
    D = q_mu.shape[-1]
    log_constant = 0.5 * D * math.log(2 * math.pi)

    def block(c_mu, c_var):
        var_sum = q_var[None, :, None, :] + c_var[:, None, :, :]  # (m, K, K, D)
        sq_dist = (q_mu[None, :, None, :] - c_mu[:, None, :, :]) ** 2
        return -0.5 * np.sum(np.log(var_sum) + sq_dist / var_sum, axis=-1) - log_constant
    return _chunked(block, c_mu, c_var)


def kl_grid(q_mu: np.ndarray, q_var: np.ndarray, c_mu: np.ndarray, c_var: np.ndarray,
            reverse: bool = False) -> np.ndarray:
    """
    KL divergence between every query/candidate component pair, treating each component as its own Gaussian.
    Lower means the first distribution is better "contained" by the second. Asymmetric.

    By default grid[i, k, m] = KL(query_k ‖ candidate_i,m), i.e. the query is the child: "how well does each
    candidate cover the query?". reverse=True gives KL(candidate_i,m ‖ query_k) instead, with the query as the
    parent, while keeping the same (M, K_query, K_candidate) layout.

    Math (diagonal Gaussians P, Q):
        KL(P ‖ Q) = 0.5 * Σ_d [ log(var_Q / var_P) + (var_P + (mu_P - mu_Q)^2) / var_Q - 1 ]
    """
    def block(c_mu, c_var):
        q = (q_mu[None, :, None, :], q_var[None, :, None, :])  # broadcast to (m, K, K, D)
        c = (c_mu[:, None, :, :], c_var[:, None, :, :])
        (p_mu, p_var), (r_mu, r_var) = (c, q) if reverse else (q, c)
        return 0.5 * np.sum(np.log(r_var / p_var) + (p_var + (p_mu - r_mu) ** 2) / r_var - 1, axis=-1)
    return _chunked(block, c_mu, c_var)


def contribution_grid(log_overlap: np.ndarray, w_q: np.ndarray, w_c: np.ndarray) -> np.ndarray:
    """
    Turn a log_overlap_grid() into each component pair's contribution to the energy: the overlap weighted by how
    much mixture weight both senses carry. Higher is more similar. A rare sense's pairs score lower in proportion
    to its rarity (by log w, e.g. about -1.6 for a weight of 0.2), even if its overlap is high.

    Math:
        contribution[k, m] = log w_q[k] + log w_c[m] + log_Δ(k, m)
    """
    with np.errstate(divide="ignore"):  # A weight that underflowed to exactly 0 gives log 0 = -inf, which logsumexp handles
        return np.log(w_q)[None, :, None] + np.log(w_c)[:, None, :] + log_overlap


def best_pair(grid: np.ndarray, higher_is_better: bool,
              sense_a: Optional[int] = None, sense_b: Optional[int] = None) -> Reduced:
    """
    Collapse a (M, K, K) grid to the best component pair per candidate: max for similarities, min for distances.

    sense_a / sense_b restrict the search to one query sense and/or one candidate sense:
        both given -> that single pair's score
        one given  -> best of that fixed sense against all of the other word's senses
        neither    -> best across every pair
    """
    M, K, _ = grid.shape
    mask = _sense_mask(sense_a, sense_b, K)
    worst = -np.inf if higher_is_better else np.inf
    masked = np.where(mask, grid, worst).reshape(M, K * K)

    flat = masked.argmax(axis=1) if higher_is_better else masked.argmin(axis=1)
    values = np.take_along_axis(masked, flat[:, None], axis=1)[:, 0]
    return Reduced(values=values, sense_a=flat // K, sense_b=flat % K)


def energy(contribution: np.ndarray, sense_a: Optional[int] = None, sense_b: Optional[int] = None) -> Reduced:
    """
    Collapse a (M, K, K) contribution_grid() to one expected likelihood energy per candidate. Higher is more similar.
    A direct NumPy port of GMMWordEmbedding.gmm_energy in the training repo, extended to support fixed senses.

    Math:
        E = logsumexp over the allowed pairs (k, m) of contribution[k, m]

    Unlike best_pair(), no single pair produces the energy: it's a soft sum over all of them. The reported senses
    are the pair contributing the most, i.e. the one the energy is mostly "about".

    With no senses fixed, every pair is allowed and this is exactly the energy training optimized. Fixing a sense
    restricts the sum to that sense's row and/or column, giving that sense's share of the full energy: because
    every sense keeps its mixture weight, the per-sense energies logsumexp back together into the full energy.
    """
    M, K, _ = contribution.shape
    mask = _sense_mask(sense_a, sense_b, K)
    masked = np.where(mask, contribution, -np.inf).reshape(M, K * K)

    flat = masked.argmax(axis=1)
    return Reduced(values=_logsumexp(masked), sense_a=flat // K, sense_b=flat % K)


def _sense_mask(sense_a: Optional[int], sense_b: Optional[int], K: int) -> np.ndarray:
    """(K, K) boolean mask of which component pairs are allowed, given optional fixed senses."""
    for name, sense in (("sense_a", sense_a), ("sense_b", sense_b)):
        if sense is not None and not (isinstance(sense, (int, np.integer)) and 0 <= sense < K):
            raise ValueError(f"{name} must be an integer in 0..{K - 1} (senses are 0-indexed), got {sense!r}.")
    rows = np.ones(K, dtype=bool) if sense_a is None else np.arange(K) == sense_a
    cols = np.ones(K, dtype=bool) if sense_b is None else np.arange(K) == sense_b
    return rows[:, None] & cols[None, :]


def _logsumexp(x: np.ndarray) -> np.ndarray:
    """log(Σ exp(x)) along the last axis, without exp() underflowing for large negative inputs."""
    peak = x.max(axis=-1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0)  # A row of all -inf would otherwise give -inf - -inf = nan
    with np.errstate(divide="ignore"):
        return peak[..., 0] + np.log(np.sum(np.exp(x - peak), axis=-1))


def _chunked(block: Callable[..., np.ndarray], *candidate_arrays: np.ndarray) -> np.ndarray:
    """Run block() over slices of the candidate arrays (split along axis 0) and concatenate the results."""
    M, K, D = candidate_arrays[0].shape
    chunk = max(1, _MAX_CHUNK_ELEMENTS // (K * K * D))
    if M <= chunk:
        return block(*candidate_arrays)
    return np.concatenate([block(*(a[start:start + chunk] for a in candidate_arrays))
                           for start in range(0, M, chunk)])
