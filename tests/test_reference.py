"""
Cross-checks the NumPy port against the training repo's own GMMWordEmbedding, the model the formulas came from.

Skipped unless torch is installed and the training repo can be found: at ../gmm_program/gmm_word_embeddings next to
this repo by default, or wherever the GMM_TRAINING_REPO environment variable points. Nothing here needs a trained
model: a randomly initialized GMMWordEmbedding runs through the same math, and arrays are extracted from it the same
way GmmTrainer.save_model does.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from gmm_toolkit import GMMModel, Vocab, kernels

torch = pytest.importorskip("torch")

V, K, D = 40, 3, 8
_DEFAULT_REPO = Path(__file__).resolve().parents[2] / "gmm_program" / "gmm_word_embeddings"
_MODULE_PATH = Path(os.environ.get("GMM_TRAINING_REPO", _DEFAULT_REPO)) / "src" / "gmm_word_embedding.py"


@pytest.fixture(scope="module")
def reference():
    """(torch model, GMMModel built from its saved arrays), with the means spread out so scores vary between words."""
    if not _MODULE_PATH.exists():
        pytest.skip(f"Training repo not found at {_MODULE_PATH}; set GMM_TRAINING_REPO to run this test.")
    spec = importlib.util.spec_from_file_location("gmm_word_embedding", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(0)
    torch_model = module.GMMWordEmbedding(vocab_size=V, embedding_dim=D, K=K)
    with torch.no_grad():
        torch_model.mu_embeddings.weight.normal_(0.0, 1.0)
        torch_model.mix_logits.weight.normal_(0.0, 1.0)
        # The same extraction save_model does before writing gmm_embeddings.npz
        mu, var, weights = torch_model.get_word_params(torch.arange(V))

    vocab = Vocab({f"word{i}": [i, V - i] for i in range(V)})
    return torch_model, GMMModel(vocab, mu.numpy(), var.numpy(), weights.numpy())


def test_saved_arrays_match_documented_format(reference):
    _torch_model, model = reference
    assert (len(model), model.K, model.D) == (V, K, D)
    assert all(model.params(f"word{i}")[2].sum() == pytest.approx(1.0, abs=1e-6) for i in range(V))


def test_log_overlap_matches_reference(reference):
    torch_model, model = reference
    query, others = torch.tensor([3]), torch.arange(V)
    with torch.no_grad():
        mu1, var1, _ = torch_model.get_word_params(query.repeat(V))
        mu2, var2, _ = torch_model.get_word_params(others)
        expected = torch_model.log_overlap(mu1, mu2, var1, var2).numpy()

    mu, var, _w = model.params("word3")
    all_mu, all_var = np.stack([model.params(f"word{i}")[0] for i in range(V)]), np.stack([model.params(f"word{i}")[1] for i in range(V)])
    np.testing.assert_allclose(kernels.log_overlap_grid(mu, var, all_mu, all_var), expected, rtol=1e-5)


def test_energy_matches_reference_forward(reference):
    torch_model, model = reference
    pairs = [(i, j) for i in range(V) for j in range(V)]
    a, b = torch.tensor(pairs).T
    with torch.no_grad():
        expected = torch_model(a, b).numpy()

    actual = [model.similarity(f"word{i}", f"word{j}", "energy").value for i, j in pairs]
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_ranked_energy_matches_reference_forward(reference):
    # The whole-vocabulary path goes through different indexing than similarity(), so check it separately
    torch_model, model = reference
    with torch.no_grad():
        expected = torch_model(torch.full((V,), 7), torch.arange(V)).numpy()

    matches = {m.word: m.value for m in model.rank("word7", metric="energy")}
    np.testing.assert_allclose([matches[f"word{i}"] for i in range(V)], expected, rtol=1e-5)
    best = int(np.argmax(expected))
    assert model.rank("word7", metric="energy")[0].word == f"word{best}"
