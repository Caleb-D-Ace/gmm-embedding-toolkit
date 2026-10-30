import numpy as np
import pytest

from gmm_toolkit import METRICS, GMMModel, Match, OutOfVocabularyError, Score, Vocab, kernels
from conftest import TOY_MU, TOY_VAR, TOY_VOCAB, TOY_WEIGHTS


@pytest.fixture
def model(toy_model_dir):
    return GMMModel.load(toy_model_dir)


# --- Loading ---

def test_load(model):
    assert (len(model), model.K, model.D) == (6, 2, 3)
    assert "BANK" in model and "oak" not in model


@pytest.mark.parametrize("mu, var, w, message", [
    (TOY_MU[0], TOY_VAR, TOY_WEIGHTS, "mu must have shape"),
    (TOY_MU, TOY_VAR[:, :, :2], TOY_WEIGHTS, "var has shape"),
    (TOY_MU, TOY_VAR, TOY_WEIGHTS[:, :1], "mix_weights has shape"),
    (TOY_MU[:5], TOY_VAR[:5], TOY_WEIGHTS[:5], "different training runs"),
])
def test_rejects_mismatched_arrays(toy_vocab_path, mu, var, w, message):
    with pytest.raises(ValueError, match=message):
        GMMModel(Vocab.load(toy_vocab_path), mu, var, w)


def test_load_rejects_missing_array(toy_model_dir):
    np.savez(toy_model_dir / "gmm_embeddings.npz", mu=TOY_MU, var=TOY_VAR)
    with pytest.raises(ValueError, match="mix_weights"):
        GMMModel.load(toy_model_dir)


def test_model_is_read_only_but_callers_arrays_are_not(toy_vocab_path):
    mu = TOY_MU.copy()
    model = GMMModel(Vocab.load(toy_vocab_path), mu, TOY_VAR.copy(), TOY_WEIGHTS.copy())
    with pytest.raises(ValueError):
        model.params("bank")[0][0, 0] = 5.0
    mu[0, 0, 0] = 5.0  # The caller's own array is untouched by the model's read-only flag


# --- Single-word lookups ---

def test_params(model):
    mu, var, w = model.params("Bank")
    np.testing.assert_array_equal(mu, TOY_MU[0])
    np.testing.assert_array_equal(var, TOY_VAR[0])
    np.testing.assert_array_equal(w, TOY_WEIGHTS[0])


def test_dominant_sense_weight(model):
    assert model.dominant_sense_weight("bank") == pytest.approx(0.8)
    assert model.dominant_sense_weight("tree") == pytest.approx(0.5)


def test_frequency(model):
    assert model.frequency("river").rank == 1


def test_oov_raises_everywhere(model):
    with pytest.raises(OutOfVocabularyError):
        model.similarity("bank", "oak")
    with pytest.raises(OutOfVocabularyError):
        model.rank("bank", ["river", "oak"])
    with pytest.raises(OutOfVocabularyError):
        model.most_similar("oak")


# --- sense_matrix / similarity ---

@pytest.mark.parametrize("metric", sorted(METRICS))
def test_similarity_is_the_reduction_of_sense_matrix(model, metric):
    matrix = model.sense_matrix("bank", "water", metric)
    assert matrix.shape == (2, 2)
    score = model.similarity("bank", "water", metric)
    if metric == "energy":
        assert score.value == pytest.approx(np.log(np.exp(matrix.astype(np.float64)).sum()), rel=1e-6)
        assert score.senses == np.unravel_index(matrix.argmax(), matrix.shape)
    else:
        best = matrix.max() if METRICS[metric].higher_is_better else matrix.min()
        assert score.value == pytest.approx(best)
        assert matrix[score.senses] == pytest.approx(best)


def test_sense_matrix_rows_are_first_word(model):
    # bank sense 1 (water) vs river sense 0 (water) is the identical direction
    matrix = model.sense_matrix("bank", "river")
    assert matrix[1, 0] == pytest.approx(1.0)
    assert matrix[0, 0] == pytest.approx(0.0)


def test_similarity_picks_the_matching_sense(model):
    assert model.similarity("bank", "river") == Score(pytest.approx(1.0), (1, 0))
    assert model.similarity("bank", "money") == Score(pytest.approx(1.0), (0, 0))


def test_similarity_fixed_senses(model):
    # Fixing bank to its finance sense makes it unrelated to river
    assert model.similarity("bank", "river", sense_a=0).value == pytest.approx(0.0)
    assert model.similarity("bank", "river", sense_a=1, sense_b=1).senses == (1, 1)


def test_score_unpacks_as_value_and_senses(model):
    value, (a, b) = model.similarity("bank", "river")
    assert (a, b) == (1, 0)


def test_energy_matches_kernels_directly(model):
    lo = kernels.log_overlap_grid(TOY_MU[0], TOY_VAR[0], TOY_MU[1:2], TOY_VAR[1:2])
    expected = kernels.energy(kernels.contribution_grid(lo, TOY_WEIGHTS[0], TOY_WEIGHTS[1:2]))
    assert model.similarity("bank", "river", "energy").value == pytest.approx(float(expected.values[0]))


def test_energy_discounts_rare_senses(model):
    # bank-river and bank-money both have a perfect cosine match, but bank's water sense is the rarer one
    assert model.similarity("bank", "money", "energy").value > model.similarity("bank", "river", "energy").value


def test_unknown_metric(model):
    with pytest.raises(ValueError, match="Unknown metric"):
        model.similarity("bank", "river", "dot")


# --- rank / most_similar ---

def test_rank_sorts_best_first_in_each_direction(model):
    board = ["tree", "river", "water", "money"]
    for metric, info in METRICS.items():
        values = [m.value for m in model.rank("bank", board, metric)]
        assert values == sorted(values, reverse=info.higher_is_better), metric


def test_rank_values_match_similarity(model):
    for match in model.rank("bank", ["tree", "river"], "overlap"):
        assert (match.value, match.senses) == tuple(model.similarity("bank", match.word, "overlap"))


def test_rank_ties_keep_candidate_order(model):
    # money and river are both an exact (distance 0) match for one of bank's senses
    assert [m.word for m in model.rank("bank", ["river", "money"], "euclidean")] == ["river", "money"]
    assert [m.word for m in model.rank("bank", ["money", "river"], "euclidean")] == ["money", "river"]


def test_rank_whole_vocab_and_n(model):
    everything = model.rank("bank")
    assert len(everything) == len(TOY_VOCAB)
    assert everything[0] == Match("bank", pytest.approx(1.0), (0, 0))  # rank() keeps the word itself
    assert model.rank("bank", n=2) == everything[:2]


def test_most_similar_excludes_self(model):
    neighbors = model.most_similar("bank", n=3)
    assert len(neighbors) == 3
    assert "bank" not in [m.word for m in neighbors]
    assert {m.word for m in neighbors[:2]} == {"river", "money"}


def test_most_similar_with_fixed_sense(model):
    # bank's finance sense's nearest neighbor is money, its water sense's is river
    assert model.most_similar("bank", n=1, sense_a=0)[0].word == "money"
    assert model.most_similar("bank", n=1, sense_a=1)[0].word == "river"


# --- Hypernyms ---

def test_hypernym_score_is_directional(model):
    child_to_parent = model.hypernym_score("tree", "nature")
    parent_to_child = model.hypernym_score("nature", "tree")
    assert child_to_parent.value < parent_to_child.value
    assert child_to_parent == model.similarity("tree", "nature", "kl")


def test_hypernyms_and_hyponyms(model):
    assert model.hypernyms("tree", candidates=["money", "nature"])[0].word == "nature"
    children = model.hyponyms("nature", n=1)
    assert children[0].word == "tree"
    # Reported senses are (query sense, candidate sense) in both directions
    assert children[0].senses == model.hypernym_score("tree", "nature").senses[::-1]


def test_hypernyms_exclude_self(model):
    assert "tree" not in [m.word for m in model.hypernyms("tree", n=None)]
    assert "nature" not in [m.word for m in model.hyponyms("nature", n=None)]
