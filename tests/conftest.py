import json

import numpy as np
import pytest

# A tiny vocabulary in the same {word: [id, count]} format preprocess.py writes, ids in descending-count order
TOY_VOCAB = {
    "bank": [0, 50],
    "river": [1, 30],
    "money": [2, 15],
    "water": [3, 5],
    "nature": [4, 4],
    "tree": [5, 3],
}

# A hand-built K=2, D=3 model with known answers. Axes: 0 = finance, 1 = water, 2 = nature.
# - bank is two-sense: a dominant finance sense (0) and a rarer water sense (1)
# - river/money each sit on one axis, with a slight second sense
# - nature is broad (high variance) around the nature axis and tree is narrow inside it, for hypernym tests
TOY_MU = np.array([
    [[1, 0, 0], [0, 1, 0]],        # bank
    [[0, 1, 0], [0, 1, 0.2]],      # river
    [[1, 0, 0], [1, 0, 0.2]],      # money
    [[0, 1, 0.3], [0, 0.5, 1]],    # water
    [[0, 0, 1], [0, 0.3, 1]],      # nature
    [[0, 0, 1], [0.1, 0, 1]],      # tree
], dtype=np.float32)
TOY_VAR = np.full(TOY_MU.shape, 0.5, dtype=np.float32)
TOY_VAR[4] = 2.0  # nature
TOY_WEIGHTS = np.array([[0.8, 0.2], [0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.7, 0.3], [0.5, 0.5]], dtype=np.float32)


@pytest.fixture
def toy_vocab_path(tmp_path):
    path = tmp_path / "sorted_vocab.json"
    path.write_text(json.dumps(TOY_VOCAB), encoding="utf-8")
    return path


@pytest.fixture
def toy_model_dir(tmp_path, toy_vocab_path):
    """A directory laid out exactly like the training repo's data/model/ output."""
    np.savez(tmp_path / "gmm_embeddings.npz", mu=TOY_MU, var=TOY_VAR, mix_weights=TOY_WEIGHTS)
    return tmp_path
