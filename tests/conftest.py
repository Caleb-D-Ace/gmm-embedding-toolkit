import json

import pytest

# A tiny vocabulary in the same {word: [id, count]} format preprocess.py writes, ids in descending-count order
TOY_VOCAB = {
    "bank": [0, 50],
    "river": [1, 30],
    "money": [2, 15],
    "water": [3, 5],
}


@pytest.fixture
def toy_vocab_path(tmp_path):
    path = tmp_path / "sorted_vocab.json"
    path.write_text(json.dumps(TOY_VOCAB), encoding="utf-8")
    return path
