import pytest

from gmm_toolkit import OutOfVocabularyError, Vocab, WordFrequency
from conftest import TOY_VOCAB


@pytest.fixture
def vocab(toy_vocab_path):
    return Vocab.load(toy_vocab_path)


def test_load_round_trips_ids(vocab):
    assert len(vocab) == len(TOY_VOCAB)
    for word, (word_id, _count) in TOY_VOCAB.items():
        assert vocab.id(word) == word_id
        assert vocab.word(word_id) == word


def test_iterates_in_id_order(vocab):
    assert list(vocab) == ["bank", "river", "money", "water"]


def test_lookup_is_case_and_whitespace_insensitive(vocab):
    assert vocab.id("BANK") == 0
    assert vocab.id("  River\n") == 1
    assert "Money" in vocab


def test_contains_rejects_missing_and_non_strings(vocab):
    assert "tree" not in vocab
    assert 0 not in vocab


def test_frequency(vocab):
    total = 50 + 30 + 15 + 5
    assert vocab.frequency("river") == WordFrequency(rank=1, count=30, share=30 / total)


def test_counts_is_read_only(vocab):
    assert list(vocab.counts) == [50, 30, 15, 5]
    with pytest.raises(ValueError):
        vocab.counts[0] = 1


def test_oov_is_a_key_error_with_details(vocab):
    with pytest.raises(KeyError) as exc_info:
        vocab.id("Tree")
    err = exc_info.value
    assert isinstance(err, OutOfVocabularyError)
    assert err.word == "Tree"
    assert err.normalized == "tree"
    assert "stopword" in str(err)


@pytest.mark.parametrize("word", ["ice cream", "loch-ness", "don't"])
def test_oov_explains_multi_token_input(vocab, word):
    with pytest.raises(OutOfVocabularyError, match="splits it into"):
        vocab.id(word)


def test_oov_explains_no_word_characters(vocab):
    with pytest.raises(OutOfVocabularyError, match="no word characters"):
        vocab.id("!!!")


@pytest.mark.parametrize("bad_vocab, message", [
    ({"a": [0, 5], "b": [0, 3]}, "assigned to both"),
    ({"a": [0, 5], "b": [2, 3]}, "outside the expected range"),
])
def test_rejects_malformed_ids(bad_vocab, message):
    with pytest.raises(ValueError, match=message):
        Vocab(bad_vocab)
