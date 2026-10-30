"""
Small result types returned by the public API. NamedTuples so they unpack like plain tuples
(`value, (sense_a, sense_b) = model.similarity("bank", "river")`) but still print with field names.
"""
from typing import NamedTuple


class Score(NamedTuple):
    """
    A score between two words, and which sense pair it came from.

    Attributes:
        value: The score. Whether higher or lower is better depends on the metric (see gmm_toolkit.METRICS).
        senses: (sense of the first word, sense of the second word), 0-indexed. For best-pair metrics this is the
                pair that produced the value; for energy it's the pair contributing the most to it.
    """
    value: float
    senses: tuple[int, int]


class Match(NamedTuple):
    """One entry of a ranked list: a candidate word, its score against the query, and the sense pair (query, candidate)."""
    word: str
    value: float
    senses: tuple[int, int]


class WordFrequency(NamedTuple):
    """
    How common a word was in the training corpus.

    Attributes:
        rank: 0-indexed frequency rank (0 = most common word). Equal to the word's id, since
              sorted_vocab.json assigns ids in descending frequency order.
        count: Raw number of occurrences in the training corpus.
        share: count / (sum of every in-vocabulary word's count). NOT a share of the whole corpus:
               stopwords and words below MIN_FREQ were dropped before counting, and their totals
               aren't recorded anywhere in the model files.
    """
    rank: int
    count: int
    share: float
