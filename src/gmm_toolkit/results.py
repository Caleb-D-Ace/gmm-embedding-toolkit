"""
Small result types returned by the public API. NamedTuples so they unpack like plain tuples
(`rank, count, share = vocab.frequency("bank")`) but still print with field names.
"""
from typing import NamedTuple


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
