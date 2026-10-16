"""
vocab handles everything to do with word strings: loading sorted_vocab.json, normalizing lookups the
same way training tokenized them, translating word <-> id, and frequency information.
Nothing outside this module should need to touch a raw word string.
"""
import json
import re
from pathlib import Path
from typing import Iterator, Union

import numpy as np

from .errors import OutOfVocabularyError
from .results import WordFrequency

# Must match tokenize() in the training repo's src/preprocess.py: every token is a run of \w characters,
# lowercased. Anything that doesn't match as exactly one token can never have been in the vocabulary.
_TOKEN_REGEX = re.compile(r'\w+')


class Vocab:
    """
    The word <-> id mapping and frequency counts from a trained model's sorted_vocab.json.

    Row i of every model array belongs to the word with id i, so this is needed to interpret them.
    """

    def __init__(self, vocab: dict[str, list[int]]):
        """
        Args:
            vocab: The parsed contents of sorted_vocab.json, i.e. {word: [id, count]}.

        Raises:
            ValueError: If the ids aren't exactly 0..len(vocab)-1, each used once.
        """
        size = len(vocab)
        self._words: list[Union[str, None]] = [None] * size
        self._counts = np.zeros(size, dtype=np.int64)

        for word, (word_id, count) in vocab.items():
            if not 0 <= word_id < size:
                raise ValueError(f"Word {word!r} has id {word_id}, outside the expected range 0..{size - 1}.")
            if self._words[word_id] is not None:
                raise ValueError(f"Id {word_id} is assigned to both {self._words[word_id]!r} and {word!r}.")
            self._words[word_id] = word
            self._counts[word_id] = count

        # Every id in range was filled exactly once (no duplicates above, and len(vocab) == size), so no gaps remain
        self._ids = {word: word_id for word_id, word in enumerate(self._words)}
        self._total_count = int(self._counts.sum())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocab":
        """Load a vocabulary from a sorted_vocab.json file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    @staticmethod
    def normalize(word: str) -> str:
        """Normalize a lookup the same way training tokenized the corpus: strip surrounding whitespace, lowercase."""
        return word.strip().lower()

    def id(self, word: str) -> int:
        """
        Get a word's id, normalizing it first.

        Raises:
            OutOfVocabularyError: If the word isn't in the vocabulary, with the likely reason why.
        """
        normalized = self.normalize(word)
        word_id = self._ids.get(normalized)
        if word_id is None:
            raise OutOfVocabularyError(word, normalized, self._missing_reason(normalized))
        return word_id

    def word(self, word_id: int) -> str:
        """Get the word with a given id."""
        return self._words[word_id]

    def frequency(self, word: str) -> WordFrequency:
        """How common a word was in the training corpus. See WordFrequency for what each field means."""
        word_id = self.id(word)
        count = int(self._counts[word_id])
        return WordFrequency(rank=word_id, count=count, share=count / self._total_count)

    @property
    def counts(self) -> np.ndarray:
        """Raw frequency count of every word, indexed by id. Read-only view."""
        view = self._counts.view()
        view.flags.writeable = False
        return view

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and self.normalize(word) in self._ids

    def __len__(self) -> int:
        return len(self._words)

    def __iter__(self) -> Iterator[str]:
        """Iterate over words in id order (most common first)."""
        return iter(self._words)

    @staticmethod
    def _missing_reason(normalized: str) -> str:
        """Best-guess explanation of why a normalized word isn't in the vocabulary."""
        tokens = _TOKEN_REGEX.findall(normalized)
        if not tokens:
            return "it contains no word characters, so training's tokenizer would never produce it."
        if len(tokens) > 1 or tokens[0] != normalized:
            return (f"training's tokenizer splits it into {tokens}, so it can never be a single vocabulary entry. "
                    f"Look up the parts separately.")
        # The stopword list and MIN_FREQ cutoff used in training aren't saved with the model, so we can't tell which
        return "it was either a stopword or appeared fewer than MIN_FREQ times in the training corpus."
