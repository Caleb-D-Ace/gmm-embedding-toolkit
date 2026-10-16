"""
Exceptions raised by gmm_toolkit.
"""


class OutOfVocabularyError(KeyError):
    """
    Raised when a word isn't in the trained vocabulary.

    Subclasses KeyError so existing `except KeyError` code still catches it, but carries enough detail
    to tell the caller *why* the word is missing instead of just echoing it back.

    Attributes:
        word: The word exactly as the caller passed it.
        normalized: The word after lowercasing/stripping, i.e. what was actually looked up.
        reason: Human-readable explanation of why it can't be in the vocabulary.
    """

    def __init__(self, word: str, normalized: str, reason: str):
        self.word = word
        self.normalized = normalized
        self.reason = reason
        super().__init__(word)

    def __str__(self) -> str:
        # KeyError.__str__ would just repr() the key, so override it to include the reason
        return f"{self.word!r} is not in the vocabulary: {self.reason}"
