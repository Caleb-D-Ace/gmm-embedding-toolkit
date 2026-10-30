"""
model ties the vocabulary (word strings) to the kernels (array math) and is the public entry point of the
library: GMMModel.load() a trained model, then ask it questions about words.
"""
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Union

import numpy as np

from . import kernels
from .results import Match, Score, WordFrequency
from .vocab import Vocab


class Metric(NamedTuple):
    """How a metric's scores are read."""
    higher_is_better: bool
    description: str


# Every metric works with sense_matrix(), similarity(), rank() and most_similar().
METRICS: dict[str, Metric] = {
    "cosine": Metric(True, "Cosine similarity between component means. Ignores var and mix_weights."),
    "euclidean": Metric(False, "Euclidean distance between component means. Ignores var and mix_weights."),
    "overlap": Metric(True, "Log expected likelihood kernel between two components, ignoring how common either "
                            "sense is. Uses mu and var."),
    "energy": Metric(True, "Expected likelihood energy between the whole mixtures, the score training optimized. "
                           "Its matrix is each sense pair's weighted contribution, and its value is a soft sum over "
                           "them rather than the best one. Uses mu, var and mix_weights."),
    "kl": Metric(False, "KL(first word's component ‖ second word's component). Asymmetric; low means the first is "
                        "contained by the second. See hypernym_score() for the caveats."),
}

# Not public: the same as "kl" but with the query as the parent, used by hyponyms()
_KL_REVERSE = "kl_reverse"


class GMMModel:
    """
    A trained Gaussian Mixture Word Embedding: each word is K Gaussian components ("senses") in a D-dimensional space.

    Words are looked up case-insensitively; a word missing from the vocabulary raises OutOfVocabularyError.
    Senses are 0-indexed, matching the rows of the underlying arrays.
    """

    def __init__(self, vocab: Vocab, mu: np.ndarray, var: np.ndarray, mix_weights: np.ndarray):
        """
        Args:
            vocab: The word <-> id mapping the arrays were trained with.
            mu, var: Component means and diagonal variances, shape (vocab_size, K, D).
            mix_weights: Component weights, shape (vocab_size, K), each row summing to 1.

        Raises:
            ValueError: If the arrays' shapes don't match each other or the vocabulary.
        """
        if mu.ndim != 3:
            raise ValueError(f"mu must have shape (vocab_size, K, D), got {mu.shape}.")
        V, K, _D = mu.shape
        if var.shape != mu.shape:
            raise ValueError(f"var has shape {var.shape}, but mu has shape {mu.shape}; they must match.")
        if mix_weights.shape != (V, K):
            raise ValueError(f"mix_weights has shape {mix_weights.shape}, expected {(V, K)} to match mu.")
        if len(vocab) != V:
            raise ValueError(f"The vocabulary has {len(vocab)} words but the arrays have {V} rows. "
                             f"Were they saved by different training runs?")

        self.vocab = vocab
        # Read-only views: everything handed out (e.g. by params()) is a view of these, so callers can't corrupt the
        # model, and making views rather than flagging the arrays themselves leaves the caller's own arrays writeable
        self._mu, self._var, self._w = mu.view(), var.view(), mix_weights.view()
        # Precomputed once so cosine against the whole vocabulary is a single matmul
        self._unit = kernels.unit_rows(mu)
        for array in (self._mu, self._var, self._w, self._unit):
            array.flags.writeable = False

    @classmethod
    def load(cls, directory: Union[str, Path], vocab_file: str = "sorted_vocab.json",
             embeddings_file: str = "gmm_embeddings.npz") -> "GMMModel":
        """
        Load a model from the directory training wrote it to (data/model/ by default in the training repo).

        Raises:
            ValueError: If the .npz is missing an array or the files don't match each other.
        """
        directory = Path(directory)
        vocab = Vocab.load(directory / vocab_file)
        with np.load(directory / embeddings_file) as data:
            missing = {"mu", "var", "mix_weights"} - set(data.files)
            if missing:
                raise ValueError(f"{directory / embeddings_file} is missing the arrays {sorted(missing)}.")
            return cls(vocab, data["mu"], data["var"], data["mix_weights"])

    @property
    def K(self) -> int:
        """Number of components (senses) per word."""
        return self._mu.shape[1]

    @property
    def D(self) -> int:
        """Dimensionality of each component."""
        return self._mu.shape[2]

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, word: object) -> bool:
        return word in self.vocab

    # --- Single-word lookups ---

    def params(self, word: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A word's raw (mu (K, D), var (K, D), mix_weights (K,)). Read-only views."""
        word_id = self.vocab.id(word)
        return self._mu[word_id], self._var[word_id], self._w[word_id]

    def frequency(self, word: str) -> WordFrequency:
        """How common a word was in the training corpus. See WordFrequency."""
        return self.vocab.frequency(word)

    def dominant_sense_weight(self, word: str) -> float:
        """
        The mixture weight of a word's strongest sense: close to 1.0 means effectively single-sense, close to 1/K
        means the model genuinely splits it across senses. The per-word version of training's mix_weight_spread.
        """
        return float(self._w[self.vocab.id(word)].max())

    # --- Comparing words ---

    def sense_matrix(self, word_a: str, word_b: str, metric: str = "cosine") -> np.ndarray:
        """
        The full (K, K) matrix of a metric between every sense of word_a (rows) and every sense of word_b (columns).
        For "energy" this is each pair's weighted contribution, which logsumexp's to the energy.
        """
        return self._grid(self.vocab.id(word_a), np.array([self.vocab.id(word_b)]), metric)[0]

    def similarity(self, word_a: str, word_b: str, metric: str = "cosine",
                   sense_a: Optional[int] = None, sense_b: Optional[int] = None) -> Score:
        """
        Score two words with one metric (see METRICS). Optionally fix the sense of either or both:
            both given -> that single pair's score
            one given  -> that sense against all of the other word's senses
            neither    -> across every sense pair
        For best-pair metrics "across" means the best pair; for "energy" it's the soft sum, restricted to the
        allowed pairs (which then gives those senses' share of the full energy).
        """
        reduced = self._reduce(self.vocab.id(word_a), np.array([self.vocab.id(word_b)]), metric, sense_a, sense_b)
        return Score(float(reduced.values[0]), (int(reduced.sense_a[0]), int(reduced.sense_b[0])))

    def rank(self, word: str, candidates: Optional[Iterable[str]] = None, metric: str = "cosine",
             sense_a: Optional[int] = None, sense_b: Optional[int] = None, n: Optional[int] = None) -> list[Match]:
        """
        Score a word against a list of candidates and sort them best-first (in the metric's own direction).

        Args:
            word: The query word.
            candidates: Words to score against. None means the whole vocabulary (including the word itself; see
                        most_similar() to leave it out). Every candidate must be in the vocabulary; check with
                        `word in model` first if that isn't guaranteed.
            metric, sense_a, sense_b: As in similarity(). sense_b applies to every candidate.
            n: Return only the top n. None returns them all.

        Ties keep candidate order (or, for the whole vocabulary, the more common word first).
        """
        query_id = self.vocab.id(word)
        candidate_ids = None if candidates is None else np.array([self.vocab.id(c) for c in candidates], dtype=np.int64)
        return self._ranked(query_id, candidate_ids, metric, sense_a, sense_b, n, exclude_id=None)

    def most_similar(self, word: str, n: int = 10, metric: str = "cosine",
                     sense_a: Optional[int] = None, sense_b: Optional[int] = None) -> list[Match]:
        """The n nearest neighbors of a word across the whole vocabulary, leaving out the word itself."""
        return self._ranked(self.vocab.id(word), None, metric, sense_a, sense_b, n, exclude_id=self.vocab.id(word))

    # --- Hypernyms ---

    def hypernym_score(self, child: str, parent: str,
                       child_sense: Optional[int] = None, parent_sense: Optional[int] = None) -> Score:
        """
        Minimum component-pair KL(child ‖ parent): how well some sense of parent "contains" some sense of child.
        Lower means more likely a true hypernym in that direction. senses = (child sense, parent sense) it's about.

        Caveat: this is the paper's entailment score (§4.7), but it has only been checked on a small synthetic
        example in the training repo, not a real hypernym dataset. Treat it as a hypothesis, not a validated feature.
        Broad, high-variance words tend to "contain" everything, so compare directions (hypernym_score(a, b) vs
        hypernym_score(b, a)) rather than reading one score on its own.
        """
        return self.similarity(child, parent, "kl", child_sense, parent_sense)

    def hypernyms(self, word: str, n: int = 10, candidates: Optional[Iterable[str]] = None,
                  sense: Optional[int] = None) -> list[Match]:
        """Candidates most likely to be parents of word: lowest KL(word ‖ candidate). Same caveats as hypernym_score()."""
        return self._kl_ranked(word, n, candidates, sense, "kl")

    def hyponyms(self, word: str, n: int = 10, candidates: Optional[Iterable[str]] = None,
                 sense: Optional[int] = None) -> list[Match]:
        """Candidates most likely to be children of word: lowest KL(candidate ‖ word). Same caveats as hypernym_score()."""
        return self._kl_ranked(word, n, candidates, sense, _KL_REVERSE)

    # --- Internals ---

    def _kl_ranked(self, word, n, candidates, sense, metric) -> list[Match]:
        # Every word trivially "contains" itself (KL = 0), so it's always left out
        query_id = self.vocab.id(word)
        candidate_ids = None if candidates is None else np.array([self.vocab.id(c) for c in candidates], dtype=np.int64)
        return self._ranked(query_id, candidate_ids, metric, sense, None, n, exclude_id=query_id)

    def _ranked(self, query_id: int, candidate_ids: Optional[np.ndarray], metric: str,
                sense_a: Optional[int], sense_b: Optional[int], n: Optional[int], exclude_id: Optional[int]) -> list[Match]:
        reduced = self._reduce(query_id, candidate_ids, metric, sense_a, sense_b)
        ids = np.arange(len(self.vocab)) if candidate_ids is None else candidate_ids

        higher = self._higher_is_better(metric)
        order = np.argsort(-reduced.values if higher else reduced.values, kind="stable")
        if exclude_id is not None:
            order = order[ids[order] != exclude_id]
        if n is not None:
            order = order[:n]
        return [Match(self.vocab.word(int(ids[i])), float(reduced.values[i]),
                      (int(reduced.sense_a[i]), int(reduced.sense_b[i]))) for i in order]

    def _reduce(self, query_id: int, candidate_ids: Optional[np.ndarray], metric: str,
                sense_a: Optional[int], sense_b: Optional[int]) -> kernels.Reduced:
        """Score a query against candidates (None = whole vocabulary), one Reduced entry per candidate."""
        grid = self._grid(query_id, candidate_ids, metric)
        if metric == "energy":
            return kernels.energy(grid, sense_a, sense_b)
        return kernels.best_pair(grid, self._higher_is_better(metric), sense_a, sense_b)

    def _grid(self, query_id: int, candidate_ids: Optional[np.ndarray], metric: str) -> np.ndarray:
        """The (M, K, K) matrix of a metric between a query and candidates (None = whole vocabulary)."""
        c = slice(None) if candidate_ids is None else candidate_ids
        q = query_id
        if metric == "cosine":
            return kernels.cosine_grid(self._unit[q], self._unit[c])
        if metric == "euclidean":
            return kernels.euclidean_grid(self._mu[q], self._mu[c])
        if metric == "overlap":
            return kernels.log_overlap_grid(self._mu[q], self._var[q], self._mu[c], self._var[c])
        if metric == "energy":
            log_overlap = kernels.log_overlap_grid(self._mu[q], self._var[q], self._mu[c], self._var[c])
            return kernels.contribution_grid(log_overlap, self._w[q], self._w[c])
        if metric in ("kl", _KL_REVERSE):
            return kernels.kl_grid(self._mu[q], self._var[q], self._mu[c], self._var[c], reverse=metric == _KL_REVERSE)
        raise ValueError(f"Unknown metric {metric!r}. Choose one of {sorted(METRICS)}.")

    @staticmethod
    def _higher_is_better(metric: str) -> bool:
        return METRICS["kl" if metric == _KL_REVERSE else metric].higher_is_better
