# gmm-embedding-toolkit

A query/analysis library for models trained by [gmm-word-embeddings](https://github.com/Caleb-D-Ace/gmm-word-embeddings) — a PyTorch implementation of Gaussian Mixture Word Embeddings. That repo trains a model and saves it (`sorted_vocab.json` + `gmm_embeddings.npz`/`.pt`); this one reads those files and answers questions about the words in it.

This library has been designed to be useful across a large variety of general applications. My initial purpose in creating the gmm-word-embeddings repository (and this one, by extension) was
to create an alternative word similarity ranking method for word semantic game bots (e.g. Codenames), though it should be useful for many other things.


## Installation

Requires Python 3.10+. The only runtime dependency is NumPy.

```bash
pip install -e .          # from a clone of this repo
pip install -e '.[dev]'   # also installs pytest
```


## Quick start

Point `GMMModel.load` at the directory training wrote to, which needs `sorted_vocab.json` and `gmm_embeddings.npz`:

```python
from gmm_toolkit import GMMModel, OutOfVocabularyError

model = GMMModel.load("path/to/data/model")

value, (sense_a, sense_b) = model.similarity("bank", "river")   # A score, and the sense pair it came from
model.sense_matrix("bank", "river")                             # The full K×K matrix behind that score
model.most_similar("bank", n=10)                                # Nearest neighbors across the vocabulary
model.rank("river", ["bank", "money", "water"], metric="energy") # Sort a specific list of words
```


## Why a separate repo

The training repo's job is to produce a model; this one's job is to let something *use* a trained model without needing PyTorch, CUDA, or anything else training-specific. Splitting them keeps each repo's dependency footprint matched to what it actually needs.


## Architecture decisions

- **NumPy only, no required PyTorch dependency.** The core math (`log_overlap`/`gmm_energy`-equivalent) is a direct port of the formulas already documented in the training repo's README ("Energy from `.npz`, or another language"), written once in NumPy rather than torch. This matches why the training repo saves a `.npz` in the first place — so analysis doesn't require a GPU-capable install.
- **A PyTorch-accelerated backend is a deliberate phase 2, not a parallel effort.** The NumPy version is built and tested first, since it validates the actual hard part (the API shape — sense selection, OOV handling, what's a library primitive vs. bot logic) without also having to solve "which backend" at the same time. Torch acceleration matters most for *bot evaluation throughput* (many simulated games scoring the whole vocabulary repeatedly), not live single-turn latency, so it can wait until there's a working bot to profile. When it's added: `torch` stays an optional, lazily-imported dependency, same pattern the training repo already uses for `datasets`/`pyarrow` — the base install stays NumPy-only.
- **Library primitives vs. bot logic.** This library provides general scoring/lookup operations. Game-specific decision logic — e.g. picking a clue by maximizing the margin between "related to my team's words" and "unrelated to everyone else's" — is NOT in scope here; that belongs in whatever bot is built on top of this library, using its primitives.


## Metrics

Every metric works with `sense_matrix`, `similarity`, `rank` and `most_similar`. Pick one with `metric=` (default `"cosine"`). The table is also available in code as `gmm_toolkit.METRICS`.

| Metric | Better | Uses | Answers |
|---|---|---|---|
| `"cosine"` | higher | `mu` | How aligned are the two closest senses? |
| `"euclidean"` | lower | `mu` | How far apart are the two closest senses? |
| `"overlap"` | higher | `mu`, `var` | How much do the two best-matching senses overlap, regardless of how common they are? |
| `"energy"` | higher | `mu`, `var`, `mix_weights` | How related are the words as whole distributions? This is the score training optimized. |
| `"kl"` | lower | `mu`, `var` | How well is a sense of the first word contained by a sense of the second? Asymmetric. |

Energy differs from the others. They each score one sense pair, so their result is the best pair in the matrix. Energy is a soft sum over every pair, weighted by how common each sense is, so a strong match on a rare sense counts for less. Its matrix (`sense_matrix(a, b, "energy")`) holds each pair's contribution to the total, and the senses it reports are the pair contributing the most. Log-overlaps and energies are large negative numbers (around -60 at `D=50`); only their differences mean anything.


## Features

### Similarity
`similarity(word_a, word_b, metric, sense_a, sense_b)` returns a `Score(value, senses)`, which unpacks as `value, (a, b) = score`. Senses are 0-indexed, matching the underlying `mu`/`var` array indexing. Either or both can be fixed:
- Both senses given: that one pair's score.
- One sense given: that fixed sense against *all* of the other word's senses.
- Neither given: across every sense pair (the default).

For best-pair metrics, "across" means the max or min. For `"energy"` it means the soft sum over the allowed pairs, which gives those senses' share of the full energy: the per-sense energies combine back into the total. Only compare energies computed with the same senses fixed, since they aren't on the same scale otherwise.

### Sense matrix
`sense_matrix(word_a, word_b, metric)` returns the full `K×K` array that `similarity` reduces, with `word_a`'s senses as rows and `word_b`'s as columns.

### Batch / candidate-list scoring
One underlying primitive serves two use cases, both returning `Match(word, value, senses)` lists sorted best-first in the metric's own direction:
- **Nearest neighbors**: `most_similar(word, n=10)` scores the entire vocabulary and leaves out the word itself.
- **Ranking a specific set**: `rank(word, candidates)` scores a caller-supplied list (e.g. a clue against a fixed board of 25 words). With no candidates it scores the whole vocabulary, including the word itself.

Ties keep candidate order, or for the whole vocabulary, put the more common word first. An out-of-vocabulary candidate raises an error rather than being skipped, so filter with `word in model` first if that isn't guaranteed.

### Hypernym / entailment score
Minimum-component-pair KL divergence between two words' Gaussian components (the training repo's own stretch goal, documented in its `docs/ROADMAP.md`). Asymmetric and directional: `hypernym_score(child, parent)` returns `KL(child‖parent)`, where lower means more likely a true hypernym relationship in that direction. It returns which component pair produced the minimum, since that identifies *which sense* of each word the relationship is about. `hypernyms(word)` and `hyponyms(word)` rank candidates in each direction.

**Caveat:** validated only on a small toy synthetic example in the training repo, not a real hypernym dataset. Treat it as a hypothesis, not a validated feature. Compare the two directions (`hypernym_score(a, b)` against `hypernym_score(b, a)`) rather than reading one score alone. Between two components of equal variance, KL reduces to their squared distance, so in a whole-vocabulary `hypernyms()` scan any nearby word of similar variance can outrank a true parent.

### Multisense / dominant-sense weight
`dominant_sense_weight(word)` is the computation behind the training repo's `mix_weight_spread` metric (`mix_weights.max()` per word), applied to one word at query time. Close to `1.0` means effectively single-sense; close to `1/K` means genuinely multi-sense.

### Commonality
`frequency(word)` returns `WordFrequency(rank, count, share)`, read from the counts in `sorted_vocab.json`. `rank` is 0-indexed (0 is the most common word) and equals the word's id. `share` is the count divided by the sum of all in-vocabulary counts. It is **not** a share of the whole corpus, because stopwords and words below `MIN_FREQ` were dropped before counting and their totals aren't saved.

### Raw parameters
`params(word)` returns the word's `(mu, var, mix_weights)` as read-only arrays. `model.K` and `model.D` give the number of senses and dimensions.


## Behavior worth knowing

- **Out-of-vocabulary words.** Looking up a missing word raises `OutOfVocabularyError`, a `KeyError` subclass with `.word`, `.normalized` and `.reason`. Check first with `word in model`. Multi-word terms ("ice cream") and hyphenated ones can never be in the vocabulary, because training's tokenizer splits them; the error says so. What to do with them is left to the caller.
- **Case normalization.** Lookups are stripped of surrounding whitespace and lowercased, matching what training's tokenizer did.
- **Load-time validation.** `GMMModel.load` raises `ValueError` if an array is missing, the shapes disagree, or the vocabulary and arrays are different sizes (usually files from two different runs).
- **Performance.** Normalized means are computed once at load time, so cosine against the whole vocabulary is a single matrix operation. Metrics that need per-dimension differences run over candidates in chunks to limit memory. Scoring one word against 100,000 words (K=2, D=50, float32) took roughly 0.03s for cosine and 0.25s for energy on synthetic data on the machine this was developed on.


## Project layout

| File | Role |
|---|---|
| `src/gmm_toolkit/model.py` | `GMMModel`, the public entry point |
| `src/gmm_toolkit/vocab.py` | `Vocab`: everything to do with word strings |
| `src/gmm_toolkit/kernels.py` | Pure NumPy math, with no knowledge of words |
| `src/gmm_toolkit/results.py` | `Score`, `Match`, `WordFrequency` |
| `src/gmm_toolkit/errors.py` | `OutOfVocabularyError` |

See the [class diagram](docs/diagrams/class_diagram.md).


## Tests

```bash
pytest
```

`tests/test_reference.py` compares the NumPy math against the training repo's own `GMMWordEmbedding`. It is skipped unless `torch` is installed and the training repo is found at `../gmm_program/gmm_word_embeddings`, or wherever `GMM_TRAINING_REPO` points.


## Data format

Reads the same two files the training repo writes: `sorted_vocab.json` (`word -> [id, frequency count]`) and `gmm_embeddings.npz` (`mu`, `var`, `mix_weights` arrays, each indexed by word id). See [that repo's README](https://github.com/Caleb-D-Ace/gmm_word_embeddings#how-to-use-the-resulting-word-embedding) for the exact shapes and the underlying energy formula.
