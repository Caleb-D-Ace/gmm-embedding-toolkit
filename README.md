# gmm-embedding-toolkit

A query/analysis library for models trained by [gmm-word-embeddings](https://github.com/Caleb-D-Ace/gmm-word-embeddings) — a PyTorch implementation of Gaussian Mixture Word Embeddings. That repo trains a model and saves it (`sorted_vocab.json` + `gmm_embeddings.npz`/`.pt`); this one reads those files and answers questions about the words in it.

**Status: design/planning stage. No code yet.** This README is the working plan, not documentation of something that exists.

This library has been designed to be useful across a large variety of general applications. My initial purpose in creating the gmm-word-embeddings repository (and this one, by extension) was
to create an alternative word similarity ranking method for word semantic game bots (e.g. Codenames), though it should be useful for many other things.


## Why a separate repo

The training repo's job is to produce a model; this one's job is to let something *use* a trained model without needing PyTorch, CUDA, or anything else training-specific. Splitting them keeps each repo's dependency footprint matched to what it actually needs.


## Architecture decisions

- **NumPy only, no required PyTorch dependency.** The core math (`log_overlap`/`gmm_energy`-equivalent) is a direct port of the formulas already documented in the training repo's README ("Energy from `.npz`, or another language"), written once in NumPy rather than torch. This matches why the training repo saves a `.npz` in the first place — so analysis doesn't require a GPU-capable install.
- **A PyTorch-accelerated backend is a deliberate phase 2, not a parallel effort.** The NumPy version is built and tested first, since it validates the actual hard part (the API shape — sense selection, OOV handling, what's a library primitive vs. bot logic) without also having to solve "which backend" at the same time. Torch acceleration matters most for *bot evaluation throughput* (many simulated games scoring the whole vocabulary repeatedly), not live single-turn latency, so it can wait until there's a working bot to profile. When it's added: `torch` stays an optional, lazily-imported dependency, same pattern the training repo already uses for `datasets`/`pyarrow` — the base install stays NumPy-only.
- **Library primitives vs. bot logic.** This library provides general scoring/lookup operations. Game-specific decision logic — e.g. picking a clue by maximizing the margin between "related to my team's words" and "unrelated to everyone else's" — is NOT in scope here; that belongs in whatever bot is built on top of this library, using its primitives.


## Planned features

### Similarity
Score two words via the training repo's three documented methods: max cosine similarity, min Euclidean distance, or expected-likelihood energy. Optional per-sense selection (0-indexed, matching the underlying `mu`/`var` array indexing — not 1-indexed) on either or both words:
- Both senses given: one unambiguous pairwise score.
- One sense given: max/min of that fixed sense against *all* of the other word's senses.
- Neither given: max/min across every component pair (the default, and what the training repo's own `max_cosine` example already does).

### Hypernym / entailment score
Minimum-component-pair KL divergence between two words' Gaussian components (the training repo's own stretch goal, documented in its `docs/ROADMAP.md`). Asymmetric and directional: `hypernym_score(child, parent)` returns `KL(child‖parent)` — lower means more likely a true hypernym relationship in that direction. Returns which component pair produced the minimum, not just the score, since that identifies *which sense* of each word the relationship is actually about. **Caveat to keep visible in the actual docs once built:** validated only on a small toy synthetic example in the training repo, not a real hypernym dataset — treat as a hypothesis, not a validated feature.

### Multisense / dominant-sense weight
Not a new formula — this is the exact computation already behind the training repo's `mix_weight_spread` metric (`mix_weights.max()` per word), just applied to one word at query time instead of averaged over a training batch. Returned directly as "how dominant is this word's strongest sense" (close to `1.0` = effectively single-sense, close to `1/K` = genuinely multi-sense), rather than also exposing an inverted "multisense score" that's just `1 -` the same number.

### Commonality
Read directly from `sorted_vocab.json`, which already stores a raw frequency count per word (`word -> [id, count]`) — not inferred from position in the file. Exposes both the rank (free, already in the file) and the raw count, since rank alone only tells you *which* word is more common, not *by how much*. A normalized frequency (count ÷ total corpus tokens) is a cheap derived value on top, computed client-side from the counts already present — no new file needed.

### Batch / candidate-list scoring
One underlying primitive — score one word against a list of candidates, sorted by score — serves two distinct use cases:
- **Nearest neighbors**: candidates default to the entire vocabulary.
- **Ranking a specific set**: candidates are a caller-supplied list (e.g. scoring a clue against a fixed board of 25 words). `most_similar(word, n=10)` is a thin convenience wrapper over the same function with `candidates` defaulted to the whole vocabulary, not a separate implementation.


## Practical requirements (designed for, not yet built)

- **Out-of-vocabulary handling.** The training repo's `MIN_FREQ` filtering can drop real words from the trained vocabulary entirely. Looking up a missing word needs to be a clear, documented case (a specific exception or sentinel return) — not a raw `KeyError` leaking out of a dict lookup.
- **Case normalization.** The training pipeline lowercases everything during tokenization; this library should lowercase lookups automatically rather than expecting every caller to remember to do it first.
- **Performance via precomputation, not cleverness.** A normalized `mu` matrix built once when the model is loaded, so "score against the whole vocabulary" is a single matrix operation rather than re-normalizing on every call.


## Data format

Reads the same two files the training repo writes: `sorted_vocab.json` (`word -> [id, frequency count]`) and `gmm_embeddings.npz` (`mu`, `var`, `mix_weights` arrays, each indexed by word id). See [that repo's README](https://github.com/Caleb-D-Ace/gmm_word_embeddings#how-to-use-the-resulting-word-embedding) for the exact shapes and the underlying energy formula.
