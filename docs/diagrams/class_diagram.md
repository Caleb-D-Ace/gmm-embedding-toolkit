# Class diagram

How the pieces of `gmm_toolkit` relate. `GMMModel` is the only class most callers touch. `kernels` is a module of plain functions rather than a class, shown here with a `module` tag. Private helpers are left out.

```mermaid
classDiagram
    direction LR

    class GMMModel {
        <<public entry point>>
        +Vocab vocab
        +int K
        +int D
        -ndarray _mu
        -ndarray _var
        -ndarray _w
        -ndarray _unit
        +load(directory, vocab_file, embeddings_file)$ GMMModel
        +params(word) tuple
        +frequency(word) WordFrequency
        +dominant_sense_weight(word) float
        +sense_matrix(word_a, word_b, metric) ndarray
        +similarity(word_a, word_b, metric, sense_a, sense_b) Score
        +rank(word, candidates, metric, sense_a, sense_b, n) list~Match~
        +most_similar(word, n, metric, sense_a, sense_b) list~Match~
        +hypernym_score(child, parent, child_sense, parent_sense) Score
        +hypernyms(word, n, candidates, sense) list~Match~
        +hyponyms(word, n, candidates, sense) list~Match~
    }

    class Vocab {
        <<word strings live here>>
        -list _words
        -dict _ids
        -ndarray _counts
        +ndarray counts
        +load(path)$ Vocab
        +normalize(word)$ str
        +id(word) int
        +word(word_id) str
        +frequency(word) WordFrequency
    }

    class kernels {
        <<module: pure array math>>
        +unit_rows(mu) ndarray
        +cosine_grid(q_unit, c_unit) ndarray
        +euclidean_grid(q_mu, c_mu) ndarray
        +log_overlap_grid(q_mu, q_var, c_mu, c_var) ndarray
        +kl_grid(q_mu, q_var, c_mu, c_var, reverse) ndarray
        +contribution_grid(log_overlap, w_q, w_c) ndarray
        +best_pair(grid, higher_is_better, sense_a, sense_b) Reduced
        +energy(contribution, sense_a, sense_b) Reduced
    }

    class Metric {
        <<NamedTuple>>
        +bool higher_is_better
        +str description
    }

    class METRICS {
        <<module constant: dict of str to Metric>>
        cosine
        euclidean
        overlap
        energy
        kl
    }

    class Reduced {
        <<NamedTuple: arrays, one entry per candidate>>
        +ndarray values
        +ndarray sense_a
        +ndarray sense_b
    }

    class Score {
        <<NamedTuple>>
        +float value
        +tuple senses
    }

    class Match {
        <<NamedTuple>>
        +str word
        +float value
        +tuple senses
    }

    class WordFrequency {
        <<NamedTuple>>
        +int rank
        +int count
        +float share
    }

    class OutOfVocabularyError {
        +str word
        +str normalized
        +str reason
    }

    class KeyError

    GMMModel *-- "1" Vocab : owns
    GMMModel ..> kernels : calls
    GMMModel ..> METRICS : reads direction from
    METRICS o-- Metric : holds
    GMMModel ..> Score : returns
    GMMModel ..> Match : returns
    GMMModel ..> Reduced : converts into Score and Match
    kernels ..> Reduced : returns
    Vocab ..> WordFrequency : returns
    Vocab ..> OutOfVocabularyError : raises
    KeyError <|-- OutOfVocabularyError
```

## Reading it

- **Solid diamond (`GMMModel` to `Vocab`):** the model holds a vocabulary for its whole lifetime.
- **Dashed arrow:** uses, returns or raises, without keeping a reference.
- **`$` after a method:** a class or static method, called on the class itself, as in `GMMModel.load(...)`.
- **`+` / `-`:** public / private. In the code, private names start with an underscore.

The layering is one-way: `GMMModel` knows about `Vocab` and `kernels`, but neither of those knows about the other or about `GMMModel`. That's what would let a phase 2 torch backend replace `kernels` alone.
