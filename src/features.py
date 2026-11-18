import re

import numpy as np
import pandas as pd

from .text import split_sentences

# Long enough that every passage reaches it, so the ratio isn't a proxy for length.
VOCAB_WINDOW = 80
_WORD = re.compile(r"[A-Za-z']+")
_CONTRACTION = re.compile(r"\w(n't|'re|'ve|'ll|'d|'m)\b", re.IGNORECASE)
_FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}


def _per_100_words(count: int, n_words: int) -> float:
    return 100 * count / n_words


def passage_features(text: str) -> dict[str, float]:
    words = [w.lower() for w in _WORD.findall(text)]
    sentence_lengths = [
        len(sentence.split())
        for paragraph in text.split("\n\n")
        for sentence in split_sentences(paragraph)
    ]
    n = len(words)
    return {
        "n_words": n,
        "sentence_length_mean": float(np.mean(sentence_lengths)),
        "sentence_length_sd": float(np.std(sentence_lengths)),
        "word_length_mean": float(np.mean([len(w) for w in words])),
        "distinct_words_in_first_80": len(set(words[:VOCAB_WINDOW])) / min(n, VOCAB_WINDOW),
        "contractions_per_100": _per_100_words(len(_CONTRACTION.findall(text)), n),
        "first_person_per_100": _per_100_words(sum(w in _FIRST_PERSON for w in words), n),
        "commas_per_100": _per_100_words(text.count(","), n),
        "quote_marks_per_100": _per_100_words(text.count('"'), n),
        "digits_per_100": _per_100_words(len(re.findall(r"\d+", text)), n),
    }


def feature_table(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([passage_features(t) for t in texts], index=texts.index)
