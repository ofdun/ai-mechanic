"""Lexical retrieval: tokenization of Russian technical text and in-memory Okapi BM25.

The corpus (~1.7K pages) fits in memory and the index is rebuilt from chunks in well under a second.
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Sequence
from functools import lru_cache
from threading import local
from typing import Literal

import snowballstemmer

Analyzer = Literal['stem', 'plain']

# Numbers keep their decimal part: «0,25» and «0.25» map to the same token.
TOKEN = re.compile(r'\d+(?:[.,]\d+)*|[^\W\d_]+')
STOPWORDS = frozenset('''
    а без в во да для до же за и из или как к ко ли на над не ни но о об обо от по под при про с со так то у что
    это его ее их им ими они она оно он мы вы бы был была были было быть есть
    the a an and or of to in on for with
'''.split())


class _Analyzers(local):
    def __init__(self):
        self.russian = snowballstemmer.stemmer('russian')


# Snowball mutates its current word during stemming. Each HTTP worker thread needs its own instance.
_analyzers = _Analyzers()


def _plain(word: str) -> str:
    return word


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _analyzers.russian.stemWord(word)


def tokenizer(analyzer: Analyzer) -> Callable[[str], list[str]]:
    normalize_word = _stem if analyzer == 'stem' else _plain

    def tokenize(text: str) -> list[str]:
        text = unicodedata.normalize('NFC', text).casefold().replace('ё', 'е')
        tokens = []
        for token in TOKEN.findall(text):
            if token[0].isdigit():
                tokens.append(token.replace(',', '.'))
            elif len(token) > 1 and token not in STOPWORDS:
                tokens.append(normalize_word(token))
        return tokens

    return tokenize


class BM25:
    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.lengths = [len(tokens) for tokens in documents]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, tokens in enumerate(documents):
            for term, frequency in Counter(tokens).items():
                self.postings[term].append((index, frequency))
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def scores(self, query: Sequence[str], allowed: Collection[int] | None = None) -> dict[int, float]:
        """Scores of documents that contain at least one query term. Repeated query terms count once."""
        result: dict[int, float] = defaultdict(float)
        for term in dict.fromkeys(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequency in self.postings[term]:
                if allowed is not None and index not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                result[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return result

    def top(self, query: Sequence[str], limit: int, allowed: Collection[int] | None = None) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query, allowed).items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]
