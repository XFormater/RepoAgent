"""Dependency-free lexical features shared by local memory adapters."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


def lexical_tokens(text: str) -> list[str]:
    """Normalize English inflections while retaining non-English word tokens."""
    output: list[str] = []
    for raw in _TOKEN_RE.findall(text.casefold()):
        if raw in _STOP_WORDS:
            continue
        token = _simple_stem(raw) if raw.isascii() else raw
        if len(token) > 1 or token.isdigit():
            output.append(token)
    return output


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    """Return BM25 scores in document order without third-party dependencies."""
    if not documents:
        return []
    query_terms = set(lexical_tokens(query))
    if not query_terms:
        return [0.0] * len(documents)
    tokenized = [lexical_tokens(document) for document in documents]
    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized) for term in query_terms
    }
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
    scores: list[float] = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length_normalizer = 1.2 * (1 - 0.75 + 0.75 * len(tokens) / average_length)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            frequency_docs = document_frequency[term]
            inverse_frequency = math.log(
                1 + (len(documents) - frequency_docs + 0.5) / (frequency_docs + 0.5)
            )
            score += inverse_frequency * frequency * 2.2 / (frequency + length_normalizer)
        scores.append(score)
    return scores


def _simple_stem(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for suffix in ("ingly", "edly", "ing", "ed"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            stem = token[: -len(suffix)]
            if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
                stem = stem[:-1]
            return stem
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token
