# -*- coding: utf-8 -*-
"""Word-sentence association construction for HyperLSS.

The paper retains nouns, verbs, and adjectives as document-level word nodes.
For every retained word occurring in a sentence, two directed association
edges are created: word -> sentence and sentence -> word. Both directions use
the same TF-IDF value, normalized so that the retained word weights connected
to each sentence sum to one. IDF statistics must be fitted on training data
only and then reused unchanged for validation and test data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}


@dataclass(frozen=True)
class WordSentenceRelations:
    """Document-local word nodes and their weighted directed edges."""

    words: list[str]
    word_to_node: dict[str, int]
    edges: list[tuple[tuple[int, int], float]]


def load_pos_tagger(model_name: str = "en_core_web_sm"):
    """Load the POS tagger once so it can be reused across documents."""
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is required for content-word extraction") from exc
    return spacy.load(model_name, disable=["ner", "parser"])


def extract_content_words(
    sentences: Sequence[str],
    nlp=None,
    *,
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 256,
) -> list[list[str]]:
    """Lemmatize and retain nouns, proper nouns, verbs, and adjectives."""
    if nlp is None:
        nlp = load_pos_tagger(spacy_model)

    sentence_list = [str(sentence).strip() for sentence in sentences]
    if any(not sentence for sentence in sentence_list):
        raise ValueError("sentences must not contain empty text")

    content_sentences: list[list[str]] = []
    for document in nlp.pipe(sentence_list, batch_size=batch_size):
        content_words: list[str] = []
        for token in document:
            if token.pos_ not in CONTENT_POS or token.is_space or token.is_punct:
                continue
            word = (token.lemma_ or token.text).strip().lower()
            if word and any(character.isalnum() for character in word):
                content_words.append(word)
        content_sentences.append(content_words)
    return content_sentences


def fit_training_tfidf(
    training_content_sentences: Iterable[Sequence[str]],
) -> "TfidfVectorizer":
    """Fit TF-IDF once using content-word sentences from the training split.

    `norm=None` is intentional: HyperLSS applies its own per-sentence sum
    normalization when constructing word-sentence edge weights.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to fit TF-IDF") from exc

    corpus = [" ".join(words) for words in training_content_sentences if words]
    if not corpus:
        raise ValueError("training_content_sentences contains no content words")

    vectorizer = TfidfVectorizer(
        lowercase=False,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        norm=None,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    vectorizer.fit(corpus)
    return vectorizer


def _document_words(
    content_sentences: Sequence[Sequence[str]],
    training_vocabulary: dict[str, int],
) -> list[str]:
    """Return distinct in-vocabulary words in first-occurrence order."""
    seen: set[str] = set()
    words: list[str] = []
    for sentence in content_sentences:
        for word in sentence:
            if word in training_vocabulary and word not in seen:
                seen.add(word)
                words.append(word)
    return words


def word_sentence_edges(
    content_sentences: Sequence[Sequence[str]],
    tfidf_vectorizer: Any,
) -> WordSentenceRelations:
    """Construct bidirectional, sentence-normalized TF-IDF relations.

    Sentence node IDs are ``0 .. n-1``. Word node IDs start at ``n`` so the
    two node types never collide. Words unseen in the training TF-IDF
    vocabulary are omitted because their training-corpus IDF is undefined.
    """
    if not hasattr(tfidf_vectorizer, "vocabulary_"):
        raise ValueError("tfidf_vectorizer must already be fitted on training data")

    sentence_count = len(content_sentences)
    vocabulary: dict[str, int] = tfidf_vectorizer.vocabulary_
    words = _document_words(content_sentences, vocabulary)
    word_to_node = {
        word: sentence_count + local_index for local_index, word in enumerate(words)
    }

    edges: list[tuple[tuple[int, int], float]] = []
    for sentence_node, tokens in enumerate(content_sentences):
        retained = [word for word in dict.fromkeys(tokens) if word in word_to_node]
        if not retained:
            continue

        tfidf_row = tfidf_vectorizer.transform([" ".join(tokens)])
        feature_values = {
            int(feature_index): float(value)
            for feature_index, value in zip(tfidf_row.indices, tfidf_row.data)
        }
        raw_weights = {
            word: feature_values.get(vocabulary[word], 0.0) for word in retained
        }
        denominator = sum(raw_weights.values())
        if denominator <= 0.0:
            continue

        for word in retained:
            word_node = word_to_node[word]
            weight = raw_weights[word] / denominator
            edges.append(((word_node, sentence_node), weight))
            edges.append(((sentence_node, word_node), weight))

    return WordSentenceRelations(
        words=words,
        word_to_node=word_to_node,
        edges=edges,
    )


def build_word_sentence_edges(
    sentences: Sequence[str],
    tfidf_vectorizer: Any,
    nlp=None,
    *,
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 256,
) -> tuple[list[list[str]], WordSentenceRelations]:
    """Extract content words and construct one document's relations."""
    content_sentences = extract_content_words(
        sentences,
        nlp=nlp,
        spacy_model=spacy_model,
        batch_size=batch_size,
    )
    relations = word_sentence_edges(content_sentences, tfidf_vectorizer)
    return content_sentences, relations


def sentence_weight_sums(
    relations: WordSentenceRelations,
    sentence_count: int,
) -> np.ndarray:
    """Return incoming word-edge weight sums for validation/debugging."""
    sums = np.zeros(sentence_count, dtype=np.float32)
    word_nodes = set(relations.word_to_node.values())
    for (source, target), weight in relations.edges:
        if source in word_nodes and 0 <= target < sentence_count:
            sums[target] += weight
    return sums

