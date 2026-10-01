from __future__ import annotations

import unicodedata
from collections.abc import Iterable


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}


def invalid_character_ratio(text: str) -> float:
    if not text:
        return 1.0
    invalid = 0
    for char in text:
        if char in "\n\r\t":
            continue
        category = unicodedata.category(char)
        if char == "\ufffd" or category in {"Cs", "Co", "Cn"} or category == "Cc":
            invalid += 1
    return invalid / len(text)


def extract_content_tokens(docs: Iterable[object]) -> list[list[str]]:
    result: list[list[str]] = []
    for doc in docs:
        tokens: list[str] = []
        for token in doc:  # type: ignore[union-attr]
            if token.pos_ not in CONTENT_POS or token.is_space or token.is_punct:
                continue
            value = (token.lemma_ or token.text).strip().lower()
            if value and any(char.isalnum() for char in value):
                tokens.append(value)
        result.append(tokens)
    return result
