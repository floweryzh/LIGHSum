from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield line_number, value


def write_jsonl(stream: TextIO, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _as_sentence_list(value: Any, split_sentences: Callable[[str], list[str]]) -> list[str]:
    def clean(text: str) -> str:
        text = html.unescape(text)
        text = re.sub(r"</?S>", "", text, flags=re.IGNORECASE)
        return " ".join(text.split()).strip()

    if value is None:
        return []
    if isinstance(value, str):
        return [clean(text) for text in split_sentences(value) if clean(text)]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = clean(item)
                if text:
                    result.append(text)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("sentence")
                if isinstance(text, str) and clean(text):
                    result.append(clean(text))
        return result
    return []


def normalize_raw_record(
    raw: dict[str, Any],
    fallback_id: str,
    split_sentences: Callable[[str], list[str]],
) -> tuple[str, list[str], list[int], list[str], list[str]]:
    """Normalize common PubMed/arXiv JSONL layouts without changing split membership."""
    document_id = str(raw.get("article_id") or raw.get("id") or raw.get("document_id") or fallback_id)
    reference = _as_sentence_list(
        raw.get("abstract_text", raw.get("abstract", raw.get("summary"))), split_sentences
    )

    structured = raw.get("sections")
    if isinstance(structured, list) and structured and all(isinstance(x, dict) for x in structured):
        sentences: list[str] = []
        section_ids: list[int] = []
        names: list[str] = []
        for section_index, section in enumerate(structured):
            name = str(section.get("heading") or section.get("title") or f"section-{section_index}")
            section_sentences = _as_sentence_list(
                section.get("sentences", section.get("text")), split_sentences
            )
            if not section_sentences:
                continue
            names.append(name)
            normalized_index = len(names) - 1
            sentences.extend(section_sentences)
            section_ids.extend([normalized_index] * len(section_sentences))
        if sentences:
            return document_id, sentences, section_ids, names, reference

    if isinstance(structured, list) and structured:
        raw_names = raw.get("section_names")
        sentences = []
        section_ids = []
        names = []
        for section_index, section in enumerate(structured):
            section_sentences = _as_sentence_list(section, split_sentences)
            if not section_sentences:
                continue
            if isinstance(raw_names, list) and section_index < len(raw_names):
                name = str(raw_names[section_index]).strip() or f"section-{section_index}"
            else:
                name = f"section-{section_index}"
            names.append(name)
            normalized_index = len(names) - 1
            sentences.extend(section_sentences)
            section_ids.extend([normalized_index] * len(section_sentences))
        if sentences:
            return document_id, sentences, section_ids, names, reference

    sentences = _as_sentence_list(
        raw.get("article_text", raw.get("sentences", raw.get("document", raw.get("text")))),
        split_sentences,
    )
    raw_names = raw.get("section_names")
    if isinstance(raw_names, list) and len(raw_names) == len(sentences):
        # Preserve the original section *occurrences*.  The same heading can
        # legitimately appear again later in a document; those are separate
        # source sections and must not be merged into one hyperedge.
        names = []
        section_ids = []
        previous_name: str | None = None
        current_id = -1
        for value in raw_names:
            name = str(value).strip() or "untitled"
            if previous_name is None or name != previous_name:
                current_id += 1
                names.append(name)
                previous_name = name
            section_ids.append(current_id)
    else:
        names = ["document"] if sentences else []
        section_ids = [0] * len(sentences)
    return document_id, sentences, section_ids, names, reference
