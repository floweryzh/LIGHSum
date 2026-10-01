from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PreprocessConfig
from .io import iter_jsonl, normalize_raw_record, write_jsonl
from .oracle import greedy_oracle_labels
from .schema import CleanDocument, Rejection
from .text import extract_content_tokens, invalid_character_ratio


class CorpusCleaner:
    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        try:
            import spacy
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency error is user-facing
            raise RuntimeError("spaCy and transformers are required for preprocessing") from exc

        self.nlp = spacy.load(config.spacy_model, disable=["ner"])
        if "parser" not in self.nlp.pipe_names and "senter" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
        self.tokenizer = AutoTokenizer.from_pretrained(config.sentence_model)

    def split_sentences(self, text: str) -> list[str]:
        return [span.text.strip() for span in self.nlp(text).sents if span.text.strip()]

    def _token_count(self, sentence: str) -> int:
        encoded = self.tokenizer(
            sentence,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])

    def clean_record(self, raw: dict[str, Any], fallback_id: str) -> CleanDocument | Rejection:
        document_id, sentences, section_ids, section_names, reference = normalize_raw_record(
            raw, fallback_id, self.split_sentences
        )
        if not sentences:
            return Rejection(document_id, "empty_source")
        if not reference:
            return Rejection(document_id, "missing_reference")

        combined = "\n".join(sentences + reference)
        ratio = invalid_character_ratio(combined)
        if ratio > self.config.invalid_character_ratio:
            return Rejection(document_id, "invalid_characters", f"ratio={ratio:.6f}")

        token_counts = [self._token_count(sentence) for sentence in sentences]
        for index, count in enumerate(token_counts):
            if count <= self.config.max_sentence_tokens:
                continue
            truncation_ratio = (count - self.config.max_sentence_tokens) / count
            if truncation_ratio > self.config.max_sentence_truncation_ratio:
                return Rejection(
                    document_id,
                    "sentence_truncation",
                    f"sentence={index},tokens={count},ratio={truncation_ratio:.6f}",
                )

        source_tokens = sum(token_counts)
        if source_tokens < self.config.min_source_tokens:
            return Rejection(document_id, "source_too_short", f"tokens={source_tokens}")
        if source_tokens > self.config.max_source_tokens:
            return Rejection(document_id, "source_too_long", f"tokens={source_tokens}")
        if len(sentences) < self.config.min_source_sentences:
            return Rejection(document_id, "too_few_sentences", f"sentences={len(sentences)}")
        if len(sentences) > self.config.max_source_sentences:
            return Rejection(document_id, "too_many_sentences", f"sentences={len(sentences)}")

        docs = self.nlp.pipe(sentences, batch_size=256)
        content_tokens = extract_content_tokens(docs)
        oracle_labels, oracle_indices, oracle_objective = greedy_oracle_labels(
            sentences,
            reference,
            self.config.target_summary_sentences,
        )
        clean = CleanDocument(
            document_id=document_id,
            sentences=sentences,
            section_ids=section_ids,
            section_names=section_names,
            reference_sentences=reference,
            content_tokens=content_tokens,
            sentence_token_counts=token_counts,
            source_token_count=source_tokens,
            oracle_labels=oracle_labels,
            oracle_indices=oracle_indices,
            oracle_objective=oracle_objective,
        )
        clean.validate()
        return clean

    def clean_file(self, input_path: str | Path, output_path: str | Path) -> dict[str, int]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reject_path = output_path.with_suffix(output_path.suffix + ".rejected.jsonl")
        accepted = rejected = 0
        reasons: dict[str, int] = {}
        with output_path.open("w", encoding="utf-8") as output, reject_path.open(
            "w", encoding="utf-8"
        ) as rejects:
            for line_number, raw in iter_jsonl(input_path):
                value = self.clean_record(raw, f"line-{line_number}")
                if isinstance(value, Rejection):
                    rejected += 1
                    reasons[value.reason] = reasons.get(value.reason, 0) + 1
                    write_jsonl(rejects, value.__dict__)
                else:
                    accepted += 1
                    write_jsonl(output, value.to_dict())
        stats = {"accepted": accepted, "rejected": rejected, **reasons}
        output_path.with_suffix(output_path.suffix + ".stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return stats
