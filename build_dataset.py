"""Build the paper's HyperLSS graph/hypergraph dataset artifacts.

The script is deliberately an orchestration layer: preprocessing and graph
construction remain in their dedicated modules.  It fits TF-IDF and LDA on
the training split only, reuses those fitted objects for validation/test, and
writes one NPZ plus one metadata JSON per document and a JSONL manifest.

Example::

    python build_dataset.py --dataset pubmed \
        --train data/pubmed/train.jsonl \
        --validation data/pubmed/val.jsonl \
        --test data/pubmed/test.jsonl \
        --glove data/glove.6B.300d.txt \
        --output data/processed/pubmed

Both JSONL and the common ``.npy`` record layout (``article_id``,
``article_text``, ``abstract_text`` and original section fields) are accepted.
The input must contain original section membership; the script never invents a
section hyperedge for an unstructured article.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
PREPROCESS_SRC = ROOT / "pre-training" / "src"
GRAPH_DIR = ROOT / "Heterogeneous Graph Construction"
HYPERGRAPH_DIR = ROOT / "Hypergraph Construction"
for import_path in (PREPROCESS_SRC, GRAPH_DIR, HYPERGRAPH_DIR):
    sys.path.insert(0, str(import_path))

from hyperlss_preprocessing.cleaning import CorpusCleaner  # noqa: E402
from hyperlss_preprocessing.config import PreprocessConfig  # noqa: E402
from hyperlss_preprocessing.io import write_jsonl  # noqa: E402
from hyperlss_preprocessing.schema import CleanDocument, Rejection  # noqa: E402
from key import build_keyword_hyperedges  # noqa: E402
from section import section_hyperedges_from_ids  # noqa: E402
from sentence_order_edges import sentence_order_edges  # noqa: E402
from sentences_similarity_edges import (  # noqa: E402
    Sentences_similarity,
    Sentences_vector_embedding,
    get_feature_similarity,
    load_sentence_encoder,
)
from topic import build_topic_hyperedges, fit_training_lda, save_training_lda  # noqa: E402
from word_sentence_edges import fit_training_tfidf, word_sentence_edges  # noqa: E402
from word_similarity_edges import load_glove_vectors, word_similarity_edges  # noqa: E402


# Values specified by the paper.
SENTENCE_SIMILARITY_THRESHOLD = 0.7
WORD_SIMILARITY_THRESHOLD = 0.6
GLOVE_DIMENSION = 300
TOPIC_COUNT = 50
KEYWORD_COUNT = 15
SENTENCES_PER_KEYWORD = 15
MIN_HYPEREDGE_SIZE = 2
MAX_HYPEREDGE_SIZE = 25
MAX_SENTENCE_TOKENS = 256
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _numpy_record(value: Any) -> dict[str, Any]:
    """Convert a numpy object/structured record to an ordinary mapping."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _numpy_record(value.item())
    if isinstance(value, np.void) and value.dtype.names:
        return {name: value[name].tolist() for name in value.dtype.names}
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, Mapping):
            return dict(item)
    raise ValueError(".npy records must be mappings or structured records")


def iter_raw_records(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream JSONL or numpy records without changing split membership."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        records = np.load(source, allow_pickle=True)
        for index, value in enumerate(records, 1):
            yield index, _numpy_record(value)
        return

    with source.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            yield line_number, record


def has_original_sections(raw: Mapping[str, Any], sentence_count: int | None = None) -> bool:
    """Return whether the raw record exposes source section membership."""
    sections = raw.get("sections")
    if isinstance(sections, list) and len(sections) > 0:
        return True
    names = raw.get("section_names")
    if not isinstance(names, list) or not names:
        return False
    if sentence_count is None:
        return True
    # Sentence-aligned section names are accepted; section-level names are
    # accepted only when nested sections are present (handled above).
    return len(names) == sentence_count


def clean_split(
    raw_path: str | Path,
    cleaned_path: Path,
    cleaner: CorpusCleaner,
) -> dict[str, int]:
    """Clean one split and persist a streamable JSONL representation."""
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = cleaned_path.with_suffix(cleaned_path.suffix + ".rejected.jsonl")
    stats: dict[str, int] = {"accepted": 0, "rejected": 0}
    with cleaned_path.open("w", encoding="utf-8") as output, rejected_path.open(
        "w", encoding="utf-8"
    ) as rejects:
        for line_number, raw in iter_raw_records(raw_path):
            fallback_id = f"line-{line_number}"
            if not has_original_sections(raw):
                value: CleanDocument | Rejection = Rejection(
                    str(raw.get("article_id") or raw.get("id") or fallback_id),
                    "missing_sections",
                    "original section membership is required by HyperLSS",
                )
            else:
                value = cleaner.clean_record(raw, fallback_id)
                if (
                    isinstance(value, CleanDocument)
                    and value.section_names == ["document"]
                    and not isinstance(raw.get("sections"), list)
                ):
                    value = Rejection(
                        value.document_id,
                        "missing_sections",
                        "section membership was not preserved by the input schema",
                    )
            if isinstance(value, Rejection):
                stats["rejected"] += 1
                stats[value.reason] = stats.get(value.reason, 0) + 1
                write_jsonl(rejects, value.__dict__)
            else:
                stats["accepted"] += 1
                write_jsonl(output, value.to_dict())
    cleaned_path.with_suffix(cleaned_path.suffix + ".stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def iter_clean_documents(path: Path) -> Iterator[CleanDocument]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid cleaned JSON at {path}:{line_number}") from exc
            yield CleanDocument.from_dict(value)


def training_sentences(path: Path) -> Iterator[str]:
    for document in iter_clean_documents(path):
        yield from document.sentences


def training_content_sentences(path: Path) -> Iterator[list[str]]:
    for document in iter_clean_documents(path):
        yield from document.content_tokens


def _edge_arrays(
    edge_groups: list[tuple[int, Iterable[tuple[tuple[int, int], float]]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges: list[tuple[int, int]] = []
    types: list[int] = []
    weights: list[float] = []
    for edge_type, group in edge_groups:
        for (source, target), weight in group:
            edges.append((int(source), int(target)))
            types.append(edge_type)
            weights.append(float(weight))
    if not edges:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0,), dtype=np.int8),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.asarray(edges, dtype=np.int64).T,
        np.asarray(types, dtype=np.int8),
        np.asarray(weights, dtype=np.float32),
    )


def _hyperedge_arrays(
    groups: list[tuple[str, str, list[int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    pointers = [0]
    nodes: list[int] = []
    kinds: list[str] = []
    names: list[str] = []
    for kind, name, members in groups:
        nodes.extend(int(member) for member in members)
        pointers.append(len(nodes))
        kinds.append(kind)
        names.append(name)
    return (
        np.asarray(pointers, dtype=np.int64),
        np.asarray(nodes, dtype=np.int64),
        np.asarray(kinds, dtype=str),
        names,
    )


def build_document(
    document: CleanDocument,
    *,
    sentence_model: Any,
    keyword_model: Any,
    tfidf: Any,
    lda_dictionary: Any,
    lda_model: Any,
    glove_vectors: Mapping[str, np.ndarray],
    output_dir: Path,
    split: str,
    batch_size: int,
) -> dict[str, Any]:
    """Construct and persist one document's exact local graph artifacts."""
    sentences = document.sentences
    sentence_count = len(sentences)
    sentence_features = Sentences_vector_embedding(
        sentences, model=sentence_model, batch_size=batch_size
    )
    if sentence_features.shape[1] != 384:
        raise ValueError(
            "HyperLSS requires 384-dimensional all-MiniLM-L6-v2 sentence features; "
            f"got {sentence_features.shape[1]}"
        )
    sentence_similarity = get_feature_similarity(
        Sentences_similarity(sentence_features),
        threshold=SENTENCE_SIMILARITY_THRESHOLD,
    )
    missing_glove_words = sorted({
        word
        for sentence_tokens in document.content_tokens
        for word in sentence_tokens
        if word not in glove_vectors
    })
    # The paper initializes every retained word node with its original
    # 300-dimensional GloVe vector.  OOV words therefore do not become zero
    # vectors; they are excluded before constructing word-sentence relations.
    content_tokens = [
        [word for word in sentence_tokens if word in glove_vectors]
        for sentence_tokens in document.content_tokens
    ]
    content_relations = word_sentence_edges(content_tokens, tfidf)
    word_relations = word_similarity_edges(
        content_relations.words,
        content_relations.word_to_node,
        glove_vectors,
        threshold=WORD_SIMILARITY_THRESHOLD,
    )
    word_features = (
        np.vstack([glove_vectors[word] for word in content_relations.words])
        if content_relations.words
        else np.empty((0, GLOVE_DIMENSION), dtype=np.float32)
    )

    order = sentence_order_edges(sentence_count)
    edge_index, edge_type, edge_weight = _edge_arrays([
        (0, order.edges),
        (1, sentence_similarity),
        (2, content_relations.edges),
        (3, word_relations.edges),
    ])

    section_edges = section_hyperedges_from_ids(
        document.section_ids,
        document.section_names,
        min_hyperedge_size=MIN_HYPEREDGE_SIZE,
        max_hyperedge_size=MAX_HYPEREDGE_SIZE,
    )
    topic_ids, topic_edges = build_topic_hyperedges(
        sentences,
        lda_dictionary,
        lda_model,
        min_hyperedge_size=MIN_HYPEREDGE_SIZE,
        max_hyperedge_size=MAX_HYPEREDGE_SIZE,
    )
    keyword_edges = build_keyword_hyperedges(
        sentences,
        sentence_model=sentence_model,
        keyword_model=keyword_model,
        sentence_embeddings=sentence_features,
        keyword_count=KEYWORD_COUNT,
        sentences_per_keyword=SENTENCES_PER_KEYWORD,
        min_hyperedge_size=MIN_HYPEREDGE_SIZE,
        max_hyperedge_size=MAX_HYPEREDGE_SIZE,
        # The existing KeyBERT module's (1, 3) default is retained; the
        # paper specifies top keywords but does not specify an n-gram range.
        keyphrase_ngram_range=(1, 3),
    )
    hyperedge_groups: list[tuple[str, str, list[int]]] = []
    hyperedge_groups.extend(
        ("section", edge.section_name, edge.sentence_nodes) for edge in section_edges
    )
    hyperedge_groups.extend(
        ("topic", f"topic-{edge.topic_id}", edge.sentence_nodes) for edge in topic_edges
    )
    hyperedge_groups.extend(
        ("keyword", edge.keyword, edge.sentence_nodes) for edge in keyword_edges
    )
    hyperedge_ptr, hyperedge_nodes, hyperedge_type, hyperedge_names = _hyperedge_arrays(
        hyperedge_groups
    )

    split_dir = output_dir / split
    arrays_dir = split_dir / "arrays"
    metadata_dir = split_dir / "metadata"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    stem = document.document_id.replace("/", "_").replace("\\", "_")
    stem = stem or "document"
    npz_path = arrays_dir / f"{stem}.npz"
    metadata_path = metadata_dir / f"{stem}.json"
    np.savez_compressed(
        npz_path,
        sentence_features=np.asarray(sentence_features, dtype=np.float32),
        word_features=word_features,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        hyperedge_ptr=hyperedge_ptr,
        hyperedge_nodes=hyperedge_nodes,
        hyperedge_type=hyperedge_type,
        labels=np.asarray(document.oracle_labels, dtype=np.float32),
    )
    metadata = {
        "document_id": document.document_id,
        "sentences": sentences,
        "reference_sentences": document.reference_sentences,
        "words": content_relations.words,
        "section_names": document.section_names,
        "topic_ids": topic_ids.astype(int).tolist(),
        "hyperedge_names": hyperedge_names,
        "missing_glove_words": missing_glove_words,
        "oracle_indices": document.oracle_indices,
        "oracle_objective": document.oracle_objective,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        "document_id": document.document_id,
        # Manifest files live at output_dir/<split>/manifest.jsonl, so paths
        # are relative to that split directory for HyperLSSDataset.
        "arrays": str(npz_path.relative_to(split_dir)),
        "metadata": str(metadata_path.relative_to(split_dir)),
    }


def build_split(
    cleaned_path: Path,
    *,
    split: str,
    output_dir: Path,
    **build_kwargs: Any,
) -> int:
    manifest_path = output_dir / split / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for document in iter_clean_documents(cleaned_path):
            record = build_document(document, split=split, output_dir=output_dir, **build_kwargs)
            write_jsonl(manifest, record)
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pubmed", "arxiv"), required=True)
    parser.add_argument("--train", required=True, help="training JSONL or NPY")
    parser.add_argument("--validation", required=True, help="validation JSONL or NPY")
    parser.add_argument("--test", required=True, help="test JSONL or NPY")
    parser.add_argument("--glove", required=True, help="300d GloVe text file")
    parser.add_argument("--output", required=True, help="processed dataset directory")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--sentence-model", default=SENTENCE_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lda-workers", type=int, default=1,
                        help="LDA worker count; a resource setting, not a paper hyperparameter")
    parser.add_argument("--lda-passes", type=int, default=1,
                        help="LDA passes; the source implementation uses one pass")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    work_dir = output_dir / "preprocessed"
    work_dir.mkdir(parents=True, exist_ok=True)
    config = PreprocessConfig.for_dataset(args.dataset)
    if config.max_sentence_tokens != MAX_SENTENCE_TOKENS:
        raise ValueError("the preprocessing configuration must use T_max=256")
    config = PreprocessConfig(
        **{**config.__dict__, "sentence_model": args.sentence_model, "spacy_model": args.spacy_model}
    )
    cleaner = CorpusCleaner(config)

    raw_splits = {"train": args.train, "validation": args.validation, "test": args.test}
    clean_paths: dict[str, Path] = {}
    for split, raw_path in raw_splits.items():
        clean_path = work_dir / f"{split}.clean.jsonl"
        stats = clean_split(raw_path, clean_path, cleaner)
        clean_paths[split] = clean_path
        print(f"{split}: accepted={stats['accepted']} rejected={stats['rejected']}")

    tfidf = fit_training_tfidf(training_content_sentences(clean_paths["train"]))
    with (work_dir / "training_tfidf.pkl").open("wb") as stream:
        pickle.dump(tfidf, stream)
    lda_dictionary, lda_model = fit_training_lda(
        training_sentences(clean_paths["train"]),
        topic_count=TOPIC_COUNT,
        passes=args.lda_passes,
        workers=args.lda_workers,
    )
    save_training_lda(lda_dictionary, lda_model, work_dir / "training_lda")

    # The training vocabulary is the only vocabulary needed by this dataset.
    glove_vocabulary = set(tfidf.vocabulary_.keys())
    glove_vectors = load_glove_vectors(
        args.glove, glove_vocabulary, dimension=GLOVE_DIMENSION
    )
    sentence_model = load_sentence_encoder(args.sentence_model, device=args.device)
    # The paper fixes the sentence encoder maximum input length at 256 tokens.
    sentence_model.max_seq_length = MAX_SENTENCE_TOKENS
    try:
        from keybert import KeyBERT
    except ImportError as exc:
        raise RuntimeError("keybert is required for keyword hyperedges") from exc
    # KeyBERT wraps the very same all-MiniLM encoder used for sentence nodes.
    keyword_model = KeyBERT(model=sentence_model)

    for split, clean_path in clean_paths.items():
        count = build_split(
            clean_path,
            split=split,
            output_dir=output_dir,
            sentence_model=sentence_model,
            keyword_model=keyword_model,
            tfidf=tfidf,
            lda_dictionary=lda_dictionary,
            lda_model=lda_model,
            glove_vectors=glove_vectors,
            batch_size=args.batch_size,
        )
        print(f"{split}: wrote {count} graph documents")


if __name__ == "__main__":
    main()
