from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class HyperLSSBatch:
    sentence_features: torch.Tensor
    word_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_weight: torch.Tensor
    incidence_sentence: torch.Tensor
    incidence_hyperedge: torch.Tensor
    labels: torch.Tensor
    document_ptr: torch.Tensor
    sentences: list[list[str]]
    references: list[list[str]]
    document_ids: list[str]

    def to(self, device: torch.device) -> "HyperLSSBatch":
        tensor_fields = {
            name: getattr(self, name).to(device, non_blocking=True)
            for name in (
                "sentence_features", "word_features", "edge_index", "edge_type",
                "edge_weight", "incidence_sentence", "incidence_hyperedge",
                "labels", "document_ptr",
            )
        }
        return HyperLSSBatch(
            **tensor_fields,
            sentences=self.sentences,
            references=self.references,
            document_ids=self.document_ids,
        )


class HyperLSSDataset(Dataset):
    """Load graph/hypergraph samples listed by a JSONL manifest.

    Each line contains `arrays` and `metadata` paths relative to the manifest.
    Numeric NPZ keys: sentence_features, word_features, edge_index, edge_type,
    edge_weight, hyperedge_ptr, hyperedge_nodes, labels. Local node IDs place
    sentences first and words after them. Metadata contains document_id,
    sentences, and reference_sentences.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.entries: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if "arrays" not in value or "metadata" not in value:
                    raise ValueError(f"invalid manifest line {line_number}")
                self.entries.append(value)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        array_path = self.root / entry["arrays"]
        metadata_path = self.root / entry["metadata"]
        with np.load(array_path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"arrays": arrays, "metadata": metadata}

    def positive_class_weight(self) -> float:
        positives = negatives = 0
        for entry in self.entries:
            with np.load(self.root / entry["arrays"], allow_pickle=False) as stored:
                labels = stored["labels"].astype(np.int64, copy=False)
            positives += int(labels.sum())
            negatives += int(labels.size - labels.sum())
        if positives == 0:
            raise ValueError("training split contains no positive oracle labels")
        return negatives / positives


def collate_hyperlss(samples: list[dict[str, Any]]) -> HyperLSSBatch:
    sentence_features: list[torch.Tensor] = []
    word_features: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    edge_weights: list[torch.Tensor] = []
    incidence_sentences: list[torch.Tensor] = []
    incidence_hyperedges: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    document_ptr = [0]
    sentence_offset = word_offset = hyperedge_offset = 0
    texts: list[list[str]] = []
    references: list[list[str]] = []
    document_ids: list[str] = []

    # Global heterogeneous node order is all sentences followed by all words.
    total_sentences = sum(len(sample["arrays"]["sentence_features"]) for sample in samples)
    for sample in samples:
        arrays, metadata = sample["arrays"], sample["metadata"]
        sf = torch.as_tensor(arrays["sentence_features"], dtype=torch.float32)
        wf = torch.as_tensor(arrays["word_features"], dtype=torch.float32)
        n_sentences, n_words = len(sf), len(wf)
        sentence_features.append(sf)
        word_features.append(wf)
        labels.append(torch.as_tensor(arrays["labels"], dtype=torch.float32))

        local_edges = torch.as_tensor(arrays["edge_index"], dtype=torch.long)
        if local_edges.numel():
            mapped = local_edges.clone()
            for endpoint in range(2):
                values = mapped[endpoint]
                word_mask = values >= n_sentences
                values[~word_mask] += sentence_offset
                values[word_mask] = (
                    total_sentences + word_offset + values[word_mask] - n_sentences
                )
            edge_indices.append(mapped)
            edge_types.append(torch.as_tensor(arrays["edge_type"], dtype=torch.long))
            edge_weights.append(torch.as_tensor(arrays["edge_weight"], dtype=torch.float32))

        ptr = np.asarray(arrays["hyperedge_ptr"], dtype=np.int64)
        members = np.asarray(arrays["hyperedge_nodes"], dtype=np.int64)
        for local_hyperedge in range(len(ptr) - 1):
            selected = members[ptr[local_hyperedge] : ptr[local_hyperedge + 1]]
            incidence_sentences.append(
                torch.as_tensor(selected + sentence_offset, dtype=torch.long)
            )
            incidence_hyperedges.append(
                torch.full((len(selected),), hyperedge_offset, dtype=torch.long)
            )
            hyperedge_offset += 1

        sentence_offset += n_sentences
        word_offset += n_words
        document_ptr.append(sentence_offset)
        texts.append(list(metadata["sentences"]))
        references.append(list(metadata["reference_sentences"]))
        document_ids.append(str(metadata["document_id"]))

    def cat_or_empty(parts: list[torch.Tensor], shape: tuple[int, ...], dtype) -> torch.Tensor:
        return torch.cat(parts, dim=-1 if len(shape) == 2 else 0) if parts else torch.empty(shape, dtype=dtype)

    sentence_tensor = torch.cat(sentence_features, dim=0)
    word_dim = word_features[0].shape[1] if word_features else 300
    word_tensor = torch.cat(word_features, dim=0) if word_features else torch.empty((0, word_dim))
    return HyperLSSBatch(
        sentence_features=sentence_tensor,
        word_features=word_tensor,
        edge_index=cat_or_empty(edge_indices, (2, 0), torch.long),
        edge_type=cat_or_empty(edge_types, (0,), torch.long),
        edge_weight=cat_or_empty(edge_weights, (0,), torch.float32),
        incidence_sentence=cat_or_empty(incidence_sentences, (0,), torch.long),
        incidence_hyperedge=cat_or_empty(incidence_hyperedges, (0,), torch.long),
        labels=torch.cat(labels),
        document_ptr=torch.tensor(document_ptr, dtype=torch.long),
        sentences=texts,
        references=references,
        document_ids=document_ids,
    )

