# HyperLSS

Implementation of **HyperLSS: Hypergraph-Enhanced Lexical, Structural, and Semantic Graph Learning for Long Document Extractive Summarization**.

HyperLSS combines a heterogeneous graph over sentence and word nodes with a sentence hypergraph containing section, topic, and keyword hyperedges. The two channels are fused after each encoding layer, and the first fused sentence representation is fed back into both second-layer encoders.

## Repository structure

```text
pythonProject/
├── pre-training/src/hyperlss_preprocessing/
│   ├── cleaning.py
│   ├── cli.py
│   ├── config.py
│   ├── io.py
│   ├── oracle.py
│   ├── schema.py
│   └── text.py
├── Heterogeneous Graph Construction/
│   ├── sentence_order_edges.py
│   ├── sentences_similarity_edges.py
│   ├── word_sentence_edges.py
│   └── word_similarity_edges.py
├── Hypergraph Construction/
│   ├── section.py
│   ├── topic.py
│   └── key.py
├── HyperLSS Training/
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── model.py
│   ├── training.py
│   └── train.py
├── evaluate.py
├── build_dataset.py
└── README.md
```

## Implemented components

### Preprocessing

- normalization of common PubMed/arXiv JSONL layouts;
- UTF-8 and invalid-character checks;
- source length, sentence count, and sentence truncation filtering;
- noun (including proper nouns), verb, and adjective extraction;
- greedy oracle labels maximizing ROUGE-1, ROUGE-2, and ROUGE-L F1.

The paper-aligned preprocessing limits are: `T_max=256` encoder tokens,
minimum 100 source tokens and 10 source sentences, a maximum source length of
4,000 tokens / 400 sentences for PubMed and 5,500 tokens / 500 sentences for
arXiv, invalid-character ratio at most 5%, and sentence truncation ratio at
most 30%.

### Heterogeneous graph

| Relation ID | Relation | Construction |
|---:|---|---|
| 0 | Sentence order | Forward consecutive-sentence edge, weight 1 |
| 1 | Sentence similarity | Bidirectional edge when Sentence-BERT cosine similarity is at least 0.7 |
| 2 | Word-sentence association | Bidirectional edge with sentence-normalized TF-IDF weight |
| 3 | Word similarity | Bidirectional edge when GloVe cosine similarity is at least 0.6 |

Sentence nodes precede word nodes in each local graph. No self-loops are added.
TF-IDF IDF statistics are fitted on the training split only. Word nodes without
an original 300-dimensional GloVe vector are excluded rather than replaced by
zero vectors.

### Sentence hypergraph

- **Section:** sentences from the same original document section;
- **Topic:** sentences assigned to the same maximum-posterior topic from a 50-topic training-only LDA model;
- **Keyword:** the 15 most similar sentences for each of the top 15 KeyBERT keywords/keyphrases; if a document has fewer than 15 sentences, all of its sentences are considered.

All hyperedge types retain only hyperedges containing 2 through 25 sentences.

### Dual-channel model

- shared hidden dimension: 256;
- two relation-aware heterogeneous GAT layers;
- 4 heterogeneous attention heads with 64 dimensions per head;
- two Hypergraph Transformer layers;
- 8 hypergraph attention heads with 32 dimensions per head;
- Hypergraph Transformer FFN: 256 → 1024 → 256;
- independent gated fusion after both encoding layers;
- sentence classifier: 256 → 128 → 1;
- dropout: 0.3.

## Dependencies

Python 3.10–3.12 is recommended. Install the compatible dependency ranges
from [`requirements.txt`](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

Install the spaCy English model separately:

```bash
python -m spacy download en_core_web_sm
```

The heterogeneous graph also requires the 300-dimensional `glove.6B.300d.txt` vectors. Hugging Face model identifiers may be replaced with local model directories for offline execution.

The paper's exact experiment uses four NVIDIA RTX 4090 GPUs. CPU and
single-GPU execution are supported for development, but they are not the
reported hardware setting.

## Preprocessing

The preprocessing package accepts common JSONL fields such as:

```json
{
  "article_id": "document-id",
  "article_text": ["Sentence one.", "Sentence two."],
  "abstract_text": ["Reference summary."],
  "section_names": ["Introduction", "Methods"]
}
```

`section_names` is sentence-aligned. Repeated headings at different positions
remain separate source sections. Alternatively, provide a nested `sections`
field. The graph builder rejects records without original section membership.

Process one split:

```bash
PYTHONPATH="pre-training/src" python -m hyperlss_preprocessing clean \
  --dataset pubmed \
  --input data/pubmed/train.jsonl \
  --output processed/pubmed/train.jsonl
```

Process all splits:

```bash
PYTHONPATH="pre-training/src" python -m hyperlss_preprocessing run \
  --dataset pubmed \
  --train data/pubmed/train.jsonl \
  --validation data/pubmed/validation.jsonl \
  --test data/pubmed/test.jsonl \
  --output-dir processed/pubmed
```

## Training-data contract

The training loader reads a JSONL manifest. Each line references a numeric NPZ file and a text metadata file relative to the manifest:

```json
{"arrays":"arrays/document.npz","metadata":"metadata/document.json"}
```

Required NPZ arrays:

| Key | Shape | Description |
|---|---|---|
| `sentence_features` | `[n, 384]` | all-MiniLM-L6-v2 sentence representations |
| `word_features` | `[m, 300]` | GloVe word representations |
| `edge_index` | `[2, E]` | Directed heterogeneous edges |
| `edge_type` | `[E]` | Relation IDs 0–3 |
| `edge_weight` | `[E]` | Relation weights |
| `hyperedge_ptr` | `[H+1]` | CSR-style hyperedge offsets |
| `hyperedge_nodes` | `[I]` | Sentence members of all hyperedges |
| `hyperedge_type` | `[H]` | `section`, `topic`, or `keyword` |
| `labels` | `[n]` | Binary oracle sentence labels |

Required metadata:

```json
{
  "document_id": "document-id",
  "sentences": ["..."],
  "reference_sentences": ["..."]
}
```

The repository also includes `build_dataset.py`, the orchestration entry point
that cleans all three splits, fits training-only TF-IDF/LDA, constructs the
paper's four graph relations and three hyperedge types, and writes the NPZ,
metadata, and split manifests consumed by the training loader.

```bash
python build_dataset.py \
  --dataset pubmed \
  --train data/pubmed/train.jsonl \
  --validation data/pubmed/validation.jsonl \
  --test data/pubmed/test.jsonl \
  --glove data/glove.6B.300d.txt \
  --output structures/pubmed
```

The input must preserve original section membership (`sections` or
sentence-aligned `section_names`); the builder rejects unstructured records
instead of fabricating a section hyperedge. It accepts JSONL and the common
article-record `.npy` layout. It fits TF-IDF and LDA on the training split,
then reuses those fitted artifacts for validation and test. The generated
manifests are `structures/pubmed/train/manifest.jsonl`,
`structures/pubmed/validation/manifest.jsonl`, and
`structures/pubmed/test/manifest.jsonl`.

## Training

The implementation uses AdamW with learning rate `1e-3`, weight decay `4e-4`,
global batch size 64, at most 20 epochs, and early stopping after seven
validation-loss evaluations without improvement. The weighted BCE uses
`pos_weight=Nneg/Npos`; loss is averaged within each document before averaging
over a batch, matching Equation (33). The test checkpoint is selected by mean
validation ROUGE.

Single process:

```bash
python "HyperLSS Training/train.py" \
  --dataset pubmed \
  --train-manifest structures/pubmed/train/manifest.jsonl \
  --validation-manifest structures/pubmed/validation/manifest.jsonl \
  --test-manifest structures/pubmed/test/manifest.jsonl \
  --output-dir outputs/pubmed
```

Four GPUs:

```bash
torchrun --nproc_per_node=4 "HyperLSS Training/train.py" \
  --dataset pubmed \
  --train-manifest structures/pubmed/train/manifest.jsonl \
  --validation-manifest structures/pubmed/validation/manifest.jsonl \
  --test-manifest structures/pubmed/test/manifest.jsonl \
  --output-dir outputs/pubmed
```

Training runs three seeds and writes `best.pt`, per-seed metrics, and mean test ROUGE scores.

## Evaluation

```bash
python evaluate.py \
  --dataset pubmed \
  --manifest structures/pubmed/test/manifest.jsonl \
  --checkpoint outputs/pubmed/seed-13/best.pt \
  --output outputs/pubmed/seed-13/evaluation.json
```

Evaluation ranks sentences by predicted salience, applies lowercase punctuation-stripped trigram blocking, selects 8 sentences for PubMed or 7 for arXiv, restores source order, and reports ROUGE-1, ROUGE-2, and ROUGE-L F1. The output JSON also contains every generated extractive summary and its selected source indices.

