# Eternal Architecture

This document describes the internal architecture, design decisions, and implementation details of the Eternal RAG system.

## System Overview

```
                                    +-----------------+
                                    |    CLI/API      |
                                    |   (main.zig)    |
                                    +--------+--------+
                                             |
                                             v
+----------------+            +-----------------------------+
|   Markdown     |            |           RAG               |
|    Files       +----------->|        (rag.zig)            |
+----------------+            |   Orchestrates all ops      |
                              +----+------------+-----------+
                                   |            |
                    +--------------+            +--------------+
                    v                                          v
          +------------------+                    +----------------------+
          |  Markdown Parser |                    |    Vector Store      |
          |  (markdown.zig)  |                    |  (vectorstore.zig)   |
          +--------+---------+                    +----------+-----------+
                   |                                         |
                   v                                         v
          +------------------+                    +----------------------+
          |     Chunker      |                    |   TF-IDF Embedder    |
          |   (chunker.zig)  |                    |   (embeddings.zig)   |
          +------------------+                    +----------------------+
```

## Component Details

### 1. RAG Orchestrator (`rag.zig`)

The `Rag` struct is the central coordinator that manages the complete indexing and retrieval pipeline.

**Responsibilities:**
- File I/O and directory traversal
- Coordinating parser, chunker, and store
- Managing file-to-chunk mappings
- Index persistence

**Key Data Structures:**
```zig
pub const Rag = struct {
    allocator: Allocator,
    config: RagConfig,
    store: VectorStore,              // Owns all embeddings
    text_chunker: Chunker,           // Stateless chunking
    md_parser: Parser,               // Stateless parsing
    indexed_files: StringHashMap(...), // path -> [chunk_ids]
};
```

**Design Decisions:**
- Single ownership model: `Rag` owns `VectorStore` which owns all documents
- Lazy IDF computation: Statistics updated incrementally during indexing
- Re-indexing detection: Tracks files to remove old chunks before re-indexing

---

### 2. Markdown Parser (`markdown.zig`)

A streaming parser that converts markdown text to structured blocks.

**Parsing Strategy:**
1. Line-by-line processing
2. State machine for code blocks (triple backtick toggle)
3. Regex-free heading detection (count leading `#` characters)
4. Inline markdown stripped from content (bold, italic, links)

**Supported Elements:**

| Element | Detection | Example |
|---------|-----------|---------|
| Heading | `^#{1,6} ` | `## Title` |
| Code Block | `^```.*$` | ` ```zig ` |
| List Item | `^[-*+] ` or `^\d+\. ` | `- item` |
| Blockquote | `^> ` | `> quote` |
| Horizontal Rule | `^[-*_]{3,}$` | `---` |
| Paragraph | Default | Plain text |

**Why Custom Parser?**
- Zero dependencies
- Minimal allocations (reuses line buffer)
- Preserves structure information for smart chunking
- Strip markdown syntax for cleaner embeddings

---

### 3. Chunker (`chunker.zig`)

Splits documents into semantically meaningful pieces optimized for retrieval.

**Chunking Algorithm:**

```
For each document:
  1. Group blocks by heading boundaries (if configured)
  2. Accumulate content until target_chunk_size
  3. Find best split point:
     a. Paragraph boundary (\n\n)
     b. Sentence boundary (. ! ?)
     c. Word boundary (space)
  4. Create chunk with overlap from previous
  5. Preserve heading context for each chunk
```

**Configuration Defaults:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `target_chunk_size` | 512 | Characters per chunk |
| `min_chunk_size` | 100 | Avoid tiny chunks |
| `max_chunk_size` | 1024 | Hard upper limit |
| `chunk_overlap` | 50 | Context continuity |
| `respect_headings` | true | Semantic boundaries |

**Why These Defaults?**
- 512 characters ≈ 100-150 tokens ≈ good retrieval granularity
- Overlap prevents context loss at chunk boundaries
- Heading respect keeps related content together

---

### 4. TF-IDF Embedder (`embeddings.zig`)

Converts text to sparse vector representations using Term Frequency-Inverse Document Frequency.

**Algorithm:**

```
TF(term, doc) = count(term in doc) / total_terms_in_doc

IDF(term) = log((N + 1) / (df(term) + 1)) + 1

Weight(term, doc) = TF(term, doc) * IDF(term)

Vector = L2_normalize({term_hash: weight})
```

**Why TF-IDF over Dense Embeddings?**

| Aspect | TF-IDF (Sparse) | Dense (Neural) |
|--------|-----------------|----------------|
| Memory | ~1KB/doc | ~3KB/doc (768 dims) |
| CPU | Cheap | GPU preferred |
| Dependencies | None | PyTorch/ONNX |
| Interpretability | High | Low |
| Quality | Good for lexical | Better for semantic |

TF-IDF is chosen for Eternal because:
1. Zero dependencies aligns with project goals
2. Excellent for documentation/technical content
3. Fast indexing and search
4. Interpretable (you can see which terms matched)

**Stop Words:**

~100 English stop words are filtered during tokenization:
- Articles: a, an, the
- Pronouns: I, you, he, she, it, we, they
- Prepositions: in, on, at, by, for, with
- Conjunctions: and, but, or, if, because
- Common verbs: is, are, was, were, be, have, do

**Term Hashing:**

Terms are hashed using djb2 (case-insensitive) for compact storage:

```zig
hash = 5381
for c in term:
    hash = ((hash << 5) + hash) + lowercase(c)
```

---

### 5. Vector Store (`vectorstore.zig`)

Persistent storage with fast similarity search.

**Data Structures:**

```zig
pub const VectorStore = struct {
    documents: ArrayList(StoredDocument),  // Linear storage
    embedder: *TfIdfEmbedder,             // Shared embedder
    inverted_index: HashMap(term_hash -> [doc_indices]),
    next_id: u64,
};
```

**Search Algorithm:**

```
1. Embed query using same TF-IDF method
2. Collect candidate documents from inverted index:
   For each query term:
     Add all documents containing that term to candidates
3. Score only candidates (not all documents):
   score = cosine_similarity(query_vec, doc_vec)
4. Sort by score descending
5. Return top-k results
```

**Complexity Analysis:**

| Operation | Average Case | Worst Case |
|-----------|--------------|------------|
| Add | O(terms) | O(terms) |
| Search | O(candidates * query_terms) | O(n * query_terms) |
| Remove | O(n) | O(n) |

The inverted index typically reduces candidates to 1-10% of total documents for well-distributed queries.

**Serialization Format (v2):**

```
[Magic: u32 = 0x45544E4C ("ETNL")]
[Version: u32 = 2]
[NumDocs: u32]
[NextID: u64]
[NumDocs (IDF): u32]
[DocFreqCount: u32]
[DocFreq entries: (term_hash: u64, freq: u32)...]
[Documents: (
    id: u64,
    chunk_id: u64,
    content_len: u32,
    content: bytes,
    path_len: u32,
    path: bytes,
    heading_len: u32,
    heading: bytes,
    start_block: u64,
    end_block: u64,
    num_terms: u32,
    terms: (hash: u64, weight: f32)...,
    norm: f32
)...]
```

**Backward Compatibility:**
- Version 1 files (no magic) are detected and loaded
- Missing IDF stats default to num_docs=0, flat weighting
- Search still works, but quality may differ slightly

---

## Data Flow

### Indexing Flow

```
File Path
    |
    v
[Read File] --> Raw Markdown Text
    |
    v
[Parse] --> Document { blocks: [Block] }
    |
    v
[Chunk] --> [Chunk] (with heading context)
    |
    v
[For each chunk:]
    |
    +---> [Add to corpus] --> Updates IDF stats
    |
    +---> [Embed] --> SparseVector
    |
    +---> [Store] --> Adds to documents + inverted index
    |
    v
[Track] --> indexed_files[path] = [chunk_ids]
```

### Query Flow

```
Query Text
    |
    v
[Tokenize] --> ["machine", "learning"]
    |
    v
[Embed] --> SparseVector (using corpus IDF)
    |
    v
[Inverted Index Lookup] --> Candidate doc indices
    |
    v
[Score Candidates] --> (doc_idx, similarity_score)[]
    |
    v
[Sort & Filter] --> Top-k results above min_score
    |
    v
QueryResult { contexts: [...] }
```

---

## Performance Characteristics

### Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| Per chunk | ~200 bytes | ID, pointers, metadata |
| Per embedding | ~50-200 bytes | Depends on unique terms |
| IDF table | ~16 bytes/term | Hash + count |
| Inverted index | ~20 bytes/posting | Hash + doc list |

**Rule of thumb:** ~500 bytes per chunk including overhead

For 10,000 chunks: ~5MB RAM

### Indexing Speed

Factors affecting indexing speed:
1. File I/O (dominant for large files)
2. Markdown parsing (linear in text size)
3. Tokenization (linear in text size)
4. Hash table operations (amortized O(1))

**Approximate rates:** 1-10 MB/second depending on hardware

### Query Speed

For N documents with average T terms per document:

| Index Size | Typical Query Time |
|------------|-------------------|
| 100 docs | < 1ms |
| 1,000 docs | 1-5ms |
| 10,000 docs | 5-20ms |
| 100,000 docs | 50-200ms |

The inverted index keeps query time sub-linear in practice.

---

## Design Trade-offs

### Chosen: TF-IDF over Dense Embeddings

**Pros:**
- No ML framework dependencies
- Fast, CPU-only computation
- Interpretable results
- Smaller memory footprint

**Cons:**
- No semantic understanding ("car" won't match "automobile")
- Less effective for short queries
- English-centric stop words

### Chosen: Single-File Binary Format

**Pros:**
- Atomic saves (no partial state)
- Easy backup/restore
- No database dependency

**Cons:**
- Full rewrite on save
- Memory spike during serialization
- 100MB practical limit

### Chosen: In-Memory Inverted Index

**Pros:**
- Fast O(1) term lookup
- No external index files
- Automatically rebuilt on load

**Cons:**
- Memory overhead (~20% extra)
- Rebuild cost on load

---

## Future Improvements

### Near-term

1. **Incremental saves** - Only write changed documents
2. **Compression** - LZ4 for serialized data
3. **SIMD similarity** - Vectorized dot products

### Medium-term

1. **Hybrid search** - TF-IDF + simple semantic similarity
2. **Index sharding** - Multiple index files for large corpora
3. **Query expansion** - Automatic synonym handling

### Long-term

1. **Optional dense embeddings** - Via ONNX runtime
2. **Approximate nearest neighbor** - HNSW or IVF
3. **Distributed search** - Multi-node indexing

---

## Testing Strategy

### Unit Tests

Each module has focused unit tests:
- `embeddings.zig`: Tokenization, embedding creation, similarity
- `vectorstore.zig`: CRUD operations, serialization
- `chunker.zig`: Boundary detection, overlap handling
- `markdown.zig`: Block parsing, syntax stripping

### Integration Tests

Full pipeline tests in `rag.zig`:
- Index-save-load-query roundtrip
- IDF statistics preservation
- File mapping reconstruction

### Running Tests

```bash
# All tests
zig build test

# Single module
zig test src/embeddings.zig

# With leak detection (testing allocator)
zig test src/rag.zig
```

---

## Code Organization

```
src/
├── main.zig          # CLI entry point
├── root.zig          # Library exports
├── rag.zig           # High-level orchestrator
├── vectorstore.zig   # Storage + search
├── embeddings.zig    # TF-IDF implementation
├── chunker.zig       # Document splitting
└── markdown.zig      # Markdown parser

docs/
├── API.md            # API reference
├── ARCHITECTURE.md   # This document
└── USAGE.md          # Usage guide

test_docs/            # Sample markdown for testing
```

---

## Contributing

When contributing to Eternal:

1. **Follow Zig idioms** - Use allocators, error unions, defer
2. **Add tests** - Every new function needs test coverage
3. **Document APIs** - Public functions need doc comments
4. **Benchmark changes** - Performance regressions are bugs
5. **Keep dependencies minimal** - Stdlib only

Run before submitting:
```bash
zig build                    # Must compile
zig build test              # Must pass
zig fmt src/*.zig           # Must be formatted
```
