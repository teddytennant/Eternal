# Eternal API Reference

Complete API documentation for the Eternal RAG system.

## Table of Contents

- [Core Types](#core-types)
  - [Rag](#rag)
  - [RagConfig](#ragconfig)
  - [QueryResult](#queryresult)
  - [IndexStats](#indexstats)
- [Vector Store](#vector-store)
  - [VectorStore](#vectorstore)
  - [StoredDocument](#storeddocument)
  - [SearchResult](#searchresult)
- [Embeddings](#embeddings)
  - [TfIdfEmbedder](#tfidfembedder)
  - [SparseVector](#sparsevector)
- [Document Processing](#document-processing)
  - [Chunker](#chunker)
  - [ChunkerConfig](#chunkerconfig)
  - [Chunk](#chunk)
- [Markdown Parser](#markdown-parser)
  - [Parser](#parser)
  - [Document](#document)
  - [Block](#block)
  - [BlockType](#blocktype)

---

## Core Types

### Rag

The main orchestrator for the RAG system. Coordinates document indexing, storage, and retrieval.

```zig
pub const Rag = struct {
    allocator: Allocator,
    config: RagConfig,
    store: VectorStore,
    text_chunker: Chunker,
    md_parser: Parser,
    indexed_files: StringHashMapUnmanaged(ArrayListUnmanaged(u64)),
};
```

#### Functions

##### `init`

```zig
pub fn init(allocator: Allocator) !Rag
```

Initialize a new Rag instance with default configuration.

**Parameters:**
- `allocator`: Memory allocator for all internal allocations

**Returns:** Initialized `Rag` instance

**Errors:** `OutOfMemory` if allocation fails

**Example:**
```zig
var rag = try Rag.init(allocator);
defer rag.deinit();
```

---

##### `initWithConfig`

```zig
pub fn initWithConfig(allocator: Allocator, config: RagConfig) !Rag
```

Initialize with custom configuration.

**Parameters:**
- `allocator`: Memory allocator
- `config`: Custom `RagConfig` settings

**Returns:** Configured `Rag` instance

**Example:**
```zig
const config = RagConfig{
    .top_k = 10,
    .min_score = 0.2,
};
var rag = try Rag.initWithConfig(allocator, config);
defer rag.deinit();
```

---

##### `deinit`

```zig
pub fn deinit(self: *Rag) void
```

Clean up all resources. Must be called when done with the instance.

---

##### `indexFile`

```zig
pub fn indexFile(self: *Rag, path: []const u8) !usize
```

Index a markdown file from disk.

**Parameters:**
- `path`: Path to the markdown file

**Returns:** Number of chunks created from the file

**Errors:**
- `FileNotFound` if path doesn't exist
- `OutOfMemory` if allocation fails

**Behavior:**
- Parses markdown structure (headings, paragraphs, code blocks)
- Splits into semantically meaningful chunks
- Creates TF-IDF embeddings for each chunk
- Stores in vector store with source metadata
- If file was previously indexed, removes old chunks first

**Example:**
```zig
const num_chunks = try rag.indexFile("./docs/guide.md");
std.debug.print("Created {d} chunks\n", .{num_chunks});
```

---

##### `indexDirectory`

```zig
pub fn indexDirectory(self: *Rag, dir_path: []const u8) !IndexStats
```

Recursively index all markdown files in a directory.

**Parameters:**
- `dir_path`: Path to directory

**Returns:** `IndexStats` with counts and file list

**Behavior:**
- Walks directory recursively
- Indexes all `.md` and `.markdown` files
- Continues on individual file failures (logs warning)

**Example:**
```zig
var stats = try rag.indexDirectory("./knowledge-base/");
defer stats.deinit(allocator);

std.debug.print("Indexed {d} files, {d} chunks\n", .{
    stats.num_documents,
    stats.num_chunks,
});
```

---

##### `indexText`

```zig
pub fn indexText(self: *Rag, text: []const u8, source_name: ?[]const u8) !usize
```

Index raw text content (non-markdown).

**Parameters:**
- `text`: Text content to index
- `source_name`: Optional identifier for the source

**Returns:** Number of chunks created

**Example:**
```zig
const content = "Machine learning is a subset of AI...";
_ = try rag.indexText(content, "ml-notes");
```

---

##### `query`

```zig
pub fn query(self: *Rag, query_text: []const u8) !QueryResult
```

Search the index for relevant content.

**Parameters:**
- `query_text`: Natural language query

**Returns:** `QueryResult` containing ranked contexts

**Algorithm:**
1. Tokenizes query and removes stop words
2. Creates TF-IDF embedding for query
3. Uses inverted index to find candidate documents (O(terms) lookup)
4. Computes cosine similarity with candidates
5. Returns top-k results above minimum score threshold

**Example:**
```zig
var result = try rag.query("What is machine learning?");
defer result.deinit();

for (result.contexts.items) |ctx| {
    std.debug.print("Score: {d:.3}\n{s}\n\n", .{ctx.score, ctx.content});
}
```

---

##### `removeFile`

```zig
pub fn removeFile(self: *Rag, path: []const u8) !void
```

Remove a previously indexed file from the index.

**Parameters:**
- `path`: Path of file to remove (must match path used during indexing)

---

##### `save`

```zig
pub fn save(self: *Rag, path: []const u8) !void
```

Persist the index to disk.

**Parameters:**
- `path`: Output file path (e.g., `.eternal/index.bin`)

**File Format:**
- Binary format with magic number `ETNL`
- Version 2 includes IDF statistics for consistent search quality after reload
- Includes: documents, embeddings, term frequencies, inverted index metadata

**Example:**
```zig
try rag.save(".eternal/index.bin");
```

---

##### `load`

```zig
pub fn load(self: *Rag, path: []const u8) !void
```

Load a previously saved index.

**Parameters:**
- `path`: Path to index file

**Behavior:**
- Replaces current index contents
- Restores IDF statistics for consistent scoring
- Rebuilds inverted index for fast queries
- Rebuilds indexed_files mapping from stored metadata

**Example:**
```zig
var rag = try Rag.init(allocator);
defer rag.deinit();

rag.load(".eternal/index.bin") catch |err| {
    if (err == error.FileNotFound) {
        // Index doesn't exist yet
    } else return err;
};
```

---

##### `clear`

```zig
pub fn clear(self: *Rag) !void
```

Remove all documents from the index.

---

##### `getStats`

```zig
pub fn getStats(self: *Rag) !IndexStats
```

Get statistics about the current index.

**Returns:** `IndexStats` struct (caller must call `deinit`)

---

### RagConfig

Configuration options for the RAG system.

```zig
pub const RagConfig = struct {
    /// Number of results to retrieve (default: 5)
    top_k: usize = 5,

    /// Minimum similarity score to include in results (default: 0.1)
    min_score: f32 = 0.1,

    /// Chunker configuration
    chunker_config: ChunkerConfig = .{},

    /// Path to persist the index (optional)
    index_path: ?[]const u8 = null,
};
```

---

### QueryResult

Results from a query operation.

```zig
pub const QueryResult = struct {
    query: []const u8,
    contexts: ArrayListUnmanaged(Context),
    allocator: Allocator,

    pub const Context = struct {
        content: []const u8,    // The retrieved text
        score: f32,             // Similarity score (0-1)
        source: ?[]const u8,    // Source file path
        heading: ?[]const u8,   // Section heading context
    };
};
```

#### Functions

##### `deinit`

```zig
pub fn deinit(self: *QueryResult) void
```

Free all allocated memory. Must be called when done.

---

##### `getCombinedContext`

```zig
pub fn getCombinedContext(self: *const QueryResult, allocator: Allocator) ![]u8
```

Concatenate all contexts into a single string, separated by `---`.

**Returns:** Owned slice containing combined text (caller must free)

**Example:**
```zig
const combined = try result.getCombinedContext(allocator);
defer allocator.free(combined);

// Use combined as context for LLM
```

---

### IndexStats

Statistics about the current index.

```zig
pub const IndexStats = struct {
    num_documents: usize,                    // Number of source files
    num_chunks: usize,                       // Total chunks stored
    sources: ArrayListUnmanaged([]const u8), // List of indexed file paths
};
```

#### Functions

##### `deinit`

```zig
pub fn deinit(self: *IndexStats, allocator: Allocator) void
```

Free allocated memory.

---

## Vector Store

### VectorStore

Low-level storage and search engine for document embeddings.

```zig
pub const VectorStore = struct {
    allocator: Allocator,
    documents: ArrayListUnmanaged(StoredDocument),
    embedder: *TfIdfEmbedder,
    next_id: u64,
    inverted_index: AutoHashMapUnmanaged(u64, ArrayListUnmanaged(usize)),
};
```

#### Functions

##### `init`

```zig
pub fn init(allocator: Allocator) !VectorStore
```

Create a new empty vector store.

---

##### `deinit`

```zig
pub fn deinit(self: *VectorStore) void
```

Clean up all resources.

---

##### `addChunk`

```zig
pub fn addChunk(self: *VectorStore, chunk: Chunk) !u64
```

Add a chunk to the store.

**Returns:** Unique document ID

**Behavior:**
1. Updates corpus IDF statistics
2. Creates TF-IDF embedding
3. Stores document with embedding
4. Updates inverted index for fast lookup

---

##### `search`

```zig
pub fn search(self: *VectorStore, query: []const u8, top_k: usize) !ArrayListUnmanaged(SearchResult)
```

Find the most similar documents to a query.

**Parameters:**
- `query`: Search query text
- `top_k`: Maximum number of results

**Returns:** Ranked list of results (caller must call `deinit` on the list)

**Complexity:**
- Average case: O(candidates * terms) where candidates << total documents
- Worst case: O(n * terms) when query has no indexed terms

---

##### `remove`

```zig
pub fn remove(self: *VectorStore, id: u64) bool
```

Remove a document by ID.

**Returns:** `true` if document was found and removed

---

##### `count`

```zig
pub fn count(self: *const VectorStore) usize
```

Get the number of stored documents.

---

##### `clear`

```zig
pub fn clear(self: *VectorStore) !void
```

Remove all documents.

---

##### `saveToFile`

```zig
pub fn saveToFile(self: *const VectorStore, path: []const u8) !void
```

Serialize store to a binary file.

---

##### `loadFromFile`

```zig
pub fn loadFromFile(allocator: Allocator, path: []const u8) !VectorStore
```

Deserialize store from a binary file.

**Backward Compatibility:**
- Supports version 1 (legacy) and version 2 (with IDF stats) formats
- Automatically detects format via magic number

---

### StoredDocument

Internal representation of a stored document.

```zig
pub const StoredDocument = struct {
    id: u64,
    chunk: Chunk,
    vector: SparseVector,
};
```

---

### SearchResult

A single search result.

```zig
pub const SearchResult = struct {
    doc_id: u64,
    score: f32,
    content: []const u8,
    source_path: ?[]const u8,
    heading_context: ?[]const u8,
};
```

---

## Embeddings

### TfIdfEmbedder

TF-IDF based text embedding model.

```zig
pub const TfIdfEmbedder = struct {
    allocator: Allocator,
    doc_freq: AutoHashMapUnmanaged(u64, u32),  // Term -> document frequency
    num_docs: u32,                              // Total documents in corpus
    stop_word_hashes: AutoHashMapUnmanaged(u64, void),
};
```

#### Functions

##### `init`

```zig
pub fn init(allocator: Allocator) !TfIdfEmbedder
```

Create embedder with default English stop words (~100 words).

---

##### `initInPlace`

```zig
pub fn initInPlace(self: *TfIdfEmbedder, allocator: Allocator) !void
```

Initialize an existing struct in-place. Useful for avoiding copies.

---

##### `deinit`

```zig
pub fn deinit(self: *TfIdfEmbedder) void
```

Clean up resources.

---

##### `tokenize`

```zig
pub fn tokenize(self: *TfIdfEmbedder, text: []const u8) !ArrayListUnmanaged([]const u8)
```

Split text into tokens, filtering stop words.

**Tokenization Rules:**
- Splits on non-alphanumeric characters
- Filters tokens < 2 characters
- Case-insensitive stop word filtering
- Returns slices into original text (no allocation for token content)

---

##### `addDocument`

```zig
pub fn addDocument(self: *TfIdfEmbedder, text: []const u8) !void
```

Add a document to the corpus statistics.

**Important:** Call this before `embed()` for accurate IDF weights.

---

##### `embed`

```zig
pub fn embed(self: *TfIdfEmbedder, text: []const u8) !SparseVector
```

Create a TF-IDF embedding for text.

**Algorithm:**
1. Tokenize text
2. Compute term frequency (TF) = count / total_tokens
3. Compute inverse document frequency (IDF) = log((N+1)/(df+1)) + 1
4. Weight = TF * IDF
5. L2 normalize the vector

---

##### `hashTermStatic`

```zig
pub fn hashTermStatic(term: []const u8) u64
```

Compute case-insensitive hash for a term. Uses djb2 algorithm.

---

### SparseVector

Sparse vector representation for embeddings.

```zig
pub const SparseVector = struct {
    terms: AutoHashMapUnmanaged(u64, f32),  // term_hash -> weight
    norm: f32,                               // L2 norm
    allocator: Allocator,
};
```

#### Functions

##### `init`

```zig
pub fn init(allocator: Allocator) SparseVector
```

Create empty sparse vector.

---

##### `deinit`

```zig
pub fn deinit(self: *SparseVector) void
```

Free memory.

---

##### `clone`

```zig
pub fn clone(self: *const SparseVector, allocator: Allocator) !SparseVector
```

Create a deep copy.

---

##### `cosineSimilarity`

```zig
pub fn cosineSimilarity(self: *const SparseVector, other: *const SparseVector) f32
```

Compute cosine similarity with another vector.

**Returns:** Value in range [0, 1]

**Complexity:** O(min(|self|, |other|)) - iterates smaller vector

---

##### `computeNorm`

```zig
pub fn computeNorm(self: *SparseVector) void
```

Recompute and store L2 norm.

---

## Document Processing

### Chunker

Splits documents into retrievable chunks.

```zig
pub const Chunker = struct {
    allocator: Allocator,
    config: ChunkerConfig,
    next_id: u64,
};
```

#### Functions

##### `init`

```zig
pub fn init(allocator: Allocator) Chunker
```

Create with default config.

---

##### `initWithConfig`

```zig
pub fn initWithConfig(allocator: Allocator, config: ChunkerConfig) Chunker
```

Create with custom config.

---

##### `chunkDocument`

```zig
pub fn chunkDocument(self: *Chunker, doc: *const Document) !ArrayListUnmanaged(Chunk)
```

Chunk a parsed markdown document.

**Behavior:**
- Respects heading boundaries (configurable)
- Preserves heading context in chunks
- Applies overlap between chunks
- Finds smart split points (paragraph > sentence > word)

---

##### `chunkText`

```zig
pub fn chunkText(self: *Chunker, text: []const u8, source_path: ?[]const u8) !ArrayListUnmanaged(Chunk)
```

Chunk raw text (non-markdown).

---

### ChunkerConfig

Configuration for document chunking.

```zig
pub const ChunkerConfig = struct {
    /// Target size for each chunk in characters (default: 512)
    target_chunk_size: usize = 512,

    /// Minimum chunk size - smaller chunks merged (default: 100)
    min_chunk_size: usize = 100,

    /// Maximum chunk size - hard limit (default: 1024)
    max_chunk_size: usize = 1024,

    /// Overlap between consecutive chunks (default: 50)
    chunk_overlap: usize = 50,

    /// Whether to split at heading boundaries (default: true)
    respect_headings: bool = true,
};
```

---

### Chunk

A chunk of text with metadata.

```zig
pub const Chunk = struct {
    id: u64,
    content: []const u8,
    source_path: ?[]const u8,
    start_block: usize,
    end_block: usize,
    heading_context: ?[]const u8,
};
```

#### Functions

##### `deinit`

```zig
pub fn deinit(self: *Chunk, allocator: Allocator) void
```

Free allocated fields.

---

## Markdown Parser

### Parser

Parses markdown text into structured blocks.

```zig
pub const Parser = struct {
    allocator: Allocator,
};
```

#### Functions

##### `init`

```zig
pub fn init(allocator: Allocator) Parser
```

Create a new parser.

---

##### `parse`

```zig
pub fn parse(self: *Parser, text: []const u8) !Document
```

Parse markdown text into a Document.

**Supported Elements:**
- Headings (H1-H6)
- Paragraphs
- Code blocks (triple backtick)
- Unordered lists (`-`, `*`, `+`)
- Ordered lists (`1.`, `2.`, etc.)
- Blockquotes (`>`)
- Horizontal rules (`---`, `***`, `___`)

---

### parseFile

```zig
pub fn parseFile(allocator: Allocator, path: []const u8) !Document
```

Parse a markdown file from disk.

**Limits:** Maximum file size 10MB

---

### Document

A parsed markdown document.

```zig
pub const Document = struct {
    blocks: ArrayListUnmanaged(Block),
    allocator: Allocator,
    source_path: ?[]const u8,
};
```

#### Functions

##### `init`

```zig
pub fn init(allocator: Allocator) Document
```

Create empty document.

---

##### `deinit`

```zig
pub fn deinit(self: *Document) void
```

Free all resources.

---

##### `setSourcePath`

```zig
pub fn setSourcePath(self: *Document, path: []const u8) !void
```

Set the source file path.

---

##### `toPlainText`

```zig
pub fn toPlainText(self: *const Document, allocator: Allocator) ![]u8
```

Extract plain text (all block contents concatenated).

---

### Block

A single markdown block.

```zig
pub const Block = struct {
    block_type: BlockType,
    content: []const u8,  // Stripped of markdown syntax
    raw: []const u8,      // Original markdown text
};
```

---

### BlockType

Types of markdown blocks.

```zig
pub const BlockType = enum {
    heading1,
    heading2,
    heading3,
    heading4,
    heading5,
    heading6,
    paragraph,
    code_block,
    list_item,
    blockquote,
    horizontal_rule,
};
```

---

## Error Handling

All functions that can fail return Zig error unions. Common errors:

| Error | Description |
|-------|-------------|
| `OutOfMemory` | Allocation failed |
| `FileNotFound` | File or directory doesn't exist |
| `EndOfStream` | Unexpected end of serialized data |
| `PathAlreadyExists` | When creating directories |

---

## Thread Safety

Eternal is **not thread-safe**. All operations on a single `Rag` instance must be serialized. For concurrent access:

1. Use separate `Rag` instances per thread
2. Protect shared instances with a mutex
3. Use read-write locks for concurrent queries with exclusive indexing

---

## Memory Management

Eternal uses Zig's allocator interface for all memory operations:

- All `init` functions take an allocator
- All `deinit` functions must be called to prevent leaks
- Result types (like `QueryResult`) own their memory and must be cleaned up
- The testing allocator can detect memory leaks in tests

**Pattern:**
```zig
var rag = try Rag.init(allocator);
defer rag.deinit();  // Always pair init/deinit

var result = try rag.query("test");
defer result.deinit();  // Results need cleanup too
```
