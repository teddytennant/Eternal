# Eternal
> **High-performance, local-first RAG engine for Markdown knowledge bases.**

Eternal is a Retrieval-Augmented Generation (RAG) system written in **Zig**. It is designed to act as a persistent, updatable external memory for AI agents and humans alike. Unlike Python-based vector stores or heavy cloud infrastructure, Eternal compiles to a single, lightning-fast binary with zero runtime dependencies.

## Why Eternal?

Traditional approaches to teaching computers new information are flawed. Eternal takes a different approach focused on speed, efficiency, and simplicity.

### 1. Solves the "Continual Learning" Problem
*   **The Old Way (Fine-tuning):** To teach an LLM new information, you typically have to fine-tune the model. This is slow, computationally expensive, and often leads to **"catastrophic forgetting"** (where the model loses previous knowledge to make room for the new).
*   **The Eternal Way:** Eternal separates *reasoning* (the LLM) from *knowledge* (the Index). To learn something new, you simply index a new Markdown file. The knowledge is **immediately available** for retrieval without any training. You can keep adding files forever without degrading previous knowledge.

### 2. Radical Efficiency
*   **The Old Way (Python/Docker):** Typical RAG pipelines are heavy. They often require gigabytes of RAM, complex Python virtual environments, Docker containers, and sometimes even GPUs just to run the database.
*   **The Eternal Way:** Written in **Zig**, Eternal runs in a few megabytes of RAM. It utilizes sparse vector embedding (TF-IDF) which is computationally cheap yet highly effective for text retrieval. It runs instantly on any laptop, no GPU required.

### 3. Native Markdown Understanding
*   **The Old Way:** Many chunkers blindly cut text at fixed character limits, breaking sentences and losing the context of where headers or sections begin.
*   **The Eternal Way:** Eternal is built for Markdown. It respects document structure, headings, and sections, ensuring that retrieved chunks retain their semantic context.

## FAQ: Is this an AI Agent (like Claude)?

No. It is important to distinguish between the **Brain** (Reasoning) and the **Memory** (Retrieval).

*   **Claude/ChatGPT (The Brain):** These are reasoning engines. They are smart but have limited memory (context window) and don't know your private data unless you paste it in.
*   **Eternal (The Memory):** Eternal is a retrieval engine. It is not "smart" in the reasoning sense—it cannot write code or summarize text for you. Its job is to find the *exact* pieces of information relevant to your query from a massive library of documents.

**How they work together:**
Eternal provides the "R" (Retrieval) in a RAG pipeline. You use Eternal to find the relevant docs, and then you feed those docs into an agent like Claude to generate the answer.

## Features

*   ⚡ **Blazing Fast:** Written in Zig for native performance.
*   📄 **Markdown Support:** First-class support for `.md` files, preserving document structure.
*   🔍 **Hybrid Search:** Uses advanced TF-IDF weighting and Cosine Similarity to find the most relevant context.
*   💾 **Persistent:** Saves your index to disk (`.eternal/index.bin`) for instant loading on subsequent runs.
*   🛠️ **CLI Interface:** Simple, composable commands.

## Installation & Build

Ensure you have [Zig](https://ziglang.org/download/) (0.15.2 or later) installed.

```bash
# Clone the repository
git clone https://github.com/teddytennant/Eternal
cd eternal

# Build for release (optimized)
zig build -Doptimize=ReleaseFast
```

The executable will be located at `zig-out/bin/eternal`.

## Usage

Eternal works through a simple CLI interface.

### 1. Indexing Knowledge
Index a single file or an entire directory of Markdown files.

```bash
# Index a directory of notes
./zig-out/bin/eternal index ./my-notes/

# Index a specific file
./zig-out/bin/eternal index ./README.md
```

### 2. Querying
Ask questions to your knowledge base. Eternal will return the most relevant context chunks.

```bash
./zig-out/bin/eternal query "What is the difference between TCP and UDP?"
```

### 3. Check Statistics
See how big your knowledge base has grown.

```bash
./zig-out/bin/eternal stats
```

### 4. Clear Index
Start fresh.

```bash
./zig-out/bin/eternal clear
```

## How it Works

1.  **Ingestion:** Eternal reads your Markdown files.
2.  **Chunking:** It splits text into manageable pieces, respecting paragraph and header boundaries.
3.  **Embedding:** It calculates sparse vector embeddings (TF-IDF) for every chunk.
4.  **Retrieval:** When you query, it finds the vectors mathematically closest to your query's vector (Cosine Similarity) and returns the original text.

## License

MIT
