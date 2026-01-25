# Eternal User Guide

A complete guide to using Eternal for personal knowledge management and AI-assisted learning.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Core Commands](#core-commands)
  - [Index](#index-command)
  - [Query](#query-command)
  - [Stats](#stats-command)
  - [Clear](#clear-command)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Introduction

Eternal is a local-first Retrieval Augmented Generation (RAG) system designed for your personal notes, documentation, and markdown files. Unlike cloud-based AI tools, Eternal runs entirely on your machine, respects your privacy, and works offline.

### What it does

1. **Reads** your markdown files (`.md`)
2. **Chunks** them into meaningful pieces
3. **Indexes** them using statistical analysis (TF-IDF)
4. **Retrieves** the most relevant content when you ask a question

### Why use it?

- **Privacy:** Your notes never leave your computer.
- **Speed:** Instant search results with no network latency.
- **Simplicity:** Zero external dependencies—just a single binary.
- **Integration:** Can be used as a CLI tool or embedded in other Zig programs.

---

## Installation

### Prerequisites

- [Zig Compiler](https://ziglang.org/download/) (version 0.15.2 or later)
- macOS, Linux, or Windows (WSL recommended)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/eternal.git
cd eternal

# Build the release binary
zig build -Doptimize=ReleaseSafe
```

The executable will be available at `zig-out/bin/eternal`.

### Add to Path (Optional)

For convenience, add the binary to your system path:

```bash
# macOS/Linux
export PATH=$PATH:$(pwd)/zig-out/bin
```

---

## Getting Started

Let's index a directory of markdown files and ask a question.

### 1. Prepare your notes

Create a folder with some markdown files:

```bash
mkdir -p my_notes
echo "# Zig Language\nZig is a general-purpose programming language and toolchain for maintaining robust, optimal and reusable software." > my_notes/zig.md
echo "# Rust Language\nRust is a multi-paradigm, high-level, general-purpose programming language that emphasizes performance, type safety, and concurrency." > my_notes/rust.md
```

### 2. Create an index

Run the `index` command pointing to your notes directory:

```bash
./zig-out/bin/eternal index my_notes
```

You should see output like:
```
Indexing: my_notes
Indexed 2 document(s), 2 chunk(s)
Index saved to: /Users/you/.eternal/index.bin
```

### 3. Ask a question

Run the `query` command:

```bash
./zig-out/bin/eternal query "what is zig"
```

Output:
```
Query: what is zig
Found 1 relevant contexts:

--- Context 1 (score: 0.707) ---
Source: my_notes/zig.md
Section: Zig Language
Zig is a general-purpose programming language and toolchain for maintaining robust, optimal and reusable software.
```

---

## Core Commands

### Index Command

Scans a file or directory recursively and builds a searchable index.

```bash
eternal index <path>
```

**Behavior:**
- Creates a new index if none exists.
- Updates the existing index if one is found.
- Automatically handles file modifications (re-indexes changed files).
- Saves the index to `~/.eternal/index.bin` (or platform equivalent).

**Examples:**
```bash
eternal index README.md           # Index a single file
eternal index ~/Documents/Notes   # Index a directory recursively
```

### Query Command

Searches the index for relevant content based on your natural language query.

```bash
eternal query <question>
```

**Behavior:**
- Loads the index into memory.
- Ranks documents by relevance (TF-IDF cosine similarity).
- Displays top results with source file and context.

**Examples:**
```bash
eternal query "how do I configure neovim"
eternal query "project architecture diagram"
```

### Stats Command

Displays statistics about the current index.

```bash
eternal stats
```

**Output:**
- Index file location
- Total number of documents
- Total chunks
- List of indexed source files

### Clear Command

Removes the index file, effectively resetting the database.

```bash
eternal clear
```

**Use case:**
- Corrupted index file
- Want to start fresh
- Freeing up disk space

---

## Best Practices

### Organizing Your Notes

Eternal works best with well-structured markdown:

1. **Use Headings:** Eternal uses headings (`#`, `##`) to provide context for chunks.
   ```markdown
   # Project X
   ## Setup
   ...
   ## Deployment
   ...
   ```

2. **Keep Files Focused:** Smaller, topic-specific files are better than one giant file.

3. **Meaningful Filenames:** The filename is stored and displayed in results.

### Performance

- **Index Size:** Eternal comfortably handles thousands of markdown files.
- **Re-indexing:** Run `eternal index` periodically to keep the search up-to-date.
- **Memory:** Search is efficient, but very large indices (hundreds of MBs) will use proportional RAM.

---

## Troubleshooting

### "No index found"

**Error:**
```
Error: No index found at /Users/you/.eternal/index.bin
Run 'eternal index <path>' first to create an index.
```

**Solution:**
You need to create an index before querying. Run `eternal index <path-to-notes>` first.

### "Command not found"

**Error:**
```
zsh: command not found: eternal
```

**Solution:**
Ensure the binary is in your PATH, or run it with the relative path `./zig-out/bin/eternal`.

### Poor Search Results

**Symptoms:**
- Query returns irrelevant documents.
- Relevant documents are missing.

**Solutions:**
1. **Check your query:** Try using more specific keywords.
2. **Re-index:** Run `eternal clear` followed by `eternal index <path>` to rebuild from scratch.
3. **Check content:** Ensure the target information actually exists in your markdown files.
