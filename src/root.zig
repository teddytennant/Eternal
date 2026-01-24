//! Eternal - A Markdown-based RAG system for continual learning
//!
//! This library provides a complete RAG (Retrieval-Augmented Generation) system
//! that can index markdown documents and retrieve relevant context for queries.

const std = @import("std");

// Re-export all modules
pub const markdown = @import("markdown.zig");
pub const chunker = @import("chunker.zig");
pub const embeddings = @import("embeddings.zig");
pub const vectorstore = @import("vectorstore.zig");
pub const rag = @import("rag.zig");

// Convenience re-exports
pub const Rag = rag.Rag;
pub const RagConfig = rag.RagConfig;
pub const QueryResult = rag.QueryResult;
pub const Document = markdown.Document;
pub const Parser = markdown.Parser;
pub const Chunk = chunker.Chunk;
pub const Chunker = chunker.Chunker;
pub const VectorStore = vectorstore.VectorStore;

/// Create a new RAG instance with default configuration
pub fn createRag(allocator: std.mem.Allocator) !Rag {
    return Rag.init(allocator);
}

/// Create a new RAG instance with custom configuration
pub fn createRagWithConfig(allocator: std.mem.Allocator, config: RagConfig) !Rag {
    return Rag.initWithConfig(allocator, config);
}

// Tests
test "library integration" {
    const allocator = std.testing.allocator;

    var eternal = try createRag(allocator);
    defer eternal.deinit();

    // Index some content
    _ = try eternal.indexText(
        \\# Introduction to Zig
        \\
        \\Zig is a systems programming language designed for robustness and performance.
        \\It provides low-level control similar to C but with modern safety features.
    , "zig-intro.md");

    // Query
    var result = try eternal.query("What is Zig?");
    defer result.deinit();

    try std.testing.expect(result.contexts.items.len > 0);
}

test {
    // Run all module tests
    std.testing.refAllDecls(@This());
}
