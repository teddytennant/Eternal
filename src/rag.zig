const std = @import("std");
const Allocator = std.mem.Allocator;
const markdown = @import("markdown.zig");
const chunker = @import("chunker.zig");
const vectorstore = @import("vectorstore.zig");

/// Configuration for the RAG system
pub const RagConfig = struct {
    /// Number of results to retrieve
    top_k: usize = 5,
    /// Minimum similarity score to include in results
    min_score: f32 = 0.1,
    /// Chunker configuration
    chunker_config: chunker.ChunkerConfig = .{},
    /// Path to persist the index
    index_path: ?[]const u8 = null,
};

/// Query result containing retrieved context
pub const QueryResult = struct {
    query: []const u8,
    contexts: std.ArrayListUnmanaged(Context),
    allocator: Allocator,

    pub const Context = struct {
        content: []const u8,
        score: f32,
        source: ?[]const u8,
        heading: ?[]const u8,
    };

    pub fn deinit(self: *QueryResult) void {
        self.allocator.free(self.query);
        for (self.contexts.items) |ctx| {
            self.allocator.free(ctx.content);
            if (ctx.source) |s| self.allocator.free(s);
            if (ctx.heading) |h| self.allocator.free(h);
        }
        self.contexts.deinit(self.allocator);
    }

    /// Format contexts for display
    pub fn format(self: *const QueryResult, writer: anytype) !void {
        try writer.print("Query: {s}\n", .{self.query});
        try writer.print("Found {d} relevant contexts:\n\n", .{self.contexts.items.len});

        for (self.contexts.items, 0..) |ctx, i| {
            try writer.print("--- Context {d} (score: {d:.3}) ---\n", .{ i + 1, ctx.score });
            if (ctx.source) |src| {
                try writer.print("Source: {s}\n", .{src});
            }
            if (ctx.heading) |heading| {
                try writer.print("Section: {s}\n", .{heading});
            }
            try writer.print("{s}\n\n", .{ctx.content});
        }
    }

    /// Get combined context as a single string
    pub fn getCombinedContext(self: *const QueryResult, allocator: Allocator) ![]u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(allocator);

        for (self.contexts.items, 0..) |ctx, i| {
            if (i > 0) {
                try result.appendSlice(allocator, "\n\n---\n\n");
            }
            try result.appendSlice(allocator, ctx.content);
        }

        return result.toOwnedSlice(allocator);
    }
};

/// Index statistics
pub const IndexStats = struct {
    num_documents: usize,
    num_chunks: usize,
    sources: std.ArrayListUnmanaged([]const u8),

    pub fn deinit(self: *IndexStats, allocator: Allocator) void {
        for (self.sources.items) |src| {
            allocator.free(src);
        }
        self.sources.deinit(allocator);
    }
};

/// RAG system for retrieving relevant context from documents
pub const Rag = struct {
    allocator: Allocator,
    config: RagConfig,
    store: vectorstore.VectorStore,
    text_chunker: chunker.Chunker,
    md_parser: markdown.Parser,
    indexed_files: std.StringHashMapUnmanaged(std.ArrayListUnmanaged(u64)),

    pub fn init(allocator: Allocator) !Rag {
        return initWithConfig(allocator, .{});
    }

    pub fn initWithConfig(allocator: Allocator, config: RagConfig) !Rag {
        // Force alignment for the store by allocating it separately on the heap
        // This is a workaround for hash map alignment issues when embedded in a struct
        // Note: Rag.store is a value, but internally it uses pointers.
        // The issue is likely with Rag.indexed_files or VectorStore.inverted_index
        // when Rag is stack allocated.

        // Let's create the store first
        const store = try vectorstore.VectorStore.init(allocator);

        // We can't easily change the return type to *Rag without breaking API.
        // Instead, let's try to ensure the large structs inside are heap allocated where possible.
        // VectorStore already holds pointers.

        // The crash happens in header() calculation of a hash map.
        // This suggests one of the hash maps is not aligned to 8 bytes.
        // Stack allocation of Rag might be under-aligned.

        return Rag{
            .allocator = allocator,
            .config = config,
            .store = store,
            .text_chunker = chunker.Chunker.initWithConfig(allocator, config.chunker_config),
            .md_parser = markdown.Parser.init(allocator),
            .indexed_files = .{},
        };
    }

    pub fn deinit(self: *Rag) void {
        self.store.deinit();
        var iter = self.indexed_files.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.indexed_files.deinit(self.allocator);
    }

    /// Index a markdown file
    pub fn indexFile(self: *Rag, path: []const u8) !usize {
        // Check if already indexed
        if (self.indexed_files.contains(path)) {
            // Remove old chunks first
            try self.removeFile(path);
        }

        // Parse the markdown file
        var doc = try markdown.parseFile(self.allocator, path);
        defer doc.deinit();

        // Chunk the document
        var chunks = try self.text_chunker.chunkDocument(&doc);
        defer {
            for (chunks.items) |*chunk| {
                chunk.deinit(self.allocator);
            }
            chunks.deinit(self.allocator);
        }

        // Add chunks to store and track IDs
        var chunk_ids: std.ArrayListUnmanaged(u64) = .{};
        errdefer chunk_ids.deinit(self.allocator);

        for (chunks.items) |chunk| {
            const id = try self.store.addChunk(chunk);
            try chunk_ids.append(self.allocator, id);
        }

        // Store the mapping
        const path_copy = try self.allocator.dupe(u8, path);
        try self.indexed_files.put(self.allocator, path_copy, chunk_ids);

        return chunks.items.len;
    }

    /// Index a directory of markdown files
    pub fn indexDirectory(self: *Rag, dir_path: []const u8) !IndexStats {
        var stats = IndexStats{
            .num_documents = 0,
            .num_chunks = 0,
            .sources = .{},
        };
        errdefer stats.deinit(self.allocator);

        var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| {
            if (err == error.FileNotFound) {
                return stats;
            }
            return err;
        };
        defer dir.close();

        var walker = dir.walk(self.allocator) catch |err| {
            return err;
        };
        defer walker.deinit();

        while (try walker.next()) |entry| {
            if (entry.kind != .file) continue;

            // Check for markdown files
            if (std.mem.endsWith(u8, entry.basename, ".md") or
                std.mem.endsWith(u8, entry.basename, ".markdown"))
            {
                // Build full path
                const full_path = try std.fs.path.join(self.allocator, &.{ dir_path, entry.path });
                defer self.allocator.free(full_path);

                const num_chunks = self.indexFile(full_path) catch |err| {
                    std.debug.print("Warning: Failed to index {s}: {}\n", .{ full_path, err });
                    continue;
                };

                stats.num_documents += 1;
                stats.num_chunks += num_chunks;
                try stats.sources.append(self.allocator, try self.allocator.dupe(u8, full_path));
            }
        }

        return stats;
    }

    /// Index raw text
    pub fn indexText(self: *Rag, text: []const u8, source_name: ?[]const u8) !usize {
        var chunks = try self.text_chunker.chunkText(text, source_name);
        defer {
            for (chunks.items) |*chunk| {
                chunk.deinit(self.allocator);
            }
            chunks.deinit(self.allocator);
        }

        // Track IDs for this source
        var chunk_ids: std.ArrayListUnmanaged(u64) = .{};
        errdefer chunk_ids.deinit(self.allocator);

        for (chunks.items) |chunk| {
            const id = try self.store.addChunk(chunk);
            try chunk_ids.append(self.allocator, id);
        }

        // Track the source name if provided
        if (source_name) |name| {
            const name_copy = try self.allocator.dupe(u8, name);
            try self.indexed_files.put(self.allocator, name_copy, chunk_ids);
        } else {
            chunk_ids.deinit(self.allocator);
        }

        return chunks.items.len;
    }

    /// Remove a file from the index
    pub fn removeFile(self: *Rag, path: []const u8) !void {
        if (self.indexed_files.fetchRemove(path)) |kv| {
            self.allocator.free(kv.key);
            for (kv.value.items) |id| {
                _ = self.store.remove(id);
            }
            var val = kv.value;
            val.deinit(self.allocator);
        }
    }

    /// Query the RAG system
    pub fn query(self: *Rag, query_text: []const u8) !QueryResult {
        var results = try self.store.search(query_text, self.config.top_k);
        defer results.deinit(self.allocator);

        var query_result = QueryResult{
            .query = try self.allocator.dupe(u8, query_text),
            .contexts = .{},
            .allocator = self.allocator,
        };
        errdefer query_result.deinit();

        for (results.items) |result| {
            if (result.score < self.config.min_score) continue;

            try query_result.contexts.append(self.allocator, .{
                .content = try self.allocator.dupe(u8, result.content),
                .score = result.score,
                .source = if (result.source_path) |s| try self.allocator.dupe(u8, s) else null,
                .heading = if (result.heading_context) |h| try self.allocator.dupe(u8, h) else null,
            });
        }

        return query_result;
    }

    /// Get index statistics
    pub fn getStats(self: *Rag) !IndexStats {
        var stats = IndexStats{
            .num_documents = self.indexed_files.count(),
            .num_chunks = self.store.count(),
            .sources = .{},
        };
        errdefer stats.deinit(self.allocator);

        var iter = self.indexed_files.keyIterator();
        while (iter.next()) |key| {
            try stats.sources.append(self.allocator, try self.allocator.dupe(u8, key.*));
        }

        return stats;
    }

    /// Save the index to disk
    pub fn save(self: *Rag, path: []const u8) !void {
        try self.store.saveToFile(path);
    }

    /// Load the index from disk
    pub fn load(self: *Rag, path: []const u8) !void {
        self.store.deinit();
        self.store = try vectorstore.VectorStore.loadFromFile(self.allocator, path);

        // Rebuild indexed_files mapping from loaded documents
        var iter = self.indexed_files.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.indexed_files.clearRetainingCapacity();

        // Group document IDs by source path
        // First pass: collect unique source paths
        var source_set: std.StringHashMapUnmanaged(void) = .{};
        defer source_set.deinit(self.allocator);

        for (self.store.documents.items) |doc| {
            if (doc.chunk.source_path) |source| {
                try source_set.put(self.allocator, source, {});
            }
        }

        // Second pass: create owned keys and collect doc IDs
        var source_iter = source_set.keyIterator();
        while (source_iter.next()) |source_ptr| {
            const source = source_ptr.*;
            const owned_key = try self.allocator.dupe(u8, source);
            errdefer self.allocator.free(owned_key);

            var id_list: std.ArrayListUnmanaged(u64) = .{};
            errdefer id_list.deinit(self.allocator);

            for (self.store.documents.items) |doc| {
                if (doc.chunk.source_path) |doc_source| {
                    if (std.mem.eql(u8, doc_source, source)) {
                        try id_list.append(self.allocator, doc.id);
                    }
                }
            }

            try self.indexed_files.put(self.allocator, owned_key, id_list);
        }
    }

    /// Clear the entire index
    pub fn clear(self: *Rag) !void {
        try self.store.clear();
        var iter = self.indexed_files.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.indexed_files.clearRetainingCapacity();
    }
};

// Tests
test "index and query" {
    const allocator = std.testing.allocator;

    var rag_inst = try Rag.init(allocator);
    defer rag_inst.deinit();

    // Index some text
    _ = try rag_inst.indexText(
        "Machine learning is a branch of artificial intelligence. It enables computers to learn from data.",
        "ml.md",
    );
    _ = try rag_inst.indexText(
        "Deep learning is a subset of machine learning using neural networks with multiple layers.",
        "dl.md",
    );
    _ = try rag_inst.indexText(
        "Cooking pasta requires water, salt, and pasta. Boil the water first.",
        "cooking.md",
    );

    // Query
    var result = try rag_inst.query("What is machine learning?");
    defer result.deinit();

    try std.testing.expect(result.contexts.items.len > 0);
    // Should retrieve ML-related content first
    try std.testing.expect(std.mem.indexOf(u8, result.contexts.items[0].content, "learning") != null);
}

test "get stats" {
    const allocator = std.testing.allocator;

    var rag_inst = try Rag.init(allocator);
    defer rag_inst.deinit();

    _ = try rag_inst.indexText("Test content one", "test1.md");
    _ = try rag_inst.indexText("Test content two", "test2.md");

    var stats = try rag_inst.getStats();
    defer stats.deinit(allocator);

    try std.testing.expect(stats.num_chunks >= 2);
}

test "clear index" {
    const allocator = std.testing.allocator;

    var rag_inst = try Rag.init(allocator);
    defer rag_inst.deinit();

    _ = try rag_inst.indexText("Test content", "test.md");
    try std.testing.expect(rag_inst.store.count() > 0);

    try rag_inst.clear();
    try std.testing.expectEqual(@as(usize, 0), rag_inst.store.count());
}

test "full roundtrip: index -> save -> load -> query" {
    const allocator = std.testing.allocator;

    // Create a temporary file path for testing
    const test_index_path = "/tmp/eternal_test_index.bin";

    // Phase 1: Index and save
    {
        var rag_inst = try Rag.init(allocator);
        defer rag_inst.deinit();

        _ = try rag_inst.indexText(
            "Zig is a systems programming language designed to be simple and predictable.",
            "zig.md",
        );
        _ = try rag_inst.indexText(
            "Rust is a systems programming language focused on safety and concurrency.",
            "rust.md",
        );
        _ = try rag_inst.indexText(
            "Python is a high-level interpreted language known for readability.",
            "python.md",
        );

        try rag_inst.save(test_index_path);
    }

    // Phase 2: Load and verify query still works
    {
        var rag_inst = try Rag.init(allocator);
        defer rag_inst.deinit();

        try rag_inst.load(test_index_path);

        // Verify document count preserved
        try std.testing.expectEqual(@as(usize, 3), rag_inst.store.count());

        // Verify query returns relevant results
        var result = try rag_inst.query("systems programming language");
        defer result.deinit();

        try std.testing.expect(result.contexts.items.len >= 2);

        // First results should be about systems languages (Zig or Rust), not Python
        const first_content = result.contexts.items[0].content;
        const is_systems_lang = std.mem.indexOf(u8, first_content, "systems") != null;
        try std.testing.expect(is_systems_lang);
    }

    // Cleanup
    std.fs.cwd().deleteFile(test_index_path) catch {};
}

test "indexed_files mapping survives roundtrip" {
    const allocator = std.testing.allocator;
    const test_index_path = "/tmp/eternal_test_index2.bin";

    // Phase 1: Index with source paths and save
    {
        var rag_inst = try Rag.init(allocator);
        defer rag_inst.deinit();

        _ = try rag_inst.indexText("Content from file A", "fileA.md");
        _ = try rag_inst.indexText("Content from file B", "fileB.md");

        // Verify indexed_files has entries
        try std.testing.expectEqual(@as(usize, 2), rag_inst.indexed_files.count());

        try rag_inst.save(test_index_path);
    }

    // Phase 2: Load and verify indexed_files is rebuilt
    {
        var rag_inst = try Rag.init(allocator);
        defer rag_inst.deinit();

        try rag_inst.load(test_index_path);

        // Verify indexed_files was rebuilt from loaded documents
        var stats = try rag_inst.getStats();
        defer stats.deinit(allocator);

        try std.testing.expectEqual(@as(usize, 2), stats.sources.items.len);
    }

    // Cleanup
    std.fs.cwd().deleteFile(test_index_path) catch {};
}

test "IDF statistics survive roundtrip" {
    const allocator = std.testing.allocator;
    const test_index_path = "/tmp/eternal_test_index3.bin";

    // Phase 1: Index documents to build IDF stats
    var original_score: f32 = 0;
    {
        var rag_inst = try Rag.init(allocator);
        defer rag_inst.deinit();

        // Index documents with specific term distributions
        _ = try rag_inst.indexText("machine learning artificial intelligence", "ml.md");
        _ = try rag_inst.indexText("machine learning deep neural networks", "dl.md");
        _ = try rag_inst.indexText("cooking recipes food preparation", "cooking.md");

        // Query and record score
        var result = try rag_inst.query("machine learning");
        defer result.deinit();

        try std.testing.expect(result.contexts.items.len > 0);
        original_score = result.contexts.items[0].score;

        try rag_inst.save(test_index_path);
    }

    // Phase 2: Load and verify IDF statistics are preserved (scores should match)
    {
        var rag_inst = try Rag.init(allocator);
        defer rag_inst.deinit();

        try rag_inst.load(test_index_path);

        var result = try rag_inst.query("machine learning");
        defer result.deinit();

        try std.testing.expect(result.contexts.items.len > 0);
        const loaded_score = result.contexts.items[0].score;

        // Scores should be identical if IDF was properly preserved
        try std.testing.expectApproxEqAbs(original_score, loaded_score, 0.001);
    }

    // Cleanup
    std.fs.cwd().deleteFile(test_index_path) catch {};
}
