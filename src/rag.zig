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
        const store = try vectorstore.VectorStore.init(allocator);
        return .{
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

        for (chunks.items) |chunk| {
            _ = try self.store.addChunk(chunk);
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
    }

    /// Clear the entire index
    pub fn clear(self: *Rag) void {
        self.store.clear();
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

    rag_inst.clear();
    try std.testing.expectEqual(@as(usize, 0), rag_inst.store.count());
}
