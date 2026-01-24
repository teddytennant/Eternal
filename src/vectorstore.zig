const std = @import("std");
const Allocator = std.mem.Allocator;
const embeddings = @import("embeddings.zig");
const chunker = @import("chunker.zig");

/// A stored document with its embedding
pub const StoredDocument = struct {
    id: u64,
    chunk: chunker.Chunk,
    vector: embeddings.SparseVector,

    pub fn deinit(self: *StoredDocument, allocator: Allocator) void {
        self.chunk.deinit(allocator);
        self.vector.deinit();
    }
};

/// Search result with score
pub const SearchResult = struct {
    doc_id: u64,
    score: f32,
    content: []const u8,
    source_path: ?[]const u8,
    heading_context: ?[]const u8,
};

/// Vector store for storing and searching document embeddings
pub const VectorStore = struct {
    allocator: Allocator,
    documents: std.ArrayListUnmanaged(StoredDocument),
    embedder: embeddings.TfIdfEmbedder,
    next_id: u64,

    pub fn init(allocator: Allocator) !VectorStore {
        return .{
            .allocator = allocator,
            .documents = .{},
            .embedder = try embeddings.TfIdfEmbedder.init(allocator),
            .next_id = 0,
        };
    }

    pub fn deinit(self: *VectorStore) void {
        for (self.documents.items) |*doc| {
            doc.deinit(self.allocator);
        }
        self.documents.deinit(self.allocator);
        self.embedder.deinit();
    }

    /// Add a chunk to the store
    pub fn addChunk(self: *VectorStore, chunk: chunker.Chunk) !u64 {
        // Update IDF statistics
        try self.embedder.addDocument(chunk.content);

        // Create embedding
        const vector = try self.embedder.embed(chunk.content);

        const id = self.next_id;
        self.next_id += 1;

        // Clone the chunk for storage
        const stored_chunk = chunker.Chunk{
            .id = chunk.id,
            .content = try self.allocator.dupe(u8, chunk.content),
            .source_path = if (chunk.source_path) |p| try self.allocator.dupe(u8, p) else null,
            .start_block = chunk.start_block,
            .end_block = chunk.end_block,
            .heading_context = if (chunk.heading_context) |h| try self.allocator.dupe(u8, h) else null,
        };

        try self.documents.append(self.allocator, .{
            .id = id,
            .chunk = stored_chunk,
            .vector = vector,
        });

        return id;
    }

    /// Add multiple chunks
    pub fn addChunks(self: *VectorStore, chunks: []const chunker.Chunk) !std.ArrayListUnmanaged(u64) {
        var ids: std.ArrayListUnmanaged(u64) = .{};
        errdefer ids.deinit(self.allocator);

        for (chunks) |chunk| {
            const id = try self.addChunk(chunk);
            try ids.append(self.allocator, id);
        }

        return ids;
    }

    /// Search for similar documents
    pub fn search(self: *VectorStore, query: []const u8, top_k: usize) !std.ArrayListUnmanaged(SearchResult) {
        var results: std.ArrayListUnmanaged(SearchResult) = .{};
        errdefer results.deinit(self.allocator);

        if (self.documents.items.len == 0) {
            return results;
        }

        // Embed the query
        var query_vec = try self.embedder.embed(query);
        defer query_vec.deinit();

        // Score all documents
        const ScoredDoc = struct {
            idx: usize,
            score: f32,
        };

        var scored_docs: std.ArrayListUnmanaged(ScoredDoc) = .{};
        defer scored_docs.deinit(self.allocator);

        for (self.documents.items, 0..) |doc, idx| {
            const score = query_vec.cosineSimilarity(&doc.vector);
            if (score > 0) {
                try scored_docs.append(self.allocator, .{ .idx = idx, .score = score });
            }
        }

        // Sort by score descending
        std.mem.sort(ScoredDoc, scored_docs.items, {}, struct {
            fn lessThan(_: void, a: ScoredDoc, b: ScoredDoc) bool {
                return a.score > b.score;
            }
        }.lessThan);

        // Take top k results
        const limit = @min(top_k, scored_docs.items.len);
        for (scored_docs.items[0..limit]) |scored| {
            const doc = &self.documents.items[scored.idx];
            try results.append(self.allocator, .{
                .doc_id = doc.id,
                .score = scored.score,
                .content = doc.chunk.content,
                .source_path = doc.chunk.source_path,
                .heading_context = doc.chunk.heading_context,
            });
        }

        return results;
    }

    /// Remove a document by ID
    pub fn remove(self: *VectorStore, id: u64) bool {
        for (self.documents.items, 0..) |*doc, idx| {
            if (doc.id == id) {
                doc.deinit(self.allocator);
                _ = self.documents.orderedRemove(idx);
                return true;
            }
        }
        return false;
    }

    /// Get document count
    pub fn count(self: *const VectorStore) usize {
        return self.documents.items.len;
    }

    /// Clear all documents
    pub fn clear(self: *VectorStore) void {
        for (self.documents.items) |*doc| {
            doc.deinit(self.allocator);
        }
        self.documents.clearRetainingCapacity();
        self.embedder.deinit();
        self.embedder = embeddings.TfIdfEmbedder.init(self.allocator) catch unreachable;
        self.next_id = 0;
    }

    fn writeU32(list: *std.ArrayListUnmanaged(u8), allocator: Allocator, value: u32) !void {
        const bytes = std.mem.toBytes(value);
        try list.appendSlice(allocator, &bytes);
    }

    fn writeU64(list: *std.ArrayListUnmanaged(u8), allocator: Allocator, value: u64) !void {
        const bytes = std.mem.toBytes(value);
        try list.appendSlice(allocator, &bytes);
    }

    fn writeF32(list: *std.ArrayListUnmanaged(u8), allocator: Allocator, value: f32) !void {
        const bytes = std.mem.toBytes(value);
        try list.appendSlice(allocator, &bytes);
    }

    fn readU32(data: []const u8, offset: *usize) !u32 {
        if (offset.* + 4 > data.len) return error.EndOfStream;
        const bytes = data[offset.*..][0..4];
        offset.* += 4;
        return std.mem.bytesToValue(u32, bytes);
    }

    fn readU64(data: []const u8, offset: *usize) !u64 {
        if (offset.* + 8 > data.len) return error.EndOfStream;
        const bytes = data[offset.*..][0..8];
        offset.* += 8;
        return std.mem.bytesToValue(u64, bytes);
    }

    fn readF32(data: []const u8, offset: *usize) !f32 {
        if (offset.* + 4 > data.len) return error.EndOfStream;
        const bytes = data[offset.*..][0..4];
        offset.* += 4;
        return std.mem.bytesToValue(f32, bytes);
    }

    fn readBytes(data: []const u8, offset: *usize, len: usize) ![]const u8 {
        if (offset.* + len > data.len) return error.EndOfStream;
        const bytes = data[offset.*..][0..len];
        offset.* += len;
        return bytes;
    }

    /// Serialize the store to bytes
    pub fn serializeToBytes(self: *const VectorStore, allocator: Allocator) ![]u8 {
        var buffer: std.ArrayListUnmanaged(u8) = .{};
        errdefer buffer.deinit(allocator);

        // Write header
        try writeU32(&buffer, allocator, @intCast(self.documents.items.len));
        try writeU64(&buffer, allocator, self.next_id);

        // Write each document
        for (self.documents.items) |doc| {
            try writeU64(&buffer, allocator, doc.id);

            // Write chunk
            try writeU64(&buffer, allocator, doc.chunk.id);
            try writeU32(&buffer, allocator, @intCast(doc.chunk.content.len));
            try buffer.appendSlice(allocator, doc.chunk.content);

            // Source path
            if (doc.chunk.source_path) |path| {
                try writeU32(&buffer, allocator, @intCast(path.len));
                try buffer.appendSlice(allocator, path);
            } else {
                try writeU32(&buffer, allocator, 0);
            }

            // Heading context
            if (doc.chunk.heading_context) |ctx| {
                try writeU32(&buffer, allocator, @intCast(ctx.len));
                try buffer.appendSlice(allocator, ctx);
            } else {
                try writeU32(&buffer, allocator, 0);
            }

            try writeU64(&buffer, allocator, @intCast(doc.chunk.start_block));
            try writeU64(&buffer, allocator, @intCast(doc.chunk.end_block));

            // Write vector
            try writeU32(&buffer, allocator, @intCast(doc.vector.terms.count()));
            var iter = doc.vector.terms.iterator();
            while (iter.next()) |entry| {
                try writeU64(&buffer, allocator, entry.key_ptr.*);
                try writeF32(&buffer, allocator, entry.value_ptr.*);
            }
            try writeF32(&buffer, allocator, doc.vector.norm);
        }

        return buffer.toOwnedSlice(allocator);
    }

    /// Deserialize the store from bytes
    pub fn deserializeFromBytes(allocator: Allocator, data: []const u8) !VectorStore {
        var store = try VectorStore.init(allocator);
        errdefer store.deinit();

        var offset: usize = 0;

        const num_docs = try readU32(data, &offset);
        store.next_id = try readU64(data, &offset);

        var i: u32 = 0;
        while (i < num_docs) : (i += 1) {
            const id = try readU64(data, &offset);

            // Read chunk
            const chunk_id = try readU64(data, &offset);
            const content_len = try readU32(data, &offset);
            const content_data = try readBytes(data, &offset, content_len);
            const content = try allocator.dupe(u8, content_data);
            errdefer allocator.free(content);

            // Source path
            const path_len = try readU32(data, &offset);
            const source_path = if (path_len > 0) blk: {
                const path_data = try readBytes(data, &offset, path_len);
                break :blk try allocator.dupe(u8, path_data);
            } else null;
            errdefer if (source_path) |p| allocator.free(p);

            // Heading context
            const ctx_len = try readU32(data, &offset);
            const heading_context = if (ctx_len > 0) blk: {
                const ctx_data = try readBytes(data, &offset, ctx_len);
                break :blk try allocator.dupe(u8, ctx_data);
            } else null;
            errdefer if (heading_context) |c| allocator.free(c);

            const start_block = try readU64(data, &offset);
            const end_block = try readU64(data, &offset);

            // Read vector
            const num_terms = try readU32(data, &offset);
            var vector = embeddings.SparseVector.init(allocator);
            errdefer vector.deinit();

            var j: u32 = 0;
            while (j < num_terms) : (j += 1) {
                const term_hash = try readU64(data, &offset);
                const weight = try readF32(data, &offset);
                try vector.terms.put(allocator, term_hash, weight);
            }

            vector.norm = try readF32(data, &offset);

            try store.documents.append(allocator, .{
                .id = id,
                .chunk = .{
                    .id = chunk_id,
                    .content = content,
                    .source_path = source_path,
                    .start_block = @intCast(start_block),
                    .end_block = @intCast(end_block),
                    .heading_context = heading_context,
                },
                .vector = vector,
            });
        }

        return store;
    }

    /// Save store to file
    pub fn saveToFile(self: *const VectorStore, path: []const u8) !void {
        const data = try self.serializeToBytes(self.allocator);
        defer self.allocator.free(data);

        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        try file.writeAll(data);
    }

    /// Load store from file
    pub fn loadFromFile(allocator: Allocator, path: []const u8) !VectorStore {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const data = try file.readToEndAlloc(allocator, 100 * 1024 * 1024); // 100MB max
        defer allocator.free(data);

        return try deserializeFromBytes(allocator, data);
    }
};

// Tests
test "add and search chunks" {
    const allocator = std.testing.allocator;

    var store = try VectorStore.init(allocator);
    defer store.deinit();

    // Add some test chunks
    const chunks = [_]chunker.Chunk{
        .{
            .id = 0,
            .content = "Machine learning is a subset of artificial intelligence",
            .source_path = null,
            .start_block = 0,
            .end_block = 0,
            .heading_context = null,
        },
        .{
            .id = 1,
            .content = "Deep learning uses neural networks with many layers",
            .source_path = null,
            .start_block = 0,
            .end_block = 0,
            .heading_context = null,
        },
        .{
            .id = 2,
            .content = "Cooking pasta requires boiling water and salt",
            .source_path = null,
            .start_block = 0,
            .end_block = 0,
            .heading_context = null,
        },
    };

    for (chunks) |chunk| {
        _ = try store.addChunk(chunk);
    }

    try std.testing.expectEqual(@as(usize, 3), store.count());

    // Search for ML-related content
    var results = try store.search("artificial intelligence machine learning", 2);
    defer results.deinit(allocator);

    try std.testing.expect(results.items.len > 0);
    // First result should be about ML, not cooking
    try std.testing.expect(std.mem.indexOf(u8, results.items[0].content, "learning") != null);
}

test "remove document" {
    const allocator = std.testing.allocator;

    var store = try VectorStore.init(allocator);
    defer store.deinit();

    const chunk = chunker.Chunk{
        .id = 0,
        .content = "Test content",
        .source_path = null,
        .start_block = 0,
        .end_block = 0,
        .heading_context = null,
    };

    const id = try store.addChunk(chunk);
    try std.testing.expectEqual(@as(usize, 1), store.count());

    const removed = store.remove(id);
    try std.testing.expect(removed);
    try std.testing.expectEqual(@as(usize, 0), store.count());
}

test "serialize and deserialize" {
    const allocator = std.testing.allocator;

    var original = try VectorStore.init(allocator);
    defer original.deinit();

    const chunk = chunker.Chunk{
        .id = 0,
        .content = "Test content for serialization",
        .source_path = null,
        .start_block = 0,
        .end_block = 0,
        .heading_context = null,
    };

    _ = try original.addChunk(chunk);

    // Serialize
    const data = try original.serializeToBytes(allocator);
    defer allocator.free(data);

    // Deserialize
    var restored = try VectorStore.deserializeFromBytes(allocator, data);
    defer restored.deinit();

    try std.testing.expectEqual(original.count(), restored.count());
}
