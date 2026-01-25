const std = @import("std");
const Allocator = std.mem.Allocator;
const markdown = @import("markdown.zig");

/// A chunk of text with metadata for retrieval
pub const Chunk = struct {
    id: u64,
    content: []const u8,
    source_path: ?[]const u8,
    start_block: usize,
    end_block: usize,
    heading_context: ?[]const u8,

    pub fn deinit(self: *Chunk, allocator: Allocator) void {
        allocator.free(self.content);
        if (self.source_path) |path| {
            allocator.free(path);
        }
        if (self.heading_context) |ctx| {
            allocator.free(ctx);
        }
    }
};

/// Configuration for the chunker
pub const ChunkerConfig = struct {
    /// Target size for each chunk in characters
    target_chunk_size: usize = 512,
    /// Minimum chunk size - chunks smaller than this will be merged
    min_chunk_size: usize = 100,
    /// Maximum chunk size - hard limit
    max_chunk_size: usize = 1024,
    /// Overlap between consecutive chunks
    chunk_overlap: usize = 50,
    /// Whether to respect heading boundaries
    respect_headings: bool = true,
};

/// Chunker for splitting documents into retrievable pieces
pub const Chunker = struct {
    allocator: Allocator,
    config: ChunkerConfig,
    next_id: u64,

    pub fn init(allocator: Allocator) Chunker {
        return .{
            .allocator = allocator,
            .config = .{},
            .next_id = 0,
        };
    }

    pub fn initWithConfig(allocator: Allocator, config: ChunkerConfig) Chunker {
        return .{
            .allocator = allocator,
            .config = config,
            .next_id = 0,
        };
    }

    /// Chunk a markdown document
    pub fn chunkDocument(self: *Chunker, doc: *const markdown.Document) !std.ArrayListUnmanaged(Chunk) {
        var chunks: std.ArrayListUnmanaged(Chunk) = .{};
        errdefer {
            for (chunks.items) |*chunk| {
                chunk.deinit(self.allocator);
            }
            chunks.deinit(self.allocator);
        }

        if (doc.blocks.items.len == 0) {
            return chunks;
        }

        var current_content: std.ArrayListUnmanaged(u8) = .{};
        defer current_content.deinit(self.allocator);

        var current_heading: ?[]const u8 = null;
        var start_block: usize = 0;

        for (doc.blocks.items, 0..) |block, i| {
            const is_heading = switch (block.block_type) {
                .heading1, .heading2, .heading3, .heading4, .heading5, .heading6 => true,
                else => false,
            };

            // Check if we should start a new chunk at heading boundaries
            if (self.config.respect_headings and is_heading) {
                // Flush current chunk if it has content
                if (current_content.items.len >= self.config.min_chunk_size) {
                    const chunk = try self.createChunk(
                        current_content.items,
                        doc.source_path,
                        start_block,
                        i - 1,
                        current_heading,
                    );
                    try chunks.append(self.allocator, chunk);
                    current_content.clearRetainingCapacity();
                }

                // Update heading context
                if (current_heading) |_| {
                    // Keep previous heading as context continues
                }
                current_heading = block.content;
                start_block = i;
            }

            // Add content to current chunk
            if (current_content.items.len > 0) {
                try current_content.appendSlice(self.allocator, "\n\n");
            }
            try current_content.appendSlice(self.allocator, block.content);

            // Check if chunk is large enough
            if (current_content.items.len >= self.config.target_chunk_size) {
                // Try to find a good split point
                const split_point = self.findSplitPoint(current_content.items);

                if (split_point > self.config.min_chunk_size) {
                    const chunk = try self.createChunk(
                        current_content.items[0..split_point],
                        doc.source_path,
                        start_block,
                        i,
                        current_heading,
                    );
                    try chunks.append(self.allocator, chunk);

                    // Keep overlap for next chunk
                    const overlap_start = if (split_point > self.config.chunk_overlap)
                        split_point - self.config.chunk_overlap
                    else
                        0;

                    const remaining = try self.allocator.dupe(u8, current_content.items[overlap_start..]);
                    current_content.clearRetainingCapacity();
                    try current_content.appendSlice(self.allocator, remaining);
                    self.allocator.free(remaining);

                    start_block = i;
                }
            }
        }

        // Flush remaining content
        if (current_content.items.len > 0) {
            const chunk = try self.createChunk(
                current_content.items,
                doc.source_path,
                start_block,
                doc.blocks.items.len - 1,
                current_heading,
            );
            try chunks.append(self.allocator, chunk);
        }

        return chunks;
    }

    /// Chunk raw text (not markdown)
    pub fn chunkText(self: *Chunker, text: []const u8, source_path: ?[]const u8) !std.ArrayListUnmanaged(Chunk) {
        var chunks: std.ArrayListUnmanaged(Chunk) = .{};
        errdefer {
            for (chunks.items) |*chunk| {
                chunk.deinit(self.allocator);
            }
            chunks.deinit(self.allocator);
        }

        var start: usize = 0;
        while (start < text.len) {
            var end = @min(start + self.config.target_chunk_size, text.len);

            // Find a good split point if not at the end
            if (end < text.len) {
                const search_start = if (end > self.config.min_chunk_size)
                    end - (self.config.target_chunk_size - self.config.min_chunk_size)
                else
                    start;

                // Look for sentence boundary
                var best_split = end;
                var idx = end;
                while (idx > search_start) : (idx -= 1) {
                    if (text[idx - 1] == '.' or text[idx - 1] == '!' or text[idx - 1] == '?') {
                        if (idx < text.len and (text[idx] == ' ' or text[idx] == '\n')) {
                            best_split = idx;
                            break;
                        }
                    }
                }
                end = best_split;
            }

            const chunk = try self.createChunk(
                text[start..end],
                source_path,
                0,
                0,
                null,
            );
            try chunks.append(self.allocator, chunk);

            // If we've reached the end, we're done
            if (end >= text.len) break;

            // Move start with overlap, ensuring forward progress
            const new_start = if (end > self.config.chunk_overlap)
                end - self.config.chunk_overlap
            else
                end;

            // Ensure we always make forward progress
            start = if (new_start <= start) end else new_start;
        }

        return chunks;
    }

    fn createChunk(
        self: *Chunker,
        content: []const u8,
        source_path: ?[]const u8,
        start_block: usize,
        end_block: usize,
        heading_context: ?[]const u8,
    ) !Chunk {
        const id = self.next_id;
        self.next_id += 1;

        return .{
            .id = id,
            .content = try self.allocator.dupe(u8, content),
            .source_path = if (source_path) |p| try self.allocator.dupe(u8, p) else null,
            .start_block = start_block,
            .end_block = end_block,
            .heading_context = if (heading_context) |h| try self.allocator.dupe(u8, h) else null,
        };
    }

    /// Find a good point to split text (prefer sentence/paragraph boundaries)
    fn findSplitPoint(self: *Chunker, text: []const u8) usize {
        const target = self.config.target_chunk_size;
        if (text.len <= target) return text.len;

        // Search backwards from target for a good split point
        const search_start = if (target > 100) target - 100 else 0;

        // First try paragraph boundary
        var idx = target;
        while (idx > search_start) : (idx -= 1) {
            if (idx + 1 < text.len and text[idx] == '\n' and text[idx + 1] == '\n') {
                return idx + 2;
            }
        }

        // Then try sentence boundary
        idx = target;
        while (idx > search_start) : (idx -= 1) {
            if (text[idx - 1] == '.' or text[idx - 1] == '!' or text[idx - 1] == '?') {
                if (idx < text.len and text[idx] == ' ') {
                    return idx;
                }
            }
        }

        // Then try word boundary
        idx = target;
        while (idx > search_start) : (idx -= 1) {
            if (text[idx] == ' ' or text[idx] == '\n') {
                return idx + 1;
            }
        }

        // Fall back to target
        return target;
    }
};

// Tests
test "chunk simple document" {
    const allocator = std.testing.allocator;

    var parser = markdown.Parser.init(allocator);
    var doc = try parser.parse(
        \\# Title
        \\
        \\First paragraph with some content.
        \\
        \\Second paragraph with more content.
    );
    defer doc.deinit();

    var chunker_inst = Chunker.init(allocator);
    var chunks = try chunker_inst.chunkDocument(&doc);
    defer {
        for (chunks.items) |*chunk| {
            chunk.deinit(allocator);
        }
        chunks.deinit(allocator);
    }

    try std.testing.expect(chunks.items.len > 0);
}

test "chunk text with overlap" {
    const allocator = std.testing.allocator;

    var chunker_inst = Chunker.initWithConfig(allocator, .{
        .target_chunk_size = 50,
        .min_chunk_size = 20,
        .chunk_overlap = 10,
    });

    const text = "This is a test sentence. Another sentence here. And one more sentence to make it longer.";
    var chunks = try chunker_inst.chunkText(text, null);
    defer {
        for (chunks.items) |*chunk| {
            chunk.deinit(allocator);
        }
        chunks.deinit(allocator);
    }

    try std.testing.expect(chunks.items.len >= 1);
}

test "chunk respects heading boundaries" {
    const allocator = std.testing.allocator;

    var parser = markdown.Parser.init(allocator);
    var doc = try parser.parse(
        \\# Section 1
        \\
        \\Content for section one.
        \\
        \\# Section 2
        \\
        \\Content for section two.
    );
    defer doc.deinit();

    var chunker_inst = Chunker.initWithConfig(allocator, .{
        .target_chunk_size = 1000, // Large enough to fit everything
        .min_chunk_size = 10,
        .respect_headings = true,
    });

    var chunks = try chunker_inst.chunkDocument(&doc);
    defer {
        for (chunks.items) |*chunk| {
            chunk.deinit(allocator);
        }
        chunks.deinit(allocator);
    }

    // Should create separate chunks for each section
    try std.testing.expect(chunks.items.len >= 2);
}
