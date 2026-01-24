const std = @import("std");
const Allocator = std.mem.Allocator;

/// Represents different types of markdown blocks
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

/// A parsed markdown block with its content and metadata
pub const Block = struct {
    block_type: BlockType,
    content: []const u8,
    raw: []const u8,

    pub fn deinit(self: *Block, allocator: Allocator) void {
        allocator.free(self.content);
        allocator.free(self.raw);
    }
};

/// Parsed markdown document
pub const Document = struct {
    blocks: std.ArrayListUnmanaged(Block),
    allocator: Allocator,
    source_path: ?[]const u8,

    pub fn init(allocator: Allocator) Document {
        return .{
            .blocks = .{},
            .allocator = allocator,
            .source_path = null,
        };
    }

    pub fn deinit(self: *Document) void {
        for (self.blocks.items) |*block| {
            block.deinit(self.allocator);
        }
        self.blocks.deinit(self.allocator);
        if (self.source_path) |path| {
            self.allocator.free(path);
        }
    }

    pub fn setSourcePath(self: *Document, path: []const u8) !void {
        if (self.source_path) |old_path| {
            self.allocator.free(old_path);
        }
        self.source_path = try self.allocator.dupe(u8, path);
    }

    /// Get plain text representation of the document
    pub fn toPlainText(self: *const Document, allocator: Allocator) ![]u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(allocator);

        for (self.blocks.items) |block| {
            try result.appendSlice(allocator, block.content);
            try result.append(allocator, '\n');
        }

        return result.toOwnedSlice(allocator);
    }
};

/// Markdown parser
pub const Parser = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) Parser {
        return .{ .allocator = allocator };
    }

    /// Parse markdown text into a Document
    pub fn parse(self: *Parser, text: []const u8) !Document {
        var doc = Document.init(self.allocator);
        errdefer doc.deinit();

        var lines = std.mem.splitScalar(u8, text, '\n');
        var current_block: std.ArrayListUnmanaged(u8) = .{};
        defer current_block.deinit(self.allocator);

        var in_code_block = false;
        var code_block_content: std.ArrayListUnmanaged(u8) = .{};
        defer code_block_content.deinit(self.allocator);

        while (lines.next()) |line| {
            // Handle code blocks
            if (std.mem.startsWith(u8, line, "```")) {
                if (in_code_block) {
                    // End code block
                    const content = try self.allocator.dupe(u8, code_block_content.items);
                    const raw = try self.allocator.dupe(u8, code_block_content.items);
                    try doc.blocks.append(self.allocator, .{
                        .block_type = .code_block,
                        .content = content,
                        .raw = raw,
                    });
                    code_block_content.clearRetainingCapacity();
                    in_code_block = false;
                } else {
                    // Start code block
                    in_code_block = true;
                }
                continue;
            }

            if (in_code_block) {
                if (code_block_content.items.len > 0) {
                    try code_block_content.append(self.allocator, '\n');
                }
                try code_block_content.appendSlice(self.allocator, line);
                continue;
            }

            // Skip empty lines but flush current block
            if (line.len == 0) {
                if (current_block.items.len > 0) {
                    const content = try self.stripMarkdown(current_block.items);
                    const raw = try self.allocator.dupe(u8, current_block.items);
                    try doc.blocks.append(self.allocator, .{
                        .block_type = .paragraph,
                        .content = content,
                        .raw = raw,
                    });
                    current_block.clearRetainingCapacity();
                }
                continue;
            }

            // Parse headings
            if (self.parseHeading(line)) |heading| {
                // Flush any pending paragraph
                if (current_block.items.len > 0) {
                    const content = try self.stripMarkdown(current_block.items);
                    const raw = try self.allocator.dupe(u8, current_block.items);
                    try doc.blocks.append(self.allocator, .{
                        .block_type = .paragraph,
                        .content = content,
                        .raw = raw,
                    });
                    current_block.clearRetainingCapacity();
                }

                const content = try self.stripMarkdown(heading.text);
                const raw = try self.allocator.dupe(u8, line);
                try doc.blocks.append(self.allocator, .{
                    .block_type = heading.level,
                    .content = content,
                    .raw = raw,
                });
                continue;
            }

            // Handle horizontal rules
            if (self.isHorizontalRule(line)) {
                if (current_block.items.len > 0) {
                    const content = try self.stripMarkdown(current_block.items);
                    const raw = try self.allocator.dupe(u8, current_block.items);
                    try doc.blocks.append(self.allocator, .{
                        .block_type = .paragraph,
                        .content = content,
                        .raw = raw,
                    });
                    current_block.clearRetainingCapacity();
                }
                const raw = try self.allocator.dupe(u8, line);
                try doc.blocks.append(self.allocator, .{
                    .block_type = .horizontal_rule,
                    .content = try self.allocator.dupe(u8, ""),
                    .raw = raw,
                });
                continue;
            }

            // Handle blockquotes
            if (std.mem.startsWith(u8, line, "> ")) {
                const quote_content = line[2..];
                const content = try self.stripMarkdown(quote_content);
                const raw = try self.allocator.dupe(u8, line);
                try doc.blocks.append(self.allocator, .{
                    .block_type = .blockquote,
                    .content = content,
                    .raw = raw,
                });
                continue;
            }

            // Handle list items
            if (self.isListItem(line)) |item_content| {
                const content = try self.stripMarkdown(item_content);
                const raw = try self.allocator.dupe(u8, line);
                try doc.blocks.append(self.allocator, .{
                    .block_type = .list_item,
                    .content = content,
                    .raw = raw,
                });
                continue;
            }

            // Regular paragraph content
            if (current_block.items.len > 0) {
                try current_block.append(self.allocator, ' ');
            }
            try current_block.appendSlice(self.allocator, line);
        }

        // Flush remaining content
        if (current_block.items.len > 0) {
            const content = try self.stripMarkdown(current_block.items);
            const raw = try self.allocator.dupe(u8, current_block.items);
            try doc.blocks.append(self.allocator, .{
                .block_type = .paragraph,
                .content = content,
                .raw = raw,
            });
        }

        return doc;
    }

    const HeadingInfo = struct {
        level: BlockType,
        text: []const u8,
    };

    fn parseHeading(self: *Parser, line: []const u8) ?HeadingInfo {
        _ = self;
        if (line.len == 0) return null;

        var level: usize = 0;
        for (line) |c| {
            if (c == '#') {
                level += 1;
            } else {
                break;
            }
        }

        if (level == 0 or level > 6) return null;
        if (line.len <= level or line[level] != ' ') return null;

        const block_type: BlockType = switch (level) {
            1 => .heading1,
            2 => .heading2,
            3 => .heading3,
            4 => .heading4,
            5 => .heading5,
            6 => .heading6,
            else => return null,
        };

        return .{
            .level = block_type,
            .text = line[level + 1 ..],
        };
    }

    fn isHorizontalRule(self: *Parser, line: []const u8) bool {
        _ = self;
        const trimmed = std.mem.trim(u8, line, " ");
        if (trimmed.len < 3) return false;

        var count: usize = 0;
        var char: ?u8 = null;
        for (trimmed) |c| {
            if (c == '-' or c == '*' or c == '_') {
                if (char == null) {
                    char = c;
                } else if (char != c) {
                    return false;
                }
                count += 1;
            } else if (c != ' ') {
                return false;
            }
        }
        return count >= 3;
    }

    fn isListItem(self: *Parser, line: []const u8) ?[]const u8 {
        _ = self;
        const trimmed = std.mem.trimLeft(u8, line, " \t");

        // Unordered list: - or * or +
        if (trimmed.len >= 2) {
            if ((trimmed[0] == '-' or trimmed[0] == '*' or trimmed[0] == '+') and trimmed[1] == ' ') {
                return trimmed[2..];
            }
        }

        // Ordered list: 1. 2. etc
        var i: usize = 0;
        while (i < trimmed.len and trimmed[i] >= '0' and trimmed[i] <= '9') {
            i += 1;
        }
        if (i > 0 and i < trimmed.len - 1 and trimmed[i] == '.' and trimmed[i + 1] == ' ') {
            return trimmed[i + 2 ..];
        }

        return null;
    }

    /// Strip markdown formatting from text (bold, italic, links, etc.)
    fn stripMarkdown(self: *Parser, text: []const u8) ![]u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(self.allocator);

        var i: usize = 0;
        while (i < text.len) {
            // Handle bold/italic markers
            if (i + 1 < text.len and text[i] == '*' and text[i + 1] == '*') {
                i += 2;
                continue;
            }
            if (text[i] == '*' or text[i] == '_') {
                i += 1;
                continue;
            }

            // Handle inline code
            if (text[i] == '`') {
                i += 1;
                continue;
            }

            // Handle links [text](url) - keep text, skip url
            if (text[i] == '[') {
                const close_bracket = std.mem.indexOfScalarPos(u8, text, i, ']');
                if (close_bracket) |cb| {
                    if (cb + 1 < text.len and text[cb + 1] == '(') {
                        const close_paren = std.mem.indexOfScalarPos(u8, text, cb + 1, ')');
                        if (close_paren) |cp| {
                            // Append link text
                            try result.appendSlice(self.allocator, text[i + 1 .. cb]);
                            i = cp + 1;
                            continue;
                        }
                    }
                }
            }

            // Handle images ![alt](url) - keep alt text
            if (i + 1 < text.len and text[i] == '!' and text[i + 1] == '[') {
                const close_bracket = std.mem.indexOfScalarPos(u8, text, i + 1, ']');
                if (close_bracket) |cb| {
                    if (cb + 1 < text.len and text[cb + 1] == '(') {
                        const close_paren = std.mem.indexOfScalarPos(u8, text, cb + 1, ')');
                        if (close_paren) |cp| {
                            try result.appendSlice(self.allocator, text[i + 2 .. cb]);
                            i = cp + 1;
                            continue;
                        }
                    }
                }
            }

            try result.append(self.allocator, text[i]);
            i += 1;
        }

        return result.toOwnedSlice(self.allocator);
    }
};

/// Parse a markdown file from disk
pub fn parseFile(allocator: Allocator, path: []const u8) !Document {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024); // 10MB max
    defer allocator.free(content);

    var parser = Parser.init(allocator);
    var doc = try parser.parse(content);
    try doc.setSourcePath(path);
    return doc;
}

// Tests
test "parse simple markdown" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);

    const md =
        \\# Hello World
        \\
        \\This is a paragraph with **bold** text.
        \\
        \\## Second heading
        \\
        \\- List item 1
        \\- List item 2
    ;

    var doc = try parser.parse(md);
    defer doc.deinit();

    try std.testing.expectEqual(@as(usize, 5), doc.blocks.items.len);
    try std.testing.expectEqual(BlockType.heading1, doc.blocks.items[0].block_type);
    try std.testing.expectEqual(BlockType.paragraph, doc.blocks.items[1].block_type);
    try std.testing.expectEqual(BlockType.heading2, doc.blocks.items[2].block_type);
}

test "parse code blocks" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);

    const md =
        \\```zig
        \\const x = 42;
        \\```
    ;

    var doc = try parser.parse(md);
    defer doc.deinit();

    try std.testing.expectEqual(@as(usize, 1), doc.blocks.items.len);
    try std.testing.expectEqual(BlockType.code_block, doc.blocks.items[0].block_type);
}

test "strip markdown formatting" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator);

    const result = try parser.stripMarkdown("**bold** and [link](url)");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("bold and link", result);
}
