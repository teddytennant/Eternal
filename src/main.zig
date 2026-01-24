const std = @import("std");
const Eternal = @import("Eternal");

const version = "0.1.0";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try printUsage();
        return;
    }

    if (std.mem.eql(u8, command, "version") or std.mem.eql(u8, command, "--version") or std.mem.eql(u8, command, "-v")) {
        try printVersion();
        return;
    }

    if (std.mem.eql(u8, command, "index")) {
        try handleIndex(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "query")) {
        try handleQuery(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "stats")) {
        try handleStats(allocator, args[2..]);
        return;
    }

    if (std.mem.eql(u8, command, "clear")) {
        try handleClear(allocator, args[2..]);
        return;
    }

    std.debug.print("Unknown command: {s}\n", .{command});
    try printUsage();
}

fn printUsage() !void {
    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print(
        \\Eternal - Markdown-based RAG system for continual learning
        \\
        \\Usage: eternal <command> [options]
        \\
        \\Commands:
        \\  index <path>       Index a markdown file or directory
        \\  query <text>       Query the indexed documents
        \\  stats              Show index statistics
        \\  clear              Clear the entire index
        \\  help               Show this help message
        \\  version            Show version information
        \\
        \\Options:
        \\  --index-path <path>   Path to the index file (default: .eternal/index.bin)
        \\  --top-k <n>           Number of results to return (default: 5)
        \\
        \\Examples:
        \\  eternal index ./docs/
        \\  eternal index ./README.md
        \\  eternal query "What is machine learning?"
        \\  eternal stats
        \\
    , .{});
    try stdout.flush();
}

fn printVersion() !void {
    var stdout_buffer: [256]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print("Eternal v{s}\n", .{version});
    try stdout.flush();
}

fn getIndexPath(args: []const []const u8) []const u8 {
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--index-path") and i + 1 < args.len) {
            return args[i + 1];
        }
    }
    return ".eternal/index.bin";
}

fn getTopK(args: []const []const u8) usize {
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--top-k") and i + 1 < args.len) {
            return std.fmt.parseInt(usize, args[i + 1], 10) catch 5;
        }
    }
    return 5;
}

fn ensureIndexDir(index_path: []const u8) !void {
    const dir_path = std.fs.path.dirname(index_path) orelse return;
    std.fs.cwd().makePath(dir_path) catch |err| {
        if (err != error.PathAlreadyExists) {
            return err;
        }
    };
}

fn handleIndex(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (args.len == 0) {
        try stdout.print("Error: Please provide a path to index\n", .{});
        try stdout.flush();
        return;
    }

    const index_path = getIndexPath(args);

    // Get the path to index (first non-option argument)
    var target_path: ?[]const u8 = null;
    for (args) |arg| {
        if (!std.mem.startsWith(u8, arg, "--")) {
            target_path = arg;
            break;
        }
    }

    if (target_path == null) {
        try stdout.print("Error: Please provide a path to index\n", .{});
        try stdout.flush();
        return;
    }

    try stdout.print("Indexing: {s}\n", .{target_path.?});
    try stdout.flush();

    var rag_instance = try Eternal.Rag.init(allocator);
    defer rag_instance.deinit();

    // Try to load existing index
    rag_instance.load(index_path) catch {
        // Index doesn't exist yet, that's fine
    };

    // Check if path is a file or directory
    const stat = std.fs.cwd().statFile(target_path.?) catch |err| {
        if (err == error.FileNotFound) {
            try stdout.print("Error: Path not found: {s}\n", .{target_path.?});
            try stdout.flush();
            return;
        }
        return err;
    };

    var num_chunks: usize = 0;
    var num_docs: usize = 0;

    if (stat.kind == .directory) {
        var stats = try rag_instance.indexDirectory(target_path.?);
        defer stats.deinit(allocator);
        num_chunks = stats.num_chunks;
        num_docs = stats.num_documents;
    } else {
        num_chunks = try rag_instance.indexFile(target_path.?);
        num_docs = 1;
    }

    try stdout.print("Indexed {d} document(s), {d} chunk(s)\n", .{ num_docs, num_chunks });

    // Save the index
    try ensureIndexDir(index_path);
    try rag_instance.save(index_path);
    try stdout.print("Index saved to: {s}\n", .{index_path});
    try stdout.flush();
}

fn handleQuery(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var stdout_buffer: [16384]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (args.len == 0) {
        try stdout.print("Error: Please provide a query\n", .{});
        try stdout.flush();
        return;
    }

    const index_path = getIndexPath(args);
    const top_k = getTopK(args);

    // Build query from non-option arguments
    var query_parts: std.ArrayListUnmanaged([]const u8) = .{};
    defer query_parts.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--index-path") or std.mem.eql(u8, args[i], "--top-k")) {
            i += 1; // Skip the value
            continue;
        }
        if (!std.mem.startsWith(u8, args[i], "--")) {
            try query_parts.append(allocator, args[i]);
        }
    }

    if (query_parts.items.len == 0) {
        try stdout.print("Error: Please provide a query\n", .{});
        try stdout.flush();
        return;
    }

    // Join query parts
    var query_text: std.ArrayListUnmanaged(u8) = .{};
    defer query_text.deinit(allocator);

    for (query_parts.items, 0..) |part, idx| {
        if (idx > 0) try query_text.append(allocator, ' ');
        try query_text.appendSlice(allocator, part);
    }

    const config = Eternal.RagConfig{
        .top_k = top_k,
    };

    var rag_instance = try Eternal.Rag.initWithConfig(allocator, config);
    defer rag_instance.deinit();

    // Load the index
    rag_instance.load(index_path) catch {
        try stdout.print("Error: No index found at {s}\n", .{index_path});
        try stdout.print("Run 'eternal index <path>' first to create an index.\n", .{});
        try stdout.flush();
        return;
    };

    var result = try rag_instance.query(query_text.items);
    defer result.deinit();

    if (result.contexts.items.len == 0) {
        try stdout.print("No relevant results found for: {s}\n", .{query_text.items});
        try stdout.flush();
        return;
    }

    try result.format(stdout);
    try stdout.flush();
}

fn handleStats(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    const index_path = getIndexPath(args);

    var rag_instance = try Eternal.Rag.init(allocator);
    defer rag_instance.deinit();

    // Load the index
    rag_instance.load(index_path) catch {
        try stdout.print("No index found at {s}\n", .{index_path});
        try stdout.flush();
        return;
    };

    var stats = try rag_instance.getStats();
    defer stats.deinit(allocator);

    try stdout.print("Index Statistics:\n", .{});
    try stdout.print("  Index path: {s}\n", .{index_path});
    try stdout.print("  Documents: {d}\n", .{stats.num_documents});
    try stdout.print("  Chunks: {d}\n", .{stats.num_chunks});

    if (stats.sources.items.len > 0) {
        try stdout.print("\nIndexed files:\n", .{});
        for (stats.sources.items) |source| {
            try stdout.print("  - {s}\n", .{source});
        }
    }
    try stdout.flush();
}

fn handleClear(allocator: std.mem.Allocator, args: []const []const u8) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    const index_path = getIndexPath(args);

    // Delete the index file
    std.fs.cwd().deleteFile(index_path) catch |err| {
        if (err == error.FileNotFound) {
            try stdout.print("No index found at {s}\n", .{index_path});
            try stdout.flush();
            return;
        }
        return err;
    };

    _ = allocator;
    try stdout.print("Index cleared: {s}\n", .{index_path});
    try stdout.flush();
}

test "main module" {
    // Basic test to ensure main module compiles
    _ = Eternal;
}
