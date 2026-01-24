const std = @import("std");
const Allocator = std.mem.Allocator;

/// A sparse vector representation using term frequencies
pub const SparseVector = struct {
    /// Map from term hash to weight
    terms: std.AutoHashMapUnmanaged(u64, f32),
    /// L2 norm of the vector (for cosine similarity)
    norm: f32,
    allocator: Allocator,

    pub fn init(allocator: Allocator) SparseVector {
        return .{
            .terms = .{},
            .norm = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SparseVector) void {
        self.terms.deinit(self.allocator);
    }

    pub fn clone(self: *const SparseVector, allocator: Allocator) !SparseVector {
        var new_vec = SparseVector.init(allocator);
        var iter = self.terms.iterator();
        while (iter.next()) |entry| {
            try new_vec.terms.put(allocator, entry.key_ptr.*, entry.value_ptr.*);
        }
        new_vec.norm = self.norm;
        return new_vec;
    }

    /// Compute cosine similarity with another vector
    pub fn cosineSimilarity(self: *const SparseVector, other: *const SparseVector) f32 {
        if (self.norm == 0 or other.norm == 0) return 0;

        var dot_product: f32 = 0;

        // Iterate over the smaller vector for efficiency
        const smaller = if (self.terms.count() < other.terms.count()) self else other;
        const larger = if (self.terms.count() < other.terms.count()) other else self;

        var iter = smaller.terms.iterator();
        while (iter.next()) |entry| {
            if (larger.terms.get(entry.key_ptr.*)) |other_weight| {
                dot_product += entry.value_ptr.* * other_weight;
            }
        }

        return dot_product / (self.norm * other.norm);
    }

    /// Compute L2 norm and store it
    pub fn computeNorm(self: *SparseVector) void {
        var sum_squares: f32 = 0;
        var iter = self.terms.iterator();
        while (iter.next()) |entry| {
            sum_squares += entry.value_ptr.* * entry.value_ptr.*;
        }
        self.norm = @sqrt(sum_squares);
    }
};

/// TF-IDF based embedding model
pub const TfIdfEmbedder = struct {
    allocator: Allocator,
    /// Document frequency for each term
    doc_freq: std.AutoHashMapUnmanaged(u64, u32),
    /// Total number of documents
    num_docs: u32,
    /// Stop word hashes for fast lookup
    stop_word_hashes: std.AutoHashMapUnmanaged(u64, void),

    pub fn init(allocator: Allocator) !TfIdfEmbedder {
        var embedder = TfIdfEmbedder{
            .allocator = allocator,
            .doc_freq = .{},
            .num_docs = 0,
            .stop_word_hashes = .{},
        };

        // Initialize common English stop words (stored as hashes)
        const stop_word_list = [_][]const u8{
            "a",      "an",    "the",    "is",      "are",   "was",    "were",
            "be",     "been",  "being",  "have",    "has",   "had",    "do",
            "does",   "did",   "will",   "would",   "could", "should", "may",
            "might",  "must",  "shall",  "can",     "need",  "dare",   "ought",
            "used",   "to",    "of",     "in",      "for",   "on",     "with",
            "at",     "by",    "from",   "as",      "into",  "through", "during",
            "before", "after", "above",  "below",   "between", "under", "again",
            "further", "then", "once",   "here",    "there", "when",   "where",
            "why",    "how",   "all",    "each",    "few",   "more",   "most",
            "other",  "some",  "such",   "no",      "nor",   "not",    "only",
            "own",    "same",  "so",     "than",    "too",   "very",   "just",
            "and",    "but",   "if",     "or",      "because", "until", "while",
            "this",   "that",  "these",  "those",   "i",     "me",     "my",
            "myself", "we",    "our",    "ours",    "you",   "your",   "yours",
            "he",     "him",   "his",    "she",     "her",   "hers",   "it",
            "its",    "they",  "them",   "their",   "what",  "which",  "who",
        };

        for (stop_word_list) |word| {
            const hash = hashTermStatic(word);
            try embedder.stop_word_hashes.put(allocator, hash, {});
        }

        return embedder;
    }

    pub fn deinit(self: *TfIdfEmbedder) void {
        self.doc_freq.deinit(self.allocator);
        self.stop_word_hashes.deinit(self.allocator);
    }

    /// Tokenize text into terms
    pub fn tokenize(self: *TfIdfEmbedder, text: []const u8) !std.ArrayListUnmanaged([]const u8) {
        var tokens: std.ArrayListUnmanaged([]const u8) = .{};
        errdefer tokens.deinit(self.allocator);

        var start: ?usize = null;
        for (text, 0..) |c, idx| {
            const is_alpha = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z');
            const is_digit = c >= '0' and c <= '9';

            if (is_alpha or is_digit) {
                if (start == null) {
                    start = idx;
                }
            } else {
                if (start) |s| {
                    const token = text[s..idx];
                    if (token.len >= 2 and !self.isStopWord(token)) {
                        try tokens.append(self.allocator, token);
                    }
                    start = null;
                }
            }
        }

        // Handle last token
        if (start) |s| {
            const token = text[s..];
            if (token.len >= 2 and !self.isStopWord(token)) {
                try tokens.append(self.allocator, token);
            }
        }

        return tokens;
    }

    /// Check if a token is a stop word (case-insensitive)
    fn isStopWord(self: *TfIdfEmbedder, token: []const u8) bool {
        const hash = hashTermStatic(token);
        return self.stop_word_hashes.contains(hash);
    }

    /// Hash a term for storage (case-insensitive) - static version
    fn hashTermStatic(term: []const u8) u64 {
        var hash: u64 = 5381;
        for (term) |c| {
            // Convert to lowercase for hashing
            const lower = if (c >= 'A' and c <= 'Z') c + 32 else c;
            hash = ((hash << 5) +% hash) +% lower;
        }
        return hash;
    }

    /// Hash a term for storage (case-insensitive)
    fn hashTerm(self: *TfIdfEmbedder, term: []const u8) u64 {
        _ = self;
        return hashTermStatic(term);
    }

    /// Add a document to the corpus (updates IDF statistics)
    pub fn addDocument(self: *TfIdfEmbedder, text: []const u8) !void {
        var seen_terms: std.AutoHashMapUnmanaged(u64, void) = .{};
        defer seen_terms.deinit(self.allocator);

        var tokens = try self.tokenize(text);
        defer tokens.deinit(self.allocator);

        for (tokens.items) |token| {
            const hash = self.hashTerm(token);
            if (!seen_terms.contains(hash)) {
                try seen_terms.put(self.allocator, hash, {});
                const current = self.doc_freq.get(hash) orelse 0;
                try self.doc_freq.put(self.allocator, hash, current + 1);
            }
        }

        self.num_docs += 1;
    }

    /// Embed text into a sparse vector
    pub fn embed(self: *TfIdfEmbedder, text: []const u8) !SparseVector {
        var vec = SparseVector.init(self.allocator);
        errdefer vec.deinit();

        var tokens = try self.tokenize(text);
        defer tokens.deinit(self.allocator);

        // Count term frequencies
        var term_counts: std.AutoHashMapUnmanaged(u64, u32) = .{};
        defer term_counts.deinit(self.allocator);

        for (tokens.items) |token| {
            const hash = self.hashTerm(token);
            const current = term_counts.get(hash) orelse 0;
            try term_counts.put(self.allocator, hash, current + 1);
        }

        // Compute TF-IDF weights
        const num_tokens: f32 = @floatFromInt(tokens.items.len);
        if (num_tokens == 0) {
            return vec;
        }

        var iter = term_counts.iterator();
        while (iter.next()) |entry| {
            const tf: f32 = @as(f32, @floatFromInt(entry.value_ptr.*)) / num_tokens;

            // IDF with smoothing
            const df = self.doc_freq.get(entry.key_ptr.*) orelse 1;
            const num_docs_f: f32 = @floatFromInt(@max(self.num_docs, 1));
            const df_f: f32 = @floatFromInt(df);
            const idf = @log((num_docs_f + 1) / (df_f + 1)) + 1;

            const weight = tf * idf;
            try vec.terms.put(self.allocator, entry.key_ptr.*, weight);
        }

        vec.computeNorm();
        return vec;
    }

    /// Batch embed multiple texts
    pub fn embedBatch(self: *TfIdfEmbedder, texts: []const []const u8) !std.ArrayListUnmanaged(SparseVector) {
        var vectors: std.ArrayListUnmanaged(SparseVector) = .{};
        errdefer {
            for (vectors.items) |*v| {
                v.deinit();
            }
            vectors.deinit(self.allocator);
        }

        for (texts) |text| {
            const vec = try self.embed(text);
            try vectors.append(self.allocator, vec);
        }

        return vectors;
    }
};

/// Simple BM25 implementation for ranking
pub const BM25 = struct {
    k1: f32 = 1.2,
    b: f32 = 0.75,
    avg_doc_len: f32,
    doc_freq: *std.AutoHashMapUnmanaged(u64, u32),
    num_docs: u32,
    allocator: Allocator,

    pub fn score(self: *const BM25, query_terms: []const u64, doc_terms: std.AutoHashMapUnmanaged(u64, u32), doc_len: usize) f32 {
        var total_score: f32 = 0;
        const doc_len_f: f32 = @floatFromInt(doc_len);

        for (query_terms) |term| {
            const tf = doc_terms.get(term) orelse continue;
            const df = self.doc_freq.get(term) orelse 1;

            const tf_f: f32 = @floatFromInt(tf);
            const df_f: f32 = @floatFromInt(df);
            const num_docs_f: f32 = @floatFromInt(self.num_docs);

            // IDF component
            const idf = @log((num_docs_f - df_f + 0.5) / (df_f + 0.5) + 1);

            // TF component with length normalization
            const tf_norm = (tf_f * (self.k1 + 1)) /
                (tf_f + self.k1 * (1 - self.b + self.b * (doc_len_f / self.avg_doc_len)));

            total_score += idf * tf_norm;
        }

        return total_score;
    }
};

// Tests
test "tokenize text" {
    const allocator = std.testing.allocator;
    var embedder = try TfIdfEmbedder.init(allocator);
    defer embedder.deinit();

    var tokens = try embedder.tokenize("Hello world! This is a test.");
    defer tokens.deinit(allocator);

    // Should filter out stop words and short tokens
    try std.testing.expect(tokens.items.len > 0);
}

test "embed text" {
    const allocator = std.testing.allocator;
    var embedder = try TfIdfEmbedder.init(allocator);
    defer embedder.deinit();

    // Add some documents to build IDF
    try embedder.addDocument("The quick brown fox jumps over the lazy dog");
    try embedder.addDocument("A fast brown fox leaps over a sleepy dog");

    var vec = try embedder.embed("brown fox");
    defer vec.deinit();

    try std.testing.expect(vec.terms.count() > 0);
    try std.testing.expect(vec.norm > 0);
}

test "cosine similarity" {
    const allocator = std.testing.allocator;
    var embedder = try TfIdfEmbedder.init(allocator);
    defer embedder.deinit();

    try embedder.addDocument("machine learning algorithms");
    try embedder.addDocument("deep learning neural networks");
    try embedder.addDocument("cooking recipes food");

    var vec1 = try embedder.embed("machine learning");
    defer vec1.deinit();

    var vec2 = try embedder.embed("learning algorithms");
    defer vec2.deinit();

    var vec3 = try embedder.embed("food recipes");
    defer vec3.deinit();

    const sim12 = vec1.cosineSimilarity(&vec2);
    const sim13 = vec1.cosineSimilarity(&vec3);

    // Similar topics should have higher similarity
    try std.testing.expect(sim12 > sim13);
}
