const std = @import("std");
const Allocator = std.mem.Allocator;

// Helper function for development
fn printStruct(s: anytype) void {
    inline for (std.meta.fields(@TypeOf(s))) |f| {
        std.debug.print(f.name ++ ": {any}\n", .{@as(f.type, @field(s, f.name))});
    }
}

// For model exported by `legacy_export()` (v0)
// NOTE: We should use `extern struct` as it supports guaranteed layout.
// Otherwise, `std.io.Reader.readStruct()` would fail.
pub const Config = extern struct {
    dim: i32, // transformer dimension (model.params.dim)
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
};

pub const TransformerWeights = struct {
    token_embedding_table: [*]f32, // (vocab_size, dim)
    rms_att_weight: [*]f32, // (layer, dim)
    rms_ffn_weight: [*]f32, // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    wq: [*]f32, // (layer, dim, n_heads * head_size)
    wk: [*]f32, // (layer, dim, n_kv_heads * head_size)
    wv: [*]f32, // (layer, dim, n_kv_heads * head_size)
    wo: [*]f32, // (layer, n_heads * head_size, dim)
    // weights for ffn
    w1: [*]f32, // (layer, hidden_dim, dim)
    w2: [*]f32, // (layer, dim, hidden_dim)
    w3: [*]f32, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: [*]f32, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    wcls: [*]f32,

    // NOTE: Here we follow the way to mmap weights in `llama2.c/runq.c` by
    // taking `*anyopaque` without presuming all weights are f32.
    pub fn init(p: *Config, weights_ptr: *anyopaque, shared_weights: bool) TransformerWeights {
        var w: TransformerWeights = undefined;

        // NOTE: cast i32 to usize to avoid overflow for 13B+ models.
        const dim: usize = @intCast(p.dim);
        const hidden_dim: usize = @intCast(p.hidden_dim);
        const n_layers: usize = @intCast(p.n_layers);
        const n_heads: usize = @intCast(p.n_heads);
        const n_kv_heads: usize = @intCast(p.n_kv_heads);
        const vocab_size: usize = @intCast(p.vocab_size);
        const seq_len: usize = @intCast(p.seq_len);
        const head_size: usize = dim / n_heads;

        var ptr: [*]f32 = @alignCast(@ptrCast(weights_ptr));

        w.token_embedding_table = ptr;
        ptr += vocab_size * dim;
        w.rms_att_weight = ptr;
        ptr += n_layers * dim;
        w.wq = ptr;
        ptr += n_layers * dim * (n_heads * head_size);
        w.wk = ptr;
        ptr += n_layers * dim * (n_kv_heads * head_size);
        w.wv = ptr;
        ptr += n_layers * dim * (n_kv_heads * head_size);
        w.wo = ptr;
        ptr += n_layers * (n_heads * head_size) * dim;
        w.rms_ffn_weight = ptr;
        ptr += n_layers * dim;
        w.w1 = ptr;
        ptr += n_layers * dim * hidden_dim;
        w.w2 = ptr;
        ptr += n_layers * hidden_dim * dim;
        w.w3 = ptr;
        ptr += n_layers * dim * hidden_dim;
        w.rms_final_weight = ptr;
        ptr += dim;
        ptr += seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
        ptr += seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
        w.wcls = if (shared_weights) w.token_embedding_table else ptr;

        return w;
    }
};

const RunState = struct {
    x: []f32, // activation at current time stamp (dim,)
    xb: []f32,
    xb2: []f32,
    hb: []f32, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: []f32,
    q: []f32, // query (dim,)
    // NOTE: we don't need to allocate memory for k, v as we can point them to
    // kv caches.
    // https://github.com/karpathy/llama2.c/blob/b3c4b6c/run.c#L255-L257
    // https://github.com/karpathy/llama2.c/pull/400
    // k: []f32, // key (dim,)
    // v: []f32, // value (dim,)
    att: []f32, // buffer for scores/attention values (n_heads, seq_len)
    logits: []f32, // output logits, distribution of vocabulary (vocab_size)
    key_cache: []f32, // (layer, seq_len, dim)
    value_cache: []f32, // (layer, seq_len, dim)

    pub fn init(p: *const Config, allocator: Allocator) !RunState {
        const dim: usize = @intCast(p.dim);
        const hidden_dim: usize = @intCast(p.hidden_dim);
        const n_layers: usize = @intCast(p.n_layers);
        const n_heads: usize = @intCast(p.n_heads);
        const n_kv_heads: usize = @intCast(p.n_kv_heads);
        const vocab_size: usize = @intCast(p.vocab_size);
        const seq_len: usize = @intCast(p.seq_len);
        const kv_dim: usize = (dim * n_kv_heads) / n_heads;

        // TODO: consider alignment for SIMD?
        // https://github.com/cgbur/llama2.zig/blob/main/src/main.zig#L140C32-L152

        return RunState{
            .x = try allocator.alloc(f32, dim),
            .xb = try allocator.alloc(f32, dim),
            .xb2 = try allocator.alloc(f32, dim),
            .hb = try allocator.alloc(f32, hidden_dim),
            .hb2 = try allocator.alloc(f32, hidden_dim),
            .q = try allocator.alloc(f32, dim),
            .key_cache = try allocator.alloc(f32, n_layers * seq_len * kv_dim),
            .value_cache = try allocator.alloc(f32, n_layers * seq_len * kv_dim),
            .att = try allocator.alloc(f32, n_heads * seq_len),
            .logits = try allocator.alloc(f32, vocab_size),
        };
    }

    pub fn deinit(self: RunState, allocator: Allocator) void {
        allocator.free(self.x);
        allocator.free(self.xb);
        allocator.free(self.xb2);
        allocator.free(self.hb);
        allocator.free(self.hb2);
        allocator.free(self.q);
        allocator.free(self.key_cache);
        allocator.free(self.value_cache);
        allocator.free(self.att);
        allocator.free(self.logits);
    }
};

pub const Transformer = struct {
    config: Config = undefined,
    weights: TransformerWeights = undefined,
    state: RunState = undefined,
    // XXX: In llama2.c, `fd` was kept to be closed manually while program is
    // about to exit, but we can actually close it right after mmap is done.
    fd: std.fs.File = undefined,
    data: *anyopaque = undefined,
    file_size: u64 = undefined,
};

pub fn readCheckpoint(checkpoint: []const u8, transformer: *Transformer) !void {
    const file = try std.fs.cwd().openFile(checkpoint, .{ .mode = .read_only });
    // NOTE: we can close file after `mmap()` is done
    defer file.close();

    var config: *Config = &transformer.config;
    config.* = try file.reader().readStruct(Config);

    // XXX: (llama2.c) negative vocab size -> unshared weights
    const shared_weights: bool = config.vocab_size > 0;
    config.vocab_size = try std.math.absInt(config.vocab_size);
    transformer.file_size = (try file.stat()).size;

    // Reposition to the head of file. Offset of `Config` will be handled later.
    try file.seekTo(0);

    // NOTE: we would delete this memory map later outside this scope.
    const data = try std.os.mmap(
        null,
        transformer.file_size,
        std.os.PROT.READ,
        std.os.MAP.PRIVATE,
        file.handle,
        0,
    );
    transformer.data = @ptrCast(data);

    // View `data` as `void*` from C perspective (`*anyopaque` in zig)
    var weights_ptr: *anyopaque = @ptrCast(data);

    // View `weights_ptr` in byte (u8), and offset it with the size of `Config`.
    // So that we don't need to assume all fields in `Config` are the same type.
    weights_ptr = @as([*]u8, @ptrCast(weights_ptr)) + @sizeOf(Config);

    transformer.weights = TransformerWeights.init(config, weights_ptr, shared_weights);
}

fn buildTransformer(
    transformer: *Transformer,
    checkpoint_path: []u8,
    allocator: Allocator,
) !void {
    try readCheckpoint(checkpoint_path, transformer);
    transformer.state = try RunState.init(&transformer.config, allocator);
}

fn freeTransformer(transformer: *Transformer, allocator: Allocator) void {
    // Cast pointer of mmap data from `*anyopaque` to the original output type
    // `[]align(std.mem.page_size) u8`.
    var mmap_data = @as(
        [*]align(std.mem.page_size) u8,
        @alignCast(@ptrCast(transformer.data)),
    )[0..transformer.file_size];

    // Delete memory map
    std.os.munmap(mmap_data);

    transformer.state.deinit(allocator);
}

pub const TokenIndex = struct {
    str: []const u8,
    id: u32,
};

pub const Tokenizer = struct {
    vocab: [][]u8 = undefined,
    vocab_scores: []f32 = undefined,
    sorted_vocab: ?[]TokenIndex = null,
    vocab_size: i32 = undefined,
    max_token_length: u32 = undefined,
    byte_pieces: [512]u8 = undefined, // stores all single-byte strings

    pub fn init(tokenizer_path: []const u8, vocab_size: i32, allocator: Allocator) !Tokenizer {
        var t = Tokenizer{};

        // NOTE: vocab_size might be written into tokenizer file in the future,
        // then we could change this accordingly.
        t.vocab_size = vocab_size;

        const n_vocab: usize = @intCast(vocab_size);
        t.vocab = try allocator.alloc([]u8, n_vocab);
        t.vocab_scores = try allocator.alloc(f32, n_vocab);

        for (0..256) |i| {
            t.byte_pieces[i * 2] = @intCast(i);
            t.byte_pieces[i * 2 + 1] = '\x00'; // null character: '\0'
        }

        const file = try std.fs.cwd().openFile(tokenizer_path, .{ .mode = .read_only });
        defer file.close();

        var buf_x32: [4]u8 = undefined;
        var buffered_file = std.io.bufferedReader(file.reader());
        // number of bytes read
        var nb_read = try buffered_file.read(&buf_x32);
        if (nb_read != 4) {
            std.debug.print("failed read\n", .{});
            return std.fs.File.ReadError.Unexpected;
        }
        t.max_token_length = std.mem.readIntSliceLittle(u32, &buf_x32);

        // read tokens, lengths of tokens, and scores
        var len: i32 = undefined;
        for (0..n_vocab) |i| {
            // score
            nb_read = try buffered_file.read(&buf_x32);
            if (nb_read != 4) {
                std.debug.print("failed read\n", .{});
                return std.fs.File.ReadError.Unexpected;
            }
            t.vocab_scores[i] = @bitCast(buf_x32);

            // length of token
            nb_read = try buffered_file.read(&buf_x32);
            if (nb_read != 4) {
                std.debug.print("failed read\n", .{});
                return std.fs.File.ReadError.Unexpected;
            }
            len = @bitCast(buf_x32);

            // token
            // NOTE: here we make use of zig's slice since it contains length
            // information of a sequence, so we don't need to append a sentinel
            // ('\x00') to the end of a string. However, if we do need it, we
            // can call `allocator.allocSentinel()` to allocate a buffer which
            // ends with a sentinel while the sentinel char is not counted into
            // `buffer.len` (this is useful for reading data in zig style since
            // the number of bytes to read is determined by length of the buffer).
            t.vocab[i] = try allocator.alloc(u8, @intCast(len));
            nb_read = try buffered_file.read(t.vocab[i]);
            if (nb_read != len) {
                std.debug.print("failed read\n", .{});
                return std.fs.File.ReadError.Unexpected;
            }
        }

        return t;
    }

    pub fn deinit(self: Tokenizer, allocator: Allocator) void {
        for (0..self.vocab.len) |i| {
            allocator.free(self.vocab[i]);
        }
        allocator.free(self.vocab);
        allocator.free(self.vocab_scores);

        if (self.sorted_vocab != null) {
            allocator.free(self.sorted_vocab.?);
        }
    }

    pub fn strLookup(self: Tokenizer, str: []const u8) ?u32 {
        const tok = TokenIndex{ .str = str, .id = undefined };
        // NOTE: `bsearch` in C returns a pointer, this returns an index.
        const res = std.sort.binarySearch(TokenIndex, tok, self.sorted_vocab.?, {}, compareToken2);

        const idx = res orelse return null;
        const tok_id = self.sorted_vocab.?[idx].id;
        return tok_id;
    }

    pub fn encode(
        self: *Tokenizer,
        text: []const u8,
        bos: bool,
        eos: bool,
        tokens: []u32,
        allocator: Allocator,
    ) !u32 {
        if (self.sorted_vocab == null) {
            // lazily initialize the vocabulary
            const n_vocab: usize = @intCast(self.vocab_size);
            self.sorted_vocab = try allocator.alloc(TokenIndex, n_vocab);
            for (0..n_vocab) |i| {
                self.sorted_vocab.?[i] = TokenIndex{
                    .str = self.vocab[i],
                    .id = @intCast(i),
                };
            }

            // sort vocab
            std.sort.pdq(TokenIndex, self.sorted_vocab.?, {}, compareToken);
        }

        // (llama2.c) Temporary buffer to store merge candidates of always two
        // consecutive tokens. *2 for concat, +1 for null terminator, +2 for
        // UTF8 (in case max_token_length is 1).
        var str_buffer = try allocator.alloc(u8, self.max_token_length * 2 + 1 + 2);
        defer allocator.free(str_buffer);

        var str_len: usize = 0;
        var n_tokens: u32 = 0; // retval

        if (bos) {
            tokens[n_tokens] = 1;
            n_tokens += 1;
        }

        // add dummy prefix
        // TODO: need to read through source code of sentencepice to figure out
        // how it work properly.
        if (text.len != 0) {
            const dummy_prefix = self.strLookup(" ").?;
            tokens[n_tokens] = dummy_prefix;
            n_tokens += 1;
        }

        // process the raw (UTF-8) byte sequence of the input string
        for (0..text.len) |i| {
            const c = text[i];

            // Check whether the highest 2 bits are 10 (0b10xxxxxx)
            // mask: 0xC0 (0b11000000)
            if ((c & 0xC0) != 0x80) {
                str_len = 0;
            }

            str_buffer[str_len] = c;
            str_len += 1;
            // NOTE: we don't need to set the last byte to null everytime here,
            // check out the comment related to `strLookup` below.
            // str_buffer[str_len] = '\x00';

            // NOTE: we will peek the next byte in text, so we need to make
            // sure the index won't exceed the length of it. (in llama2.c, this
            // loop checks with null terminator, so it doesn't need to do so)
            if ((i + 1) < text.len and (text[i + 1] & 0xC0) == 0x80 and str_len < 4) {
                continue;
            }

            // NOTE: (IMPORTANT!) since our implementation of `strcmp` checks
            // with length of string instead of the null terminator, we need to
            // pass a `slice` instead of the whole buffer to search.
            const lookup_result = self.strLookup(str_buffer[0..str_len]);
            if (lookup_result != null) {
                tokens[n_tokens] = lookup_result.?;
                n_tokens += 1;
            } else {
                // fallback: encode each byte literally
                for (0..str_len) |j| {
                    // +3: offset for the first 3 vocabs (<unk>, <s>, </s>)
                    tokens[n_tokens] = str_buffer[j] + 3;
                    n_tokens += 1;
                }
            }
            str_len = 0;
        }

        while (true) {
            var best_score: f32 = -std.math.inf(f32);
            var best_id: ?u32 = null;
            var best_idx: ?usize = null;

            for (0..(n_tokens - 1)) |i| {
                const token1 = self.vocab[tokens[i]];
                const token2 = self.vocab[tokens[i + 1]];
                _ = try std.fmt.bufPrint(str_buffer, "{s}{s}", .{ token1, token2 });
                var len = token1.len + token2.len;

                const lookup_result = self.strLookup(str_buffer[0..len]);
                if (lookup_result != null and self.vocab_scores[lookup_result.?] > best_score) {
                    const id = lookup_result.?;
                    best_score = self.vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == null) {
                break; // cannot find any more pairs to merge, so quit this loop
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx.?] = best_id.?;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for ((best_idx.? + 1)..(n_tokens - 1)) |i| {
                tokens[i] = tokens[i + 1];
            }
            n_tokens -= 1;
        }

        if (eos) {
            tokens[n_tokens] = 2;
            n_tokens += 1;
        }

        return n_tokens;
    }

    // XXX: if `self` is not specified as a pointer here, the returned value
    // will be empty (might be a garbage value once input token is 0~255 and it's
    // @constCast() from `byte_pieces` after leaving this scope?)
    pub fn decode(self: *Tokenizer, prev_token: u32, token: u32) []u8 {
        const piece = self.vocab[token];
        var offset: usize = 0;

        // TODO: check whether we can simplify code below since 0~255 would be
        // in the form of "<0x__>", so that we can skip the parsing work once
        // we found the first byte on `piece` is a spce.
        if (prev_token == 1 and piece[0] == ' ') {
            offset += 1;
        }

        // In llama2.c, `piece` is checked with pattern "<0x%02hhX>", and it
        // can be breakdown into:
        // - "<0x": literally matching these characters
        // - "%02hhX": matching a 2-digit number
        //   - "02": 2-digit number, padding with 0 if necessary
        //   - "hh": these 2-digit number are 2-byte variable
        //   - "X": interprete this 2-digit number as a hexadecimal number
        // - ">": literally matching it
        var voc = piece[offset..]; // TODO: try to avoid this, this create a new piece of memory
        if (voc.len == 6 and voc[0] == '<' and voc[5] == '>') {
            const byte_val: u8 = std.fmt.parseUnsigned(u8, voc[1..5], 0) catch |err| switch (err) {
                else => {
                    std.log.err("Failed to parse vocen, id: {d}\n", .{voc});
                    return voc;
                },
            };
            voc = @constCast(self.byte_pieces[byte_val][0..]);
            // std.debug.print("decode - byte_val: {u}\n", .{byte_val});
            // std.debug.print("decode - voc: {s}\n", .{voc});
        }

        return voc;
    }
};

// Compare strings like how `strcmp` works in C. Note that this implementation
// does not rely on null terminator, but it relies on how `slice` works in zig
// as it provides length infomation of a sequence.
pub fn strcmp(a: []const u8, b: []const u8) i32 {
    var i: usize = 0;
    while (i < a.len and i < b.len) {
        if (a[i] != b[i]) {
            return @as(i32, a[i]) - @as(i32, b[i]);
        }
        i += 1;
    }
    // Now, we ran out of characters from either a or b. So we just need to
    // check with the lengths of them.
    const len_a: i32 = @intCast(a.len);
    const len_b: i32 = @intCast(b.len);
    return len_a - len_b;
}

// Compare 2 `TokenIndex`s. True: a < b; False: a >= b.
pub fn compareToken(context: void, a: TokenIndex, b: TokenIndex) bool {
    _ = context;
    return strcmp(a.str, b.str) < 0;
}

// Compare 2 `TokenIndex`s and return `math.Order`.
pub fn compareToken2(context: void, a: TokenIndex, b: TokenIndex) std.math.Order {
    _ = context;
    const res = strcmp(a.str, b.str);
    if (res < 0) {
        return std.math.Order.lt;
    } else if (res == 0) {
        return std.math.Order.eq;
    } else {
        return std.math.Order.gt;
    }
}

pub fn buildTokenizer(
    t: *Tokenizer,
    tokenizer_path: []const u8,
    vocab_size: i32,
    allocator: Allocator,
) !void {
    t.* = try Tokenizer.init(tokenizer_path, vocab_size, allocator);
}

pub fn freeTokenizer(tokenizer: *Tokenizer, allocator: Allocator) void {
    tokenizer.deinit(allocator);
}

pub const ProbIndex = struct {
    prob: f32,
    index: i32,
};

pub const Sampler = struct {
    vocab_size: i32,
    probindex: []ProbIndex,
    temperature: f32,
    topp: f32,
    rng_state: u64,

    pub fn init(
        vocab_size: i32,
        temperature: f32,
        topp: f32,
        rng_seed: u64,
        allocator: Allocator,
    ) !Sampler {
        const n_vocab: usize = @intCast(vocab_size);

        return Sampler{
            .vocab_size = vocab_size,
            .temperature = temperature,
            .topp = topp,
            .rng_state = rng_seed,
            .probindex = try allocator.alloc(ProbIndex, n_vocab),
        };
    }

    pub fn deinit(self: Sampler, allocator: Allocator) void {
        allocator.free(self.probindex);
    }
};

fn errorUsage() void {
    const msg =
        \\ Usage:   run <checkpoint> [options]
        \\ Example: run model.bin -n 256 -i \"Once upon a time\"
        \\ Options:
        \\   -t <float>  temperature in [0,inf], default i.0
        \\   -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9
        \\   -s <int>    random seed, default time(NULL)
        \\   -n <int>    number of steps to run for, default 256. 0 = max_seq_len
        \\   -i <string> input prompt
        \\   -z <string> optional path to custom tokenizer
        \\   -m <string> mode: generate|chat, default: generate
        \\   -y <string> (optional) system prompt in chat mode
    ;
    std.debug.print("{s}\n", .{msg});
    std.process.exit(1);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("No model checkpoint is specified\n", .{});
        errorUsage();
    }

    const checkpoint_path = args[1];
    var tokenizer_path: []const u8 = "tokenizer.bin";
    var temperature: f32 = 1.0;
    var topp: f32 = 0.9;
    var steps: u32 = 256;
    var prompt: []const u8 = "";
    var rng_seed: u64 = 0;
    var mode: []const u8 = "generate";
    var system_prompt: []const u8 = "";

    var i: usize = 2;
    while (i < args.len) : (i += 2) {
        if (i + 1 > args.len) {
            errorUsage();
        }

        const arg = args[i];
        const val = args[i + 1];
        if (arg[0] != '-' or arg.len != 2) {
            errorUsage();
        }

        if (arg[1] == 't') {
            temperature = try std.fmt.parseFloat(f32, val);
        } else if (arg[1] == 'p') {
            topp = try std.fmt.parseFloat(f32, val);
        } else if (arg[1] == 's') {
            rng_seed = try std.fmt.parseUnsigned(u64, val, 10);
        } else if (arg[1] == 'n') {
            steps = try std.fmt.parseUnsigned(u32, val, 10);
        } else if (arg[1] == 'i') {
            prompt = val;
        } else if (arg[1] == 'z') {
            tokenizer_path = val;
        } else if (arg[1] == 'm') {
            mode = val;
        } else if (arg[1] == 'y') {
            system_prompt = val;
        } else {
            errorUsage();
        }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) {
        rng_seed = @intCast(std.time.timestamp());
    }
    if (temperature < 0.0) {
        temperature = 0.0;
    }
    if (topp < 0.0 or 1.0 < topp) {
        topp = 0.9;
    }
    if (steps < 0) {
        steps = 0;
    }

    var transformer = Transformer{};
    try buildTransformer(&transformer, checkpoint_path, allocator);
    defer freeTransformer(&transformer, allocator);

    // Build tokenizer
    var tokenizer = Tokenizer{};
    try buildTokenizer(&tokenizer, tokenizer_path, 32000, allocator);
    defer freeTokenizer(&tokenizer, allocator);

    // Build sampler
    var sampler = try Sampler.init(32000, temperature, topp, rng_seed, allocator);
    defer sampler.deinit(allocator);
}
