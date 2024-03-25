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
    str: *const [:0]u8,
    id: i32,
};

pub const Tokenizer = struct {
    vocab: [][:0]u8 = undefined,
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
        t.vocab = try allocator.alloc([:0]u8, n_vocab);
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
            t.vocab[i] = try allocator.allocSentinel(u8, @intCast(len), '\x00');
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

    pub fn encode(
        self: *Tokenizer,
        text: []const u8,
        bos: u8,
        eos: u8,
        tokens: *[]i32,
        allocator: Allocator,
    ) !u32 {
        _ = tokens;
        _ = eos;
        _ = bos;
        _ = text;

        if (self.sorted_vocab == null) {
            // lazily initialize the vocabulary
            const n_vocab: usize = @intCast(self.vocab_size);
            self.sorted_vocab = try allocator.alloc(TokenIndex, n_vocab);
            for (0..n_vocab) |i| {
                self.sorted_vocab.?[i] = TokenIndex{
                    .str = &self.vocab[i],
                    .id = @intCast(i),
                };
            }

            // sort vocab
            std.sort.pdq(TokenIndex, self.sorted_vocab.?, {}, compareToken);
        }

        return 0;
    }
};

// Compare string like how `strcmp` in C works. Note that inputs should be null
// terminated.
pub fn strcmp(a: [*]const u8, b: [*]const u8) bool {
    var i: usize = 0;
    while (a[i] != 0 and a[i] == b[i]) {
        i += 1;
    }
    return a[i] < b[i];
}

pub fn compareToken(context: void, a: TokenIndex, b: TokenIndex) bool {
    _ = context;
    return strcmp(a.str.ptr, b.str.ptr);
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
    var prompt: []u8 = "";
    var rng_seed: u64 = 0;
    var mode: []u8 = "";
    var system_prompt: []u8 = "";

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
