const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const VEC_SIZE_F32 = std.simd.suggestVectorSize(f32) orelse 4;

// XXX: Because of the limitation of build system in zig v0.11, we cannot
// switch between `tracy_full` and `tracy_stub` by passing compilation flags.
// So we have to do this kind of "conditional import". See also section
// "conditional compilation" in "docs/ISSUES.md".
const use_tracy = @import("build_options").use_tracy;
const ztracy = if (use_tracy) @import("ztracy");

const tracy_wrapper_stub = struct {
    pub inline fn startZone(
        _: std.builtin.SourceLocation,
        _: [*:0]const u8,
        _: u64,
    ) void {}

    pub inline fn endZone(_: *const anyopaque) void {}
};

const tracy_wrapper_full = struct {
    pub inline fn startZone(
        src_loc: std.builtin.SourceLocation,
        name: [*:0]const u8,
        color: u64,
    ) ztracy.ZoneCtx {
        const zone = if (use_tracy) ztracy.ZoneNC(src_loc, name, color);
        return zone;
    }

    pub inline fn endZone(zone: *const anyopaque) void {
        if (use_tracy) @as(*ztracy.ZoneCtx, @constCast(@alignCast(@ptrCast(zone)))).End();
    }
};

const TracyWrapper = if (use_tracy) tracy_wrapper_full else tracy_wrapper_stub;

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
    k: []f32 = undefined, // key (dim,)
    v: []f32 = undefined, // value (dim,)
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

// ----------------------------------------------------------------------
pub const Transformer = struct {
    config: Config = undefined,
    weights: TransformerWeights = undefined,
    state: RunState = undefined,
    // XXX: In llama2.c, `fd` was kept to be closed manually while program is
    // about to exit, but we can actually close it right after mmap is done.
    fd: std.fs.File = undefined,
    data: *anyopaque = undefined,
    file_size: u64 = undefined,

    pub fn forward(self: *Transformer, token: u32, pos: u32) []f32 {
        const p = self.config;
        const w = self.weights;
        var s = self.state;
        var x = s.x;
        const dim: usize = @intCast(p.dim);
        const hidden_dim: usize = @intCast(p.hidden_dim);
        const n_layers: usize = @intCast(p.n_layers);
        const n_heads: usize = @intCast(p.n_heads);
        const n_kv_heads: usize = @intCast(p.n_kv_heads);
        const vocab_size: usize = @intCast(p.vocab_size);
        const seq_len: usize = @intCast(p.seq_len);
        const kv_dim: usize = (dim * n_kv_heads) / n_heads;
        const kv_mul: usize = n_heads / n_kv_heads; // integer multiplier of the kv sharing in multiquery
        const head_size: usize = dim / n_heads;

        const content_row = w.token_embedding_table[(dim * token)..(dim * (token + 1))];
        @memcpy(x, content_row);

        // forward all the layers
        for (0..n_layers) |l| {
            // attention rmsnorm
            rmsnorm(s.xb, x, w.rms_att_weight[l * dim .. (l + 1) * dim]);

            // key and value point to the kv cache
            const loff = l * seq_len * kv_dim;
            s.k = s.key_cache[(loff + pos * kv_dim)..(loff + (pos + 1) * kv_dim)];
            s.v = s.value_cache[(loff + pos * kv_dim)..(loff + (pos + 1) * kv_dim)];

            // op: `xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)`
            // src: Attention.forward()
            matmul(s.q, s.xb, w.wq[l * dim * dim .. (l + 1) * dim * dim], dim, dim);
            matmul(s.k, s.xb, w.wk[l * dim * kv_dim .. (l + 1) * dim * kv_dim], dim, kv_dim);
            matmul(s.v, s.xb, w.wv[l * dim * kv_dim .. (l + 1) * dim * kv_dim], dim, kv_dim);

            // RoPE relative positional encoding
            var j: usize = 0;
            while (j < dim) : (j += 2) {
                const head_dim: f32 = @floatFromInt(j % head_size);
                const freq: f32 = 1.0 / std.math.pow(f32, 10000.0, head_dim / @as(f32, @floatFromInt(head_size)));
                const val: f32 = @as(f32, @floatFromInt(pos)) * freq;
                const fcr = std.math.cos(val);
                const fci = std.math.sin(val);
                const rotn: usize = if (j < kv_dim) 2 else 1; // how many vectors? 2 = q & k, 1 = q only
                for (0..rotn) |v| {
                    const vec = if (v == 0) s.q else s.k;
                    const v0 = vec[j];
                    const v1 = vec[j + 1];
                    vec[j] = v0 * fcr - v1 * fci;
                    vec[j + 1] = v0 * fci + v1 * fcr;
                }
            }

            // multihead attention. iterate over all heads
            for (0..n_heads) |h| {
                // get the query vector for this head
                const q = s.q[h * head_size .. (h + 1) * head_size];
                // attention scores for this head
                const att = s.att[h * seq_len .. (h + 1) * seq_len];
                // iterate over all timesteps, including the current one
                for (0..pos + 1) |t| {
                    const il: usize = loff + t * kv_dim + (h / kv_mul) * head_size;
                    const ir = il + head_size;
                    const k = s.key_cache[il..ir];
                    var score: f32 = 0.0;
                    for (0..head_size) |i| {
                        score += q[i] * k[i];
                    }
                    score /= std.math.sqrt(@as(f32, @floatFromInt(head_size)));
                    att[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                // NOTE: in `Attention.forward()::model.py`, this works with a mask of
                // upper triangular matrix filling with -inf.
                softmax(att[0 .. pos + 1]);

                // weighted sum of the values, store back into xb
                var xb = s.xb[h * head_size .. (h + 1) * head_size];
                @memset(xb, 0.0);
                for (0..pos + 1) |t| {
                    const il: usize = loff + t * kv_dim + (h / kv_mul) * head_size;
                    const ir = il + head_size;
                    const v = s.value_cache[il..ir];
                    const a = att[t];
                    for (0..head_size) |i| {
                        xb[i] += a * v[i];
                    }
                }
            }

            // final matmul to get the output of the attention
            // op: `output = self.wo(output)`
            // src: Attention.forward()
            matmul(s.xb2, s.xb, w.wo[l * dim * dim .. (l + 1) * dim * dim], dim, dim);

            // residual connection back into x
            // op: `h = x + self.attention.forward(...)`
            // src: TransformerBlock.forward()
            for (0..dim) |i| {
                x[i] += s.xb2[i];
            }

            // ffn rmsnorm
            rmsnorm(s.xb, x, w.rms_ffn_weight[l * dim .. (l + 1) * dim]);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            matmul(s.hb, s.xb, w.w1[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim], dim, hidden_dim);
            matmul(s.hb2, s.xb, w.w3[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim], dim, hidden_dim);

            // SwiGLU non-linearity
            for (0..hidden_dim) |i| {
                var val: f32 = s.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (1.0 / (1.0 + std.math.exp(-val)));
                // elementwise multiply with w3(x)
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // final matmul to get the output of the ffn
            matmul(s.xb, s.hb, w.w2[l * dim * hidden_dim .. (l + 1) * dim * hidden_dim], hidden_dim, dim);

            // residual connection
            for (0..dim) |i| {
                x[i] += s.xb[i];
            }
        }

        // final rmsnorm
        rmsnorm(x, x, w.rms_final_weight[0..dim]);

        // classifier into logits
        matmul(s.logits, x, w.wcls[0 .. dim * vocab_size], dim, vocab_size);
        return s.logits;
    }
};

pub fn rmsnorm(o: []f32, x: []f32, weight: []f32) void {
    assert(o.len == x.len);
    assert(o.len == weight.len);

    const size = o.len;
    var ss: f32 = 0.0;
    // calculate sum of sqaures
    for (0..size) |j| {
        ss += x[j] * x[j];
    }
    ss /= @as(f32, @floatFromInt(size));
    ss += 1e-5;
    ss = 1.0 / std.math.sqrt(ss);
    // normalize and scale
    for (0..size) |j| {
        o[j] = weight[j] * (ss * x[j]);
    }
}

pub fn softmax(x: []f32) void {
    const size = x.len;
    // find max value (for numerical stability)
    var max_val = x[0];
    for (1..size) |i| {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    var sum: f32 = 0.0;
    for (0..size) |i| {
        x[i] = std.math.exp(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (0..size) |i| {
        x[i] /= sum;
    }
}

/// Matrix multiplication: W (d,n) @ x (n,) -> xout (d,)
pub fn matmul(xout: []f32, x: []f32, w: []f32, n: usize, d: usize) void {
    const zone = TracyWrapper.startZone(@src(), "matmul", 0x00_00_ff_00);
    defer TracyWrapper.endZone(&zone);

    // matmul_naive(xout, x, w, n, d);
    matmul_simd(xout, x, w, n, d);
}

fn matmul_naive(xout: []f32, x: []f32, w: []f32, n: usize, d: usize) void {
    for (0..d) |i| {
        var val: f32 = 0.0;
        for (0..n) |j| {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

fn matmul_simd(xout: []f32, x: []f32, w: []f32, n: usize, d: usize) void {
    const vec_sz = VEC_SIZE_F32;
    const n_vec: usize = n / vec_sz;
    const n_rem: usize = n % vec_sz;

    for (0..d) |i| {
        var val: f32 = 0.0;
        const offset: usize = i * n;
        var vsum: @Vector(vec_sz, f32) = @splat(0.0);

        for (0..n_vec) |nv| {
            // NOTE: SIMD vector requires a known size at compile time, so we
            // need to access slice like this.
            const vx: @Vector(vec_sz, f32) = x[nv * vec_sz ..][0..vec_sz].*;
            const vw: @Vector(vec_sz, f32) = w[offset + nv * vec_sz ..][0..vec_sz].*;
            vsum += vx * vw;
        }
        val = @reduce(.Add, vsum);

        // Process remaining elements
        const offset2: usize = vec_sz * n_vec;
        for (0..n_rem) |j| {
            val += w[offset + offset2 + j] * x[offset2 + j];
        }

        xout[i] = val;
    }
}

/// Read checkpoint and initialize transformer. Note that user is responsible to
/// call `freeTransformer()` to delete the memory mapping.
pub fn readCheckpoint(
    checkpoint: []const u8,
    transformer: *Transformer,
    use_mmap: bool,
    allocator: Allocator,
) !void {
    const file = try std.fs.cwd().openFile(checkpoint, .{ .mode = .read_only });
    // NOTE: we can close file after `mmap()` call has returned
    defer file.close();

    var config: *Config = &transformer.config;
    config.* = try file.reader().readStruct(Config);

    // XXX: (llama2.c) negative vocab size -> unshared weights
    const shared_weights: bool = config.vocab_size > 0;
    config.vocab_size = try std.math.absInt(config.vocab_size);
    transformer.file_size = (try file.stat()).size;

    // Reposition to the head of file. Offset of `Config` will be handled later.
    try file.seekTo(0);

    var data: []align(std.mem.page_size) u8 = undefined;
    if (use_mmap) {
        data = try std.os.mmap(
            null,
            transformer.file_size,
            std.os.PROT.READ,
            std.os.MAP.PRIVATE,
            file.handle,
            0,
        );
        transformer.data = @ptrCast(data);
    } else {
        data = blk: {
            const buffer = try allocator.alignedAlloc(u8, std.mem.page_size, transformer.file_size);
            const read_len = try file.readAll(buffer);
            if (read_len != transformer.file_size) {
                std.debug.print("error: failed to read checkpoint file\n", .{});
                return std.os.ReadError.OperationAborted;
            }
            break :blk buffer;
        };
    }

    // View `data` as `void*` from C perspective (`*anyopaque` in zig)
    var weights_ptr: *anyopaque = @ptrCast(data);

    // View `weights_ptr` in byte (u8), and offset it with the size of `Config`.
    // So that we don't need to assume all fields in `Config` are the same type.
    weights_ptr = @as([*]u8, @ptrCast(weights_ptr)) + @sizeOf(Config);

    transformer.weights = TransformerWeights.init(config, weights_ptr, shared_weights);
}

fn buildTransformer(
    transformer: *Transformer,
    checkpoint_path: []const u8,
    use_mmap: bool,
    allocator: Allocator,
) !void {
    try readCheckpoint(checkpoint_path, transformer, use_mmap, allocator);
    transformer.state = try RunState.init(&transformer.config, allocator);
}

fn freeTransformer(transformer: *Transformer, use_mmap: bool, allocator: Allocator) void {
    // Cast pointer of mmap data from `*anyopaque` to the original output type
    // `[]align(std.mem.page_size) u8`.
    const data = @as(
        [*]align(std.mem.page_size) u8,
        @alignCast(@ptrCast(transformer.data)),
    )[0..transformer.file_size];

    if (use_mmap) {
        // Delete memory mapping
        std.os.munmap(data);
    } else {
        allocator.free(data);
    }

    transformer.state.deinit(allocator);
}

// ----------------------------------------------------------------------
pub const TokenIndex = struct {
    str: []const u8,
    id: u32,

    /// Comparator. True: a < b.
    pub fn desc(_: void, a: TokenIndex, b: TokenIndex) bool {
        return strcmp(a.str, b.str) < 0;
    }
};

pub const Tokenizer = struct {
    vocab: [][]u8 = undefined,
    vocab_scores: []f32 = undefined,
    sorted_vocab: ?[]TokenIndex = null,
    vocab_size: i32 = undefined,
    max_token_length: u32 = undefined,
    byte_pieces: [256]u8 = undefined, // stores all single-byte strings

    pub fn init(tokenizer_path: []const u8, vocab_size: i32, allocator: Allocator) !Tokenizer {
        var t = Tokenizer{};

        // NOTE: vocab_size might be written into tokenizer file in the future,
        // then we could change this accordingly.
        t.vocab_size = vocab_size;

        const n_vocab: usize = @intCast(vocab_size);
        t.vocab = try allocator.alloc([]u8, n_vocab);
        t.vocab_scores = try allocator.alloc(f32, n_vocab);

        // NOTE: every element in `byte_pieces` will be used as a slice with
        // length 1, so that we don't need to append a null terminator to it.
        for (0..256) |i| {
            t.byte_pieces[i] = @intCast(i);
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
        const res = std.sort.binarySearch(TokenIndex, tok, self.sorted_vocab.?, {}, compareToken);

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
        // XXX: we need to update member in Tokenizer here, that's why the first
        // parameter of this function should be a pointer. (not sure what's the
        // conventional way to do this)
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
            std.sort.pdq(TokenIndex, self.sorted_vocab.?, {}, TokenIndex.desc);
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
    // would be gibberish.
    pub fn decode(self: *Tokenizer, prev_token: u32, token: u32) []u8 {
        var piece: []u8 = self.vocab[token];

        // NOTE: (llama2.c) following BOS token, sentencepiece decoder strips
        // any leading whitespace.
        if (prev_token == 1 and piece[0] == ' ') {
            piece = piece[1..];
        }

        // In llama2.c, `piece` is checked with pattern "<0x%02hhX>", and it
        // can be breakdown into:
        // - "<0x": literally matching these characters
        // - "%02hhX": matching a 2-digit number
        //   - "02": 2-digit number, padding with 0 if necessary
        //   - "hh": these 2-digit number are 2-byte variable
        //   - "X": interprete this 2-digit number as a hexadecimal number
        // - ">": literally matching it
        if (piece.len == 6 and piece[0] == '<' and piece[5] == '>') {
            const byte_val: u8 = std.fmt.parseUnsigned(u8, piece[1..5], 0) catch |err| switch (err) {
                else => {
                    std.log.err("Failed to parse token, id: {d}\n", .{token});
                    return piece;
                },
            };

            // NOTE: type coercion explanation (`...` denotes the former item)
            // 1. `self.byte_pieces[byte_val]`: u8
            // 2. `&...`: *u8 (a single-item pointer to u8)
            // 3. `@as(*[1]u8, ...)`: *[1]u8 (a pointer to a u8 array with length 1)
            // 4. `piece = ...`: []u8 (a slice of u8)
            //
            // In 3., if we try to directly cast type to `[]u8`, compiler will
            // complain "error: expected type '[]u8', found '*u8'", because
            // compiler doesn't know the length of it.
            // In 4., it works because slice is a fat pointer (ptr + len), and
            // `*[1]u8` is a pointer with length info, so type coercion is valid.
            piece = @as(*[1]u8, &self.byte_pieces[byte_val]);
        }
        return piece;
    }
};

/// Compare strings like how `strcmp` works in C. Note that this implementation
/// does not rely on null terminator, but it relies on how `slice` works in zig
/// as it provides length infomation of a sequence.
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

/// Compare 2 `TokenIndex`s and return `math.Order`.
pub fn compareToken(context: void, a: TokenIndex, b: TokenIndex) std.math.Order {
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

pub fn safePrint(piece: []const u8) void {
    if (piece.len == 1) {
        if (piece[0] == '\x00') return;
        const byte_val: u8 = piece[0];
        if (!(std.ascii.isPrint(byte_val) or std.ascii.isWhitespace(byte_val))) {
            std.log.warn("Found non-printable input, len: {d}\n", .{piece.len});
            return;
        }
    }
    std.debug.print("{s}", .{piece});
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

// ----------------------------------------------------------------------
pub const ProbIndex = struct {
    prob: f32,
    index: usize,

    /// Comparator. True: a > b.
    pub fn asc(_: void, a: ProbIndex, b: ProbIndex) bool {
        return a.prob > b.prob;
    }
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

    pub fn sample(self: *Sampler, logits: []f32) u32 {
        // sample the token given the logits and some hyperparameters
        var next: usize = 0;
        if (self.temperature == 0.0) {
            // greedy argmax sampling: take the token with the highest probability
            next = sampleArgmax(logits);
        } else {
            // apply the temperature to the logits
            const n_vocab: usize = @intCast(self.vocab_size);
            for (0..n_vocab) |q| {
                logits[q] /= self.temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            softmax(logits);
            // flip a (float) coin (this is our source of entropy for sampling)
            const coin = randomF32(&self.rng_state);
            // we sample from this distribution to get the next token
            if (self.topp <= 0 or self.topp >= 1) {
                // simply sample from the predicted probability distribution
                next = sampleMult(logits, coin);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = sampleTopp(logits, self.topp, self.probindex, coin);
            }
        }
        return @as(u32, @intCast(next));
    }
};

// TODO: should we change the output type to u32? (other sampling functions
// below should be changed too)
pub fn sampleArgmax(probabilities: []f32) usize {
    // return the index that has the highest probability
    var max_i: usize = 0;
    var max_p: f32 = probabilities[0];
    for (1..probabilities.len) |i| {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

pub fn sampleMult(probabilities: []f32, coin: f32) usize {
    var cdf: f32 = 0.0;
    for (0..probabilities.len) |i| {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return probabilities.len - 1; // in case of rounding errors
}

pub fn sampleTopp(probabilities: []f32, topp: f32, probindex: []ProbIndex, coin: f32) usize {
    var n0: usize = 0;

    // filter out probs < (1 - topp) / (n - 1) before sorting
    const cutoff: f32 = (1.0 - topp) / @as(f32, @floatFromInt(probabilities.len - 1));
    for (0..probabilities.len) |i| {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0 += 1;
        }
    }
    std.sort.pdq(ProbIndex, probindex[0..n0], {}, ProbIndex.asc);

    // truncate the list where cumulative probability exceeds topp
    var cumulative_prob: f32 = 0.0;
    var last_idx = n0 - 1;
    for (0..n0) |i| {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // note that last index is included now
        }
    }

    // sample from the truncated list
    const r = coin * cumulative_prob;
    var cdf: f32 = 0.0;
    for (0..(last_idx + 1)) |i| {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

pub fn randomU32(state: *u64) u32 {
    state.* ^= state.* >> 12;
    state.* ^= state.* << 25;
    state.* ^= state.* >> 27;
    return @as(u32, @intCast((state.* *% @as(u64, 0x2545F4914F6CDD1D)) >> 32));
}

pub fn randomF32(state: *u64) f32 {
    // 16777216 = 2^24 = "0 10010111 00000000000000000000000"
    // sign: 0, exponent: 10010111 (-127 + 151 = 24), mantissa: 0
    const magic: f32 = 16777216.0;
    return @as(f32, @floatFromInt(randomU32(state) >> 8)) / magic;
}

// ----------------------------------------------------------------------
pub fn generate(
    transformer: *Transformer,
    tokenizer: *Tokenizer,
    sampler: *Sampler,
    prompt: []const u8,
    steps: u32,
    allocator: Allocator,
) !void {
    var prompt_tokens: []u32 = try allocator.alloc(u32, prompt.len + 3);
    defer allocator.free(prompt_tokens);

    const n_tokens = try tokenizer.encode(prompt, true, false, prompt_tokens, allocator);

    var start: i64 = 0;
    var next: u32 = undefined;
    var token = prompt_tokens[0];
    var pos: u32 = 0;

    while (pos < steps) {
        // forward the transformer to get logits for the next token
        var logits: []f32 = transformer.forward(token, pos);

        if (pos < n_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sampler.sample(logits);
        }
        pos += 1;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) {
            break;
        }

        const piece = tokenizer.decode(token, next);
        safePrint(piece);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = std.time.milliTimestamp();
        }
    }
    std.debug.print("\n", .{});

    if (pos > 1) {
        const end: i64 = std.time.milliTimestamp();
        const tok_per_sec: f32 = @as(f32, @floatFromInt(pos - 1)) / @as(f32, @floatFromInt((end - start))) * 1000;
        std.debug.print("achieved tok/s: {d}\n", .{tok_per_sec});
    }
}

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
        \\   -l <int>   (optional) use mmap for checkpoint (0: disable, 1: enable)
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
    var use_mmap: bool = true;

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
        } else if (arg[1] == 'l') {
            const tmp = try std.fmt.parseInt(u1, val, 0);
            use_mmap = if (tmp == 1) true else false;
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
    try buildTransformer(&transformer, checkpoint_path, use_mmap, allocator);
    defer freeTransformer(&transformer, use_mmap, allocator);

    // Build tokenizer
    var tokenizer = Tokenizer{};
    try buildTokenizer(&tokenizer, tokenizer_path, 32000, allocator);
    defer freeTokenizer(&tokenizer, allocator);

    // Build sampler
    var sampler = try Sampler.init(32000, temperature, topp, rng_seed, allocator);
    defer sampler.deinit(allocator);

    try generate(&transformer, &tokenizer, &sampler, prompt, steps, allocator);
}
