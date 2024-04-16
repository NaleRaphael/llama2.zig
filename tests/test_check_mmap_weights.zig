const std = @import("std");
const expect = std.testing.expect;
const utils = @import("./utils.zig");

const num_samples = 4;
const WeightSamples = extern struct {
    token_embedding_table: [num_samples]f32,
    rms_att_weight: [num_samples]f32,
    wq: [num_samples]f32,
    wk: [num_samples]f32,
    wv: [num_samples]f32,
    wo: [num_samples]f32,
    rms_ffn_weight: [num_samples]f32,
    w1: [num_samples]f32,
    w2: [num_samples]f32,
    w3: [num_samples]f32,
    rms_final_weight: [num_samples]f32,
    // wcls: [num_samples]f32,
};

test "check mmap weights" {
    const mod = @import("../run.zig");

    const allocator = std.testing.allocator;
    const fn_output: []const u8 = "tests/weight_samples.bin";
    const argv = [_][]const u8{
        "python",
        "tests/get_weights_for_check.py",
        "models/stories15M.pt",
        "--fn_output",
        fn_output,
    };

    // TODO: figure out how to make this work when `cwd` is specified
    const proc = try std.ChildProcess.exec(.{
        .allocator = allocator,
        .argv = &argv,
    });

    std.debug.print("{s}\n", .{proc.stdout});
    std.debug.print("{s}\n", .{proc.stderr});
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);

    try std.testing.expectEqual(proc.term, std.ChildProcess.Term{ .Exited = 0 });

    const f_samples = try std.fs.cwd().openFile(fn_output, .{ .mode = .read_only });
    defer f_samples.close();

    var ws: WeightSamples = try f_samples.reader().readStruct(WeightSamples);

    const checkpoint_path: []const u8 = "models/stories15M.bin";
    var transformer = mod.Transformer{};

    try mod.readCheckpoint(checkpoint_path, &transformer, true, allocator);

    const tw = transformer.weights;
    try utils.assertArrayEqual(f32, tw.token_embedding_table[0..4], ws.token_embedding_table[0..4]);
    try utils.assertArrayEqual(f32, tw.rms_att_weight[0..4], ws.rms_att_weight[0..4]);
    try utils.assertArrayEqual(f32, tw.wq[0..4], ws.wq[0..4]);
    try utils.assertArrayEqual(f32, tw.wk[0..4], ws.wk[0..4]);
    try utils.assertArrayEqual(f32, tw.wv[0..4], ws.wv[0..4]);
    try utils.assertArrayEqual(f32, tw.wo[0..4], ws.wo[0..4]);
    try utils.assertArrayEqual(f32, tw.rms_ffn_weight[0..4], ws.rms_ffn_weight[0..4]);
    try utils.assertArrayEqual(f32, tw.w1[0..4], ws.w1[0..4]);
    try utils.assertArrayEqual(f32, tw.w2[0..4], ws.w2[0..4]);
    try utils.assertArrayEqual(f32, tw.w3[0..4], ws.w3[0..4]);
}
