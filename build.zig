const std = @import("std");
// XXX: Currently we have to add zig-gamedev as a submodule, otherwise, this
// import would fail. And moving this statement into the block `if (use_tracy)`
// won't work either.
const ztracy = @import("third_party/zig-gamedev/libs/ztracy/build.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "run",
        .root_source_file = .{ .path = "run.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    const build_options = b.addOptions();
    exe.addOptions("build_options", build_options);

    const use_tracy = b.option(bool, "use_tracy", "Enable tracy for profiling") orelse false;
    if (use_tracy) {
        const ztracy_pkg = ztracy.package(b, target, optimize, .{
            .options = .{ .enable_ztracy = true },
        });
        ztracy_pkg.link(exe);
    }
    build_options.addOption(bool, "use_tracy", use_tracy);

    const run_exe = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_exe.step);

    const test_step = b.step("test", "Run unit tests");
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "tests.zig" },
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);
}
