const std = @import("std");

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

    const use_tracy = b.option(
        bool,
        "use_tracy",
        "Enable tracy for profiling",
    ) orelse false;

    const ztracy = b.dependency("ztracy", .{
        .enable_ztracy = true,
        .enable_fibers = true,
        .target = target,
        .optimize = std.builtin.OptimizeMode.ReleaseFast,
    });

    // Expose the flag `use_tracy` in code
    const build_options = b.addOptions();
    build_options.addOption(bool, "use_tracy", use_tracy);
    exe.root_module.addOptions("build_options", build_options);

    if (use_tracy) {
        exe.root_module.addImport("ztracy", ztracy.module("root"));
        exe.linkLibrary(ztracy.artifact("tracy"));
    }

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
