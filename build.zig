const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "learnwebgpu",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(exe);

    const zglfw = b.dependency("zglfw", .{});
    exe.root_module.addImport("zglfw", zglfw.module("root"));

    @import("zgpu").addLibraryPathsTo(exe);

    const zgpu = b.dependency("zgpu", .{});
    exe.root_module.addImport("zgpu", zgpu.module("root"));

    const zmath = b.dependency("zmath", .{});
    exe.root_module.addImport("zmath", zmath.module("root"));

    const obj_mod = b.dependency("obj", .{ .target = target, .optimize = optimize });
    exe.root_module.addImport("obj", obj_mod.module("obj"));

    const zjpeg = b.dependency("zjpeg", .{});
    exe.root_module.addImport("zjpeg", zjpeg.module("jpeg"));

    if (target.result.os.tag != .emscripten) {
        exe.linkLibrary(zglfw.artifact("glfw"));
        exe.linkLibrary(zgpu.artifact("zdawn"));
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
