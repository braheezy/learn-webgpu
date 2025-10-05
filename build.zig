const std = @import("std");

const zgpu_backend = .wgpu;
// zgui depends on imgui, which depends on specific wgpu functions, so it needs to work with zgpu first.
var enable_gui = false;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "learnwebgpu",
        .root_module = root_mod,
    });

    b.installArtifact(exe);

    const options = b.addOptions();
    options.addOption(bool, "gui", false);
    const options_mod = options.createModule();
    exe.root_module.addImport("config", options_mod);

    const zglfw = b.dependency("zglfw", .{});
    exe.root_module.addImport("zglfw", zglfw.module("root"));

    const zgpu = b.dependency("zgpu", .{
        .webgpu_backend = zgpu_backend,
    });
    exe.root_module.addImport("zgpu", zgpu.module("root"));

    const zmath = b.dependency("zmath", .{});
    exe.root_module.addImport("zmath", zmath.module("root"));

    const obj_mod = b.dependency("obj", .{ .target = target, .optimize = optimize });
    exe.root_module.addImport("obj", obj_mod.module("obj"));

    const zpix = b.dependency("zpix", .{});
    exe.root_module.addImport("zjpeg", zpix.module("jpeg"));
    exe.root_module.addImport("png", zpix.module("png"));

    if (enable_gui) {
        const zgui = b.dependency("zgui", .{
            .shared = false,
            .backend = .glfw_wgpu,
        });
        exe.root_module.addImport("zgui", zgui.module("root"));
        exe.linkLibrary(zgui.artifact("imgui"));
    }
    if (target.result.os.tag != .emscripten) {
        exe.linkLibrary(zglfw.artifact("glfw"));
        if (zgpu_backend == .dawn) {
            exe.linkLibrary(zgpu.artifact("zdawn"));
        } else if (zgpu_backend == .wgpu) {
            exe.linkLibrary(zgpu.artifact("zwgpu"));
        }
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_module = root_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
