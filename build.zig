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

    // const zgpu = b.dependency("zgpu", .{});
    // exe.root_module.addImport("zgpu", zgpu.module("root"));

    const zmath = b.dependency("zmath", .{});
    exe.root_module.addImport("zmath", zmath.module("root"));

    const obj_mod = b.dependency("obj", .{ .target = target, .optimize = optimize });
    exe.root_module.addImport("obj", obj_mod.module("obj"));

    const zjpeg = b.dependency("zjpeg", .{});
    exe.root_module.addImport("zjpeg", zjpeg.module("jpeg"));
    exe.root_module.addImport("png", zjpeg.module("png"));

    const wgpu_native_dep = b.dependency("wgpu", .{});
    exe.root_module.addImport("wgpu", wgpu_native_dep.module("wgpu"));

    const zgui = b.dependency("zgui", .{
        .shared = false,
        .backend = .glfw_wgpu,
    });
    exe.root_module.addImport("zgui", zgui.module("root"));
    exe.linkLibrary(zgui.artifact("imgui"));

    if (target.result.os.tag != .emscripten) {
        exe.linkLibrary(zglfw.artifact("glfw"));
        // exe.linkLibrary(zgpu.artifact("zdawn"));
    }

    const options = .{
        .uniforms_buffer_size = b.option(
            u64,
            "uniforms_buffer_size",
            "Set uniforms buffer size",
        ) orelse default_options.uniforms_buffer_size,
        .dawn_skip_validation = b.option(
            bool,
            "dawn_skip_validation",
            "Disable Dawn validation",
        ) orelse default_options.dawn_skip_validation,
        .dawn_allow_unsafe_apis = b.option(
            bool,
            "dawn_allow_unsafe_apis",
            "Allow unsafe WebGPU APIs (e.g. timestamp queries)",
        ) orelse default_options.dawn_allow_unsafe_apis,
        .buffer_pool_size = b.option(
            u32,
            "buffer_pool_size",
            "Set buffer pool size",
        ) orelse default_options.buffer_pool_size,
        .texture_pool_size = b.option(
            u32,
            "texture_pool_size",
            "Set texture pool size",
        ) orelse default_options.texture_pool_size,
        .texture_view_pool_size = b.option(
            u32,
            "texture_view_pool_size",
            "Set texture view pool size",
        ) orelse default_options.texture_view_pool_size,
        .sampler_pool_size = b.option(
            u32,
            "sampler_pool_size",
            "Set sample pool size",
        ) orelse default_options.sampler_pool_size,
        .render_pipeline_pool_size = b.option(
            u32,
            "render_pipeline_pool_size",
            "Set render pipeline pool size",
        ) orelse default_options.render_pipeline_pool_size,
        .compute_pipeline_pool_size = b.option(
            u32,
            "compute_pipeline_pool_size",
            "Set compute pipeline pool size",
        ) orelse default_options.compute_pipeline_pool_size,
        .bind_group_pool_size = b.option(
            u32,
            "bind_group_pool_size",
            "Set bind group pool size",
        ) orelse default_options.bind_group_pool_size,
        .bind_group_layout_pool_size = b.option(
            u32,
            "bind_group_layout_pool_size",
            "Set bind group layout pool size",
        ) orelse default_options.bind_group_layout_pool_size,
        .pipeline_layout_pool_size = b.option(
            u32,
            "pipeline_layout_pool_size",
            "Set pipeline layout pool size",
        ) orelse default_options.pipeline_layout_pool_size,
        .max_num_bindings_per_group = b.option(
            u32,
            "max_num_bindings_per_group",
            "Set maximum number of bindings per bind group",
        ) orelse default_options.max_num_bindings_per_group,
        .max_num_bind_groups_per_pipeline = b.option(
            u32,
            "max_num_bind_groups_per_pipeline",
            "Set maximum number of bindings groups per pipeline",
        ) orelse default_options.max_num_bind_groups_per_pipeline,
    };

    const options_step = b.addOptions();
    inline for (std.meta.fields(@TypeOf(options))) |field| {
        options_step.addOption(field.type, field.name, @field(options, field.name));
    }

    const options_module = options_step.createModule();
    exe.root_module.addImport("zgpu_options", options_module);

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

const default_options = struct {
    const uniforms_buffer_size = 4 * 1024 * 1024;
    const dawn_skip_validation = false;
    const dawn_allow_unsafe_apis = false;
    const buffer_pool_size = 256;
    const texture_pool_size = 256;
    const texture_view_pool_size = 256;
    const sampler_pool_size = 16;
    const render_pipeline_pool_size = 128;
    const compute_pipeline_pool_size = 128;
    const bind_group_pool_size = 32;
    const bind_group_layout_pool_size = 32;
    const pipeline_layout_pool_size = 32;
    const max_num_bindings_per_group = 10;
    const max_num_bind_groups_per_pipeline = 4;
};
