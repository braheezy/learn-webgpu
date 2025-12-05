const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");

pub fn main() !void {
    try zglfw.init();
    defer zglfw.terminate();

    zglfw.windowHint(.client_api, .no_api);
    const window = try zglfw.createWindow(800, 600, "Triangle (zgpu)", null);
    defer zglfw.destroyWindow(window);

    const gfx = try initGraphics(window);
    defer gfx.destroy(std.heap.page_allocator);

    const shader_src =
        \\struct VSOut { @builtin(position) pos: vec4f };
        \\@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
        \\    const positions = array<vec2f, 3>(
        \\        vec2f(-0.6, -0.6),
        \\        vec2f(0.6, -0.6),
        \\        vec2f(0.0, 0.6));
        \\    var out: VSOut;
        \\    out.pos = vec4f(positions[vi], 0.0, 1.0);
        \\    return out;
        \\}
        \\@fragment fn fs_main() -> @location(0) vec4f {
        \\    return vec4f(1.0, 0.4, 0.2, 1.0);
        \\}
    ;

    const shader_module = zgpu.createWgslShaderModule(gfx.device, shader_src, null);
    defer shader_module.release();

    const pipeline = createPipeline(gfx.device, shader_module, gfx.surface_configuration.format);
    defer pipeline.release();

    while (!window.shouldClose()) {
        zglfw.pollEvents();

        const view = gfx.getCurrentTextureView();
        defer view.release();

        const encoder = gfx.device.createCommandEncoder(null);
        defer encoder.release();

        const color_attachment = [_]zgpu.wgpu.RenderPassColorAttachment{.{
            .view = view,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0.1, .g = 0.1, .b = 0.1, .a = 1.0 },
        }};

        const pass = encoder.beginRenderPass(.{
            .color_attachment_count = 1,
            .color_attachments = &color_attachment,
        });

        pass.setPipeline(pipeline);
        pass.draw(3, 1, 0, 0);
        zgpu.endReleasePass(pass);

        const cmd = encoder.finish(null);
        defer cmd.release();
        gfx.queue.submit(&.{cmd});
        _ = gfx.present();
    }
}

fn initGraphics(window: *zglfw.Window) !*zgpu.GraphicsContext {
    return try zgpu.GraphicsContext.create(std.heap.page_allocator, .{
        .window = window,
        .fn_getTime = @ptrCast(&zglfw.getTime),
        .fn_getFramebufferSize = @ptrCast(&zglfw.Window.getFramebufferSize),
        .fn_getWin32Window = @ptrCast(&zglfw.getWin32Window),
        .fn_getX11Display = @ptrCast(&zglfw.getX11Display),
        .fn_getX11Window = @ptrCast(&zglfw.getX11Window),
        .fn_getWaylandDisplay = @ptrCast(&zglfw.getWaylandDisplay),
        .fn_getWaylandSurface = @ptrCast(&zglfw.getWaylandWindow),
        .fn_getCocoaWindow = @ptrCast(&zglfw.getCocoaWindow),
    }, .{});
}

fn createPipeline(device: zgpu.wgpu.Device, shader: zgpu.wgpu.ShaderModule, fmt: zgpu.wgpu.TextureFormat) zgpu.wgpu.RenderPipeline {
    const targets = [_]zgpu.wgpu.ColorTargetState{.{
        .format = fmt,
        .write_mask = zgpu.wgpu.ColorWriteMasks.all,
    }};

    return device.createRenderPipeline(.{
        .vertex = .{
            .module = shader,
            .entry_point = zgpu.wgpu.StringView.fromSlice("vs_main"),
        },
        .fragment = &.{ .module = shader, .entry_point = zgpu.wgpu.StringView.fromSlice("fs_main"), .target_count = targets.len, .targets = &targets },
        .primitive = .{ .topology = .triangle_list },
    });
}
