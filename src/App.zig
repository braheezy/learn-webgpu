const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");

const Context = struct {
    ready: bool,
    buffer: zgpu.wgpu.Buffer,
};

const shader_source =
    \\@vertex
    \\fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4f {
    \\    var p = vec2f(0.0, 0.0);
    \\    if (in_vertex_index == 0u) {
    \\        p = vec2f(-0.5, -0.5);
    \\    } else if (in_vertex_index == 1u) {
    \\        p = vec2f(0.5, -0.5);
    \\    } else {
    \\        p = vec2f(0.0, 0.5);
    \\    }
    \\    return vec4f(p, 0.0, 1.0);
    \\}
    \\@fragment
    \\fn fs_main() -> @location(0) vec4f {
    \\    return vec4f(0.0, 0.4, 1.0, 1.0);
    \\}
;

const App = @This();

allocator: std.mem.Allocator,
window: *zglfw.Window,
gfx: *zgpu.GraphicsContext,
pipeline: zgpu.RenderPipelineHandle,

pub fn init(allocator: std.mem.Allocator) !*App {
    try zglfw.init();
    zglfw.windowHint(.client_api, .no_api);
    zglfw.windowHint(.resizable, false);

    const window = try zglfw.createWindow(640, 480, "Learn WebGPU", null);

    const gctx = try zgpu.GraphicsContext.create(
        allocator,
        .{
            .window = window,
            .fn_getTime = @ptrCast(&zglfw.getTime),
            .fn_getFramebufferSize = @ptrCast(&zglfw.Window.getFramebufferSize),

            // optional fields
            .fn_getWin32Window = @ptrCast(&zglfw.getWin32Window),
            .fn_getX11Display = @ptrCast(&zglfw.getX11Display),
            .fn_getX11Window = @ptrCast(&zglfw.getX11Window),
            .fn_getWaylandDisplay = @ptrCast(&zglfw.getWaylandDisplay),
            .fn_getWaylandSurface = @ptrCast(&zglfw.getWaylandWindow),
            .fn_getCocoaWindow = @ptrCast(&zglfw.getCocoaWindow),
        },
        .{}, // default context creation options
    );

    const app = try allocator.create(App);
    const pipeline = createPipeline(gctx);
    app.* = App{
        .allocator = allocator,
        .window = window,
        .gfx = gctx,
        .pipeline = pipeline,
    };
    return app;
}

pub fn deinit(self: *App) void {
    self.gfx.destroy(self.allocator);

    zglfw.destroyWindow(self.window);
    zglfw.terminate();
    self.allocator.destroy(self);
}

pub fn run(self: *App) !void {
    var buffer_desc = zgpu.wgpu.BufferDescriptor{
        .label = "Some GPU-side data buffer",
        .usage = .{ .copy_dst = true, .copy_src = true },
        .size = 16,
        .mapped_at_creation = .false,
    };
    const buffer1 = self.gfx.device.createBuffer(buffer_desc);
    defer buffer1.release();

    buffer_desc.label = "Output buffer";
    buffer_desc.usage = .{ .copy_dst = true, .map_read = true };
    buffer_desc.size = 16;
    buffer_desc.mapped_at_creation = .false;
    const buffer2 = self.gfx.device.createBuffer(buffer_desc);
    defer buffer2.release();

    // First submit the write operation
    const numbers = try self.allocator.alloc(u8, 16);
    defer self.allocator.free(numbers);
    for (0..16) |i| {
        numbers[i] = @as(u8, @intCast(i));
    }
    self.gfx.queue.writeBuffer(buffer1, 0, u8, numbers);

    var context = Context{
        .ready = false,
        .buffer = buffer2,
    };

    // Main render loop
    while (!self.window.shouldClose()) {
        // while (!context.ready) {
        self.gfx.device.tick();
        // }
        zglfw.pollEvents();

        const view = self.gfx.swapchain.getCurrentTextureView();
        defer view.release();

        const encoder = self.gfx.device.createCommandEncoder(null);
        defer encoder.release();

        encoder.copyBufferToBuffer(buffer1, 0, buffer2, 0, 16);

        const color_attachment = [_]zgpu.wgpu.RenderPassColorAttachment{.{
            .view = view,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0.4, .g = 0.8, .b = 0.2, .a = 1.0 },
        }};

        const render_pass_info = zgpu.wgpu.RenderPassDescriptor{
            .color_attachments = &color_attachment,
            .color_attachment_count = 1,
        };

        const pass = encoder.beginRenderPass(render_pass_info);
        const pipeline = self.gfx.lookupResource(self.pipeline) orelse unreachable;

        pass.setPipeline(pipeline);
        pass.draw(3, 1, 0, 0);

        pass.end();
        pass.release();

        const command_buffer = encoder.finish(null);
        defer command_buffer.release();

        self.gfx.submit(&.{command_buffer});
        _ = self.gfx.present();

        // After submitting commands, we can map the buffer to read its contents
        buffer2.mapAsync(
            .{ .read = true },
            0,
            16,
            onBuffer2Mapped,
            @as(?*anyopaque, @constCast(&context)),
        );

        // Wait until the callback sets ready to true
        while (!context.ready) {
            self.gfx.device.tick();
        }

        // Now that we know it's mapped (callback was called), we can safely read it
        if (context.buffer.getConstMappedRange(u8, 0, 16)) |data| {
            std.debug.print("Buffer contents: {any}\n", .{data});
        }
        // Unmap after reading
        context.buffer.unmap();
        // Reset ready flag for next frame
        context.ready = false;

        self.window.swapBuffers();
    }
}

fn createPipeline(gctx: *zgpu.GraphicsContext) zgpu.RenderPipelineHandle {
    const shader_module = zgpu.createWgslShaderModule(gctx.device, shader_source, "main");
    defer shader_module.release();

    const pipeline_layout = zgpu.PipelineLayoutHandle.nil;

    const color_targets = [_]zgpu.wgpu.ColorTargetState{.{
        .format = .bgra8_unorm,
        .blend = &zgpu.wgpu.BlendState{
            .color = .{
                .src_factor = .src_alpha,
                .dst_factor = .one_minus_src_alpha,
                .operation = .add,
            },
            .alpha = .{
                .src_factor = .zero,
                .dst_factor = .one,
                .operation = .add,
            },
        },
        .write_mask = .all,
    }};

    const pipeline_desc = zgpu.wgpu.RenderPipelineDescriptor{
        .vertex = .{
            .module = shader_module,
            .entry_point = "vs_main",
            .buffer_count = 0,
            .buffers = null,
        },
        .primitive = .{
            .topology = .triangle_list,
            .front_face = .ccw,
            .cull_mode = .none,
        },
        .fragment = &zgpu.wgpu.FragmentState{
            .module = shader_module,
            .entry_point = "fs_main",
            .target_count = color_targets.len,
            .targets = &color_targets,
        },
        .depth_stencil = null,
    };

    return gctx.createRenderPipeline(pipeline_layout, pipeline_desc);
}

fn onBuffer2Mapped(status: zgpu.wgpu.BufferMapAsyncStatus, userdata: ?*anyopaque) callconv(.c) void {
    const ctx: *Context = @ptrCast(@alignCast(userdata));
    ctx.ready = true;
    std.debug.print("Buffer2 mapped: {any}\n", .{status});
}
