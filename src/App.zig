const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");

const shader_source =
    \\struct VertexInput {
    \\  @location(0) position: vec2f,
    \\  @location(1) color: vec3f,
    \\};
    \\struct VertexOutput {
    \\  @builtin(position) position: vec4f,
    \\  @location(0) color: vec3f,
    \\};
    \\
    \\@vertex
    \\fn vs_main(in: VertexInput) -> VertexOutput {
    \\  var out: VertexOutput;
    \\  out.position = vec4f(in.position, 0.0, 1.0);
    \\  out.color = in.color;
    \\  let ratio = 640.0 / 480.0;
    \\  out.position = vec4f(in.position.x, in.position.y * ratio, 0.0, 1.0);
    \\  return out;
    \\}
    \\@fragment
    \\fn fs_main(in: VertexOutput) -> @location(0) vec4f  {
    \\    return vec4f(in.color, 1.0);
    \\}
;

const point_data = [_]f32{
    // x,   y,     r,   g,   b
    -0.5, -0.5, 1.0, 0.0, 0.0,
    0.5,  -0.5, 0.0, 1.0, 0.0,
    0.5,  0.5,  0.0, 0.0, 1.0,
    -0.5, 0.5,  1.0, 1.0, 0.0,
};

const index_data = [_]u16{
    0, 1, 2, // Triangle #0
    0, 2, 3, // Triangle #1
};

const App = @This();

allocator: std.mem.Allocator,
window: *zglfw.Window,
gfx: *zgpu.GraphicsContext,
pipeline: zgpu.RenderPipelineHandle,
point_buffer: zgpu.wgpu.Buffer = undefined,
index_buffer: zgpu.wgpu.Buffer = undefined,
index_count: u32,

pub fn init(allocator: std.mem.Allocator) !*App {
    try zglfw.init();
    zglfw.windowHint(.client_api, .no_api);
    zglfw.windowHint(.resizable, false);

    const window = try zglfw.createWindow(640, 480, "Learn WebGPU", null);

    const gctx = try zgpu.GraphicsContext.create(allocator, .{
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
    }, .{ .required_limits = &zgpu.wgpu.RequiredLimits{
        .limits = .{
            .max_vertex_attributes = 2,
            .max_vertex_buffers = 1,
            .max_buffer_size = 6 * 5 * @sizeOf(f32),
            .max_vertex_buffer_array_stride = 5 * @sizeOf(f32),
            .max_inter_stage_shader_components = 3,
        },
    } });

    const app = try allocator.create(App);
    const pipeline = createPipeline(gctx);
    app.* = App{
        .allocator = allocator,
        .window = window,
        .gfx = gctx,
        .pipeline = pipeline,
        .index_count = index_data.len,
    };
    app.initializeBuffers();
    return app;
}

pub fn deinit(self: *App) void {
    self.point_buffer.release();
    self.index_buffer.release();
    self.gfx.destroy(self.allocator);

    zglfw.destroyWindow(self.window);
    zglfw.terminate();
    self.allocator.destroy(self);
}

fn initializeBuffers(self: *App) void {
    var buffer_desc = zgpu.wgpu.BufferDescriptor{
        .label = "Some GPU-side data buffer",
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = point_data.len * @sizeOf(f32),
        .mapped_at_creation = .false,
    };
    // create point buffer
    self.point_buffer = self.gfx.device.createBuffer(buffer_desc);
    // upload to buffer
    self.gfx.queue.writeBuffer(self.point_buffer, 0, f32, &point_data);

    // Now the index buffer, reusing the buffer descriptor
    buffer_desc.size = index_data.len * @sizeOf(u16);
    // round size to the nearest multiple of 4
    buffer_desc.size = (buffer_desc.size + 3) & ~@as(u64, 3);
    buffer_desc.usage = .{ .copy_dst = true, .index = true };
    self.index_buffer = self.gfx.device.createBuffer(buffer_desc);

    // First submit the write operation
    self.gfx.queue.writeBuffer(self.index_buffer, 0, u16, &index_data);
}

pub fn run(self: *App) !void {
    // Main render loop
    while (!self.window.shouldClose()) {
        self.gfx.device.tick();
        zglfw.pollEvents();

        const view = self.gfx.swapchain.getCurrentTextureView();
        defer view.release();

        const encoder = self.gfx.device.createCommandEncoder(null);
        defer encoder.release();

        const color_attachment = [_]zgpu.wgpu.RenderPassColorAttachment{.{
            .view = view,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0.05, .g = 0.05, .b = 0.05, .a = 1.0 },
        }};

        const render_pass_info = zgpu.wgpu.RenderPassDescriptor{
            .color_attachments = &color_attachment,
            .color_attachment_count = 1,
        };

        const pass = encoder.beginRenderPass(render_pass_info);
        const pipeline = self.gfx.lookupResource(self.pipeline) orelse unreachable;

        pass.setPipeline(pipeline);

        pass.setVertexBuffer(0, self.point_buffer, 0, self.point_buffer.getSize());
        pass.setIndexBuffer(self.index_buffer, .uint16, 0, self.index_buffer.getSize());
        pass.drawIndexed(self.index_count, 1, 0, 0, 0);

        pass.end();
        pass.release();

        const command_buffer = encoder.finish(null);
        defer command_buffer.release();

        self.gfx.submit(&.{command_buffer});
        _ = self.gfx.present();

        // After submitting commands, we can map the buffer to read its contents
        // buffer2.mapAsync(
        //     .{ .read = true },
        //     0,
        //     16,
        //     onBuffer2Mapped,
        //     @as(?*anyopaque, @constCast(&context)),
        // );

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

    const position_attribute = zgpu.wgpu.VertexAttribute{
        .shader_location = 0,
        .format = .float32x2,
        .offset = 0,
    };

    const color_attribute = zgpu.wgpu.VertexAttribute{
        .shader_location = 1,
        .format = .float32x3,
        .offset = 2 * @sizeOf(f32),
    };

    const vertex_buffer_layout = zgpu.wgpu.VertexBufferLayout{
        .array_stride = 5 * @sizeOf(f32),
        .attribute_count = 2,
        .attributes = &[_]zgpu.wgpu.VertexAttribute{
            position_attribute,
            color_attribute,
        },
    };

    const pipeline_desc = zgpu.wgpu.RenderPipelineDescriptor{
        .vertex = .{
            .module = shader_module,
            .entry_point = "vs_main",
            .buffer_count = 1,
            .buffers = &[_]zgpu.wgpu.VertexBufferLayout{vertex_buffer_layout},
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
