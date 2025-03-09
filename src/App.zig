const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");

const ResourceManager = @import("ResourceManager.zig");

const vertex_text_file = @embedFile("resources/pyramid.txt");

const MyUniforms = struct {
    color: [4]f32 = .{ 0.0, 1.0, 0.4, 1.0 },
    time: f32 = 1.0,
    padding: [3]f32 = [_]f32{0} ** 3,
};

comptime {
    std.debug.assert(@sizeOf(MyUniforms) == 32);
    std.debug.assert(@sizeOf(MyUniforms) % 16 == 0);
}

const App = @This();

allocator: std.mem.Allocator,
window: *zglfw.Window,
gfx: *zgpu.GraphicsContext = undefined,
pipeline: zgpu.RenderPipelineHandle = undefined,
point_buffer: zgpu.wgpu.Buffer = undefined,
index_buffer: zgpu.wgpu.Buffer = undefined,
index_count: u32 = 0,
bind_group: zgpu.BindGroupHandle = undefined,
depth_texture: zgpu.TextureHandle = undefined,
depth_view: zgpu.TextureViewHandle = undefined,
my_uniforms: MyUniforms = .{},

pub fn init(allocator: std.mem.Allocator) !*App {
    try zglfw.init();
    zglfw.windowHint(.client_api, .no_api);
    zglfw.windowHint(.resizable, false);

    const window = try zglfw.createWindow(640, 480, "Learn WebGPU", null);
    const app = try allocator.create(App);
    app.* = App{
        .allocator = allocator,
        .window = window,
    };

    app.gfx = try zgpu.GraphicsContext.create(allocator, .{
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
            .max_buffer_size = 15 * 5 * @sizeOf(f32),
            .max_vertex_buffer_array_stride = 6 * @sizeOf(f32),
            .max_inter_stage_shader_components = 3,
            .max_bind_groups = 1,
            .max_uniform_buffers_per_shader_stage = 1,
            .max_uniform_buffer_binding_size = 16 * 4,
            .max_dynamic_uniform_buffers_per_pipeline_layout = 1,
            .max_texture_dimension_1d = 640,
            .max_texture_dimension_2d = 480,
            .max_texture_array_layers = 1,
        },
    } });

    try app.createPipeline(allocator);

    try app.initializeBuffers();
    return app;
}

pub fn deinit(self: *App) void {
    self.point_buffer.release();
    self.index_buffer.release();
    self.gfx.releaseResource(self.depth_texture);
    self.gfx.destroyResource(self.depth_texture);
    self.gfx.releaseResource(self.depth_view);

    self.gfx.destroy(self.allocator);

    zglfw.destroyWindow(self.window);
    zglfw.terminate();
    self.allocator.destroy(self);
}

fn initializeBuffers(self: *App) !void {
    var point_data = std.ArrayList(f32).init(self.allocator);
    var index_data = std.ArrayList(u16).init(self.allocator);
    defer point_data.deinit();
    defer index_data.deinit();

    try ResourceManager.loadGeometry(
        vertex_text_file,
        &point_data,
        &index_data,
        3,
    );
    self.index_count = @intCast(index_data.items.len);

    var buffer_desc = zgpu.wgpu.BufferDescriptor{
        .label = "Vertex buffer",
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = point_data.items.len * @sizeOf(f32),
        .mapped_at_creation = .false,
    };
    buffer_desc.size = (buffer_desc.size + 3) & ~@as(u64, 3);
    // create point buffer
    self.point_buffer = self.gfx.device.createBuffer(buffer_desc);
    // upload to buffer
    self.gfx.queue.writeBuffer(self.point_buffer, 0, f32, point_data.items);

    // Now the index buffer, reusing the buffer descriptor
    buffer_desc.label = "Index Buffer";
    buffer_desc.size = index_data.items.len * @sizeOf(u16);
    // round size to the nearest multiple of 4
    buffer_desc.size = (buffer_desc.size + 3) & ~@as(u64, 3);
    buffer_desc.usage = .{ .copy_dst = true, .index = true };
    self.index_buffer = self.gfx.device.createBuffer(buffer_desc);

    if (index_data.items.len != buffer_desc.size) {
        // Pad index_data to the nearest multiple of 4
        const padding = buffer_desc.size - (index_data.items.len * @sizeOf(u16));
        const padding_data = try self.allocator.alloc(u16, padding / @sizeOf(u16));
        @memset(padding_data, 0);
        defer self.allocator.free(padding_data);
        try index_data.appendSlice(padding_data);
    }

    // First submit the write operation
    self.gfx.queue.writeBuffer(self.index_buffer, 0, u16, index_data.items);
}

pub fn run(self: *App) !void {
    // Main render loop
    while (!self.window.shouldClose()) {
        self.gfx.device.tick();
        zglfw.pollEvents();

        const time = @as(f32, @floatCast(self.gfx.stats.time));
        self.my_uniforms.time = time;
        self.my_uniforms.color = .{ 0.0, 1.0, 0.4, 1.0 }; // Green tint as in tutorial

        // Allocate and update the entire uniform struct
        const uni_mem = self.gfx.uniformsAllocate(MyUniforms, 1);
        uni_mem.slice[0] = self.my_uniforms;

        const view = self.gfx.swapchain.getCurrentTextureView();
        defer view.release();

        const encoder = self.gfx.device.createCommandEncoder(null);
        defer encoder.release();

        const color_attachment = [_]zgpu.wgpu.RenderPassColorAttachment{.{
            .view = view,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0.2, .g = 0.2, .b = 0.2, .a = 1.0 },
        }};

        const depth_view = self.gfx.lookupResource(self.depth_view) orelse unreachable;

        const depth_attachment = zgpu.wgpu.RenderPassDepthStencilAttachment{
            .view = depth_view,
            .depth_clear_value = 1.0,
            .depth_load_op = .clear,
            .depth_store_op = .store,
            .depth_read_only = .false,
            .stencil_clear_value = 0,
            .stencil_load_op = .undef,
            .stencil_store_op = .undef,
            .stencil_read_only = .true,
        };

        const render_pass_info = zgpu.wgpu.RenderPassDescriptor{
            .color_attachments = &color_attachment,
            .color_attachment_count = 1,
            .depth_stencil_attachment = &depth_attachment,
        };

        const pass = encoder.beginRenderPass(render_pass_info);
        const pipeline = self.gfx.lookupResource(self.pipeline) orelse unreachable;
        const bind_group = self.gfx.lookupResource(self.bind_group) orelse unreachable;

        pass.setPipeline(pipeline);

        pass.setVertexBuffer(0, self.point_buffer, 0, self.point_buffer.getSize());
        pass.setIndexBuffer(self.index_buffer, .uint16, 0, self.index_buffer.getSize());

        pass.setBindGroup(0, bind_group, null);

        pass.drawIndexed(self.index_count, 1, 0, 0, 0);

        pass.end();
        pass.release();

        const command_buffer = encoder.finish(null);
        defer command_buffer.release();

        self.gfx.submit(&.{command_buffer});
        _ = self.gfx.present();

        self.window.swapBuffers();
    }
}

fn createPipeline(self: *App, allocator: std.mem.Allocator) !void {
    const shader_module = try ResourceManager.loadShaderModule(
        allocator,
        "src/resources/shader.wgsl",
        self.gfx.device,
    );
    defer shader_module.release();

    const bind_group_layout = self.gfx.createBindGroupLayout(&.{
        .{
            .binding = 0,
            .visibility = .{ .vertex = true, .fragment = true },
            .buffer = .{
                .binding_type = .uniform,
                .min_binding_size = @sizeOf(MyUniforms),
            },
        },
    });
    defer self.gfx.releaseResource(bind_group_layout);

    const pipeline_layout = self.gfx.createPipelineLayout(&.{bind_group_layout});
    defer self.gfx.releaseResource(pipeline_layout);

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
        .format = .float32x3,
        .offset = 0,
    };

    const color_attribute = zgpu.wgpu.VertexAttribute{
        .shader_location = 1,
        .format = .float32x3,
        .offset = 3 * @sizeOf(f32),
    };

    const vertex_buffer_layout = zgpu.wgpu.VertexBufferLayout{
        .array_stride = 6 * @sizeOf(f32),
        .attribute_count = 2,
        .attributes = &[_]zgpu.wgpu.VertexAttribute{
            position_attribute,
            color_attribute,
        },
    };

    const depth_stencil = zgpu.wgpu.DepthStencilState{
        .depth_compare = .less,
        .depth_write_enabled = true,
        .format = .depth24_plus,
        .stencil_read_mask = 0,
        .stencil_write_mask = 0,
    };

    // texture for the depth buffer
    self.depth_texture = self.gfx.createTexture(.{
        .dimension = .tdim_2d,
        .format = .depth24_plus,
        .mip_level_count = 1,
        .sample_count = 1,
        .size = .{
            .width = self.gfx.swapchain_descriptor.width,
            .height = self.gfx.swapchain_descriptor.height,
            .depth_or_array_layers = 1,
        },
        .usage = .{ .render_attachment = true },
        .view_format_count = 1,
        .view_formats = &[_]zgpu.wgpu.TextureFormat{.depth24_plus},
    });

    self.depth_view = self.gfx.createTextureView(self.depth_texture, .{
        .aspect = .depth_only,
        .base_array_layer = 0,
        .array_layer_count = 1,
        .base_mip_level = 0,
        .mip_level_count = 1,
        .dimension = .tvdim_2d,
        .format = .depth24_plus,
    });

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
        .depth_stencil = &depth_stencil,
    };

    self.pipeline = self.gfx.createRenderPipeline(pipeline_layout, pipeline_desc);

    self.bind_group = self.gfx.createBindGroup(bind_group_layout, &.{.{
        .binding = 0,
        .buffer_handle = self.gfx.uniforms.buffer,
        .offset = 0,
        .size = @sizeOf(MyUniforms),
    }});
}
