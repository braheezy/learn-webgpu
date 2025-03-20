const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const zmath = @import("zmath");
const obj = @import("obj");
const jpeg = @import("zjpeg");

const ResourceManager = @import("ResourceManager.zig");
const VertexAttr = ResourceManager.VertexAttr;

const obj_file = "src/resources/fourareen/fourareen.obj";
const vertex_text_file = "src/resources/pyramid.txt";
const texture_file = "src/resources/fourareen/fourareen2K_albedo.jpg";

const MyUniforms = struct {
    projection: zmath.Mat = undefined,
    view: zmath.Mat = undefined,
    model: zmath.Mat = undefined,
    color: [4]f32 = .{ 0.0, 1.0, 0.4, 1.0 },
    time: f32 = 1.0,
    padding: [3]f32 = [_]f32{0} ** 3,
};

const App = @This();

allocator: std.mem.Allocator,
window: *zglfw.Window,
gfx: *zgpu.GraphicsContext = undefined,
pipeline: zgpu.RenderPipelineHandle = undefined,
vertex_buffer: zgpu.wgpu.Buffer = undefined,
vertex_count: u32 = 0,
bind_group: zgpu.BindGroupHandle = undefined,
depth_texture: zgpu.TextureHandle = undefined,
depth_view: zgpu.TextureViewHandle = undefined,
texture: zgpu.TextureHandle = undefined,
texture_view: ?zgpu.TextureViewHandle = null,
my_uniforms: MyUniforms = .{},

pub fn init(allocator: std.mem.Allocator) !*App {
    const app = try createApp(allocator);
    try app.createDepthBuffer();
    try app.createTexture(texture_file);
    try app.createGeometry();
    try app.createPipeline(allocator);
    try app.createUniforms();

    return app;
}

fn createApp(allocator: std.mem.Allocator) !*App {
    const window = try createWindow();
    const app = try allocator.create(App);
    app.* = App{
        .allocator = allocator,
        .window = window,
        .texture_view = null,
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
    }, .{
        .required_limits = &zgpu.wgpu.RequiredLimits{
            .limits = .{
                .max_vertex_attributes = 4,
                .max_vertex_buffers = 1,
                .max_buffer_size = 150000 * @sizeOf(VertexAttr),
                .max_vertex_buffer_array_stride = @sizeOf(VertexAttr),
                .max_inter_stage_shader_components = 8,
                .max_bind_groups = 1,
                .max_uniform_buffers_per_shader_stage = 1,
                .max_uniform_buffer_binding_size = 16 * 4 * @sizeOf(f32),
                .max_dynamic_uniform_buffers_per_pipeline_layout = 1,
                .max_texture_dimension_1d = 2048,
                .max_texture_dimension_2d = 2048,
                .max_texture_array_layers = 1,
                .max_sampled_textures_per_shader_stage = 1,
                .max_samplers_per_shader_stage = 1,
            },
        },
    });

    return app;
}

pub fn deinit(self: *App) void {
    self.vertex_buffer.release();
    self.gfx.releaseResource(self.depth_texture);
    self.gfx.destroyResource(self.depth_texture);
    self.gfx.releaseResource(self.depth_view);
    self.gfx.releaseResource(self.texture);
    self.gfx.destroyResource(self.texture);

    self.gfx.destroy(self.allocator);

    zglfw.destroyWindow(self.window);
    zglfw.terminate();
    self.allocator.destroy(self);
}

pub fn isRunning(self: *App) bool {
    return !self.window.shouldClose();
}

pub fn createWindow() !*zglfw.Window {
    try zglfw.init();
    zglfw.windowHint(.client_api, .no_api);
    zglfw.windowHint(.resizable, false);

    return zglfw.createWindow(640, 480, "Learn WebGPU", null);
}

fn createGeometry(self: *App) !void {
    const vertex_data = try ResourceManager.loadGeometryFromObj(
        self.allocator,
        obj_file,
    );
    defer vertex_data.deinit();
    self.vertex_count = @intCast(vertex_data.items.len);

    const buffer_desc = zgpu.wgpu.BufferDescriptor{
        .label = "Vertex buffer",
        .usage = .{ .copy_dst = true, .vertex = true },
        .size = vertex_data.items.len * @sizeOf(VertexAttr),
        .mapped_at_creation = .false,
    };
    self.vertex_buffer = self.gfx.device.createBuffer(buffer_desc);
    // upload to buffer
    std.debug.print("vertex_buffer size: {}\n", .{vertex_data.items.len});
    self.gfx.queue.writeBuffer(self.vertex_buffer, 0, VertexAttr, vertex_data.items);
}

fn createUniforms(self: *App) !void {
    self.my_uniforms.projection = zmath.perspectiveFovLh(
        toRadians(45.0),
        640.0 / 480.0,
        0.01,
        100,
    );

    self.my_uniforms.model = zmath.identity();
}

fn toRadians(degrees: f32) f32 {
    return degrees * std.math.pi / 180.0;
}

pub fn update(self: *App) !void {
    const depth_view = self.gfx.lookupResource(self.depth_view) orelse unreachable;
    const pipeline = self.gfx.lookupResource(self.pipeline) orelse unreachable;
    const bind_group = self.gfx.lookupResource(self.bind_group) orelse unreachable;

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

    // Main render loop
    while (!self.window.shouldClose()) {
        self.gfx.device.tick();
        zglfw.pollEvents();

        const time = @as(f32, @floatCast(self.gfx.stats.time));
        self.my_uniforms.time = time;

        const v0: f32 = 0.0;
        const v1: f32 = 0.25;
        const viewZ = lerp(v0, v1, @cos(2.0 * std.math.pi * time / 4.0) * 0.5 + 0.5);
        self.my_uniforms.view = zmath.lookAtLh(
            zmath.loadArr3(.{ -1.5, -3.0, viewZ + 0.25 }),
            zmath.loadArr3(.{ 0.0, 0.0, 0.0 }),
            zmath.loadArr3(.{ 0.0, 0.0, 1.0 }),
        );

        // Allocate and update the entire uniform struct
        const uni_mem = self.gfx.uniformsAllocate(MyUniforms, 1);
        uni_mem.slice[0] = self.my_uniforms;

        const view = self.gfx.swapchain.getCurrentTextureView();
        defer view.release();

        const color_attachment = [_]zgpu.wgpu.RenderPassColorAttachment{.{
            .view = view,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0.1, .g = 0.1, .b = 0.1, .a = 1.0 },
        }};

        const render_pass_info = zgpu.wgpu.RenderPassDescriptor{
            .color_attachments = &color_attachment,
            .color_attachment_count = 1,
            .depth_stencil_attachment = &depth_attachment,
        };

        const encoder = self.gfx.device.createCommandEncoder(null);
        defer encoder.release();

        const pass = encoder.beginRenderPass(render_pass_info);

        pass.setPipeline(pipeline);

        pass.setVertexBuffer(0, self.vertex_buffer, 0, self.vertex_buffer.getSize());

        pass.setBindGroup(0, bind_group, null);

        pass.draw(self.vertex_count, 1, 0, 0);

        pass.end();
        pass.release();

        const command_buffer = encoder.finish(null);
        defer command_buffer.release();

        self.gfx.submit(&.{command_buffer});
        _ = self.gfx.present();

        self.window.swapBuffers();
    }
}

pub fn lerp(v0: f32, v1: f32, t: f32) f32 {
    return v0 * (1.0 - t) + v1 * t;
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
        .{
            .binding = 1,
            .visibility = .{ .fragment = true },
            .texture = .{
                .sample_type = .float,
                .view_dimension = .tvdim_2d,
            },
        },
        .{
            .binding = 2,
            .visibility = .{ .fragment = true },
            .sampler = .{
                .binding_type = .filtering,
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

    const normal_attribute = zgpu.wgpu.VertexAttribute{
        .shader_location = 1,
        .format = .float32x3,
        .offset = @offsetOf(VertexAttr, "normal"),
    };

    const color_attribute = zgpu.wgpu.VertexAttribute{
        .shader_location = 2,
        .format = .float32x3,
        .offset = @offsetOf(VertexAttr, "color"),
    };

    const uv_attribute = zgpu.wgpu.VertexAttribute{
        .shader_location = 3,
        .format = .float32x2,
        .offset = @offsetOf(VertexAttr, "uv"),
    };

    const vertex_buffer_layout = zgpu.wgpu.VertexBufferLayout{
        .array_stride = @sizeOf(VertexAttr),
        .attribute_count = 4,
        .attributes = &[_]zgpu.wgpu.VertexAttribute{
            position_attribute,
            normal_attribute,
            color_attribute,
            uv_attribute,
        },
    };

    const depth_stencil = zgpu.wgpu.DepthStencilState{
        .depth_compare = .less,
        .depth_write_enabled = true,
        .format = .depth24_plus,
        .stencil_read_mask = 0,
        .stencil_write_mask = 0,
    };

    const sampler = self.gfx.createSampler(.{
        .address_mode_u = .repeat,
        .address_mode_v = .repeat,
        .address_mode_w = .repeat,
        .mag_filter = .linear,
        .min_filter = .linear,
        .mipmap_filter = .linear,
        .lod_min_clamp = 0.0,
        .lod_max_clamp = 8.0,
        .compare = .undef,
        .max_anisotropy = 1,
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

    self.bind_group = self.gfx.createBindGroup(bind_group_layout, &.{
        .{
            .binding = 0,
            .buffer_handle = self.gfx.uniforms.buffer,
            .offset = 0,
            .size = @sizeOf(MyUniforms),
        },
        .{
            .binding = 1,
            .texture_view_handle = self.texture_view,
        },
        .{
            .binding = 2,
            .sampler_handle = sampler,
        },
    });
}

fn createDepthBuffer(self: *App) !void {
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
}

fn createTexture(self: *App, path: []const u8) !void {
    self.texture = try ResourceManager.loadTexture(
        self.allocator,
        self.gfx,
        path,
        &self.texture_view,
    );
}
