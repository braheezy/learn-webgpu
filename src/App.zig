const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const zmath = @import("zmath");
const obj = @import("obj");
const jpeg = @import("zjpeg");
const zgui = @import("zgui");

const gui = @import("gui.zig");
const toRadians = @import("math_utils.zig").toRadians;

const ResourceManager = @import("ResourceManager.zig");
const VertexAttr = ResourceManager.VertexAttr;

const obj_file = "src/resources/fourareen/fourareen.obj";
const vertex_text_file = "src/resources/pyramid.txt";
const texture_file = "src/resources/fourareen/fourareen2K_albedo.jpg";

const depth_format = zgpu.wgpu.TextureFormat.depth24_plus;

const MyUniforms = struct {
    projection: zmath.Mat = undefined,
    view: zmath.Mat = undefined,
    model: zmath.Mat = undefined,
    color: [4]f32 = .{ 0.0, 1.0, 0.4, 1.0 },
    time: f32 = 1.0,
    padding: [3]f32 = [_]f32{0} ** 3,
};

const Camera = struct {
    angles: [2]f32 = .{ 0.8, 0.5 },
    zoom: f32 = -1.2,
};

const DragState = struct {
    // Whether a drag action is ongoing (i.e., we are between mouse press and mouse release)
    active: bool = false,
    // The position of the mouse at the beginning of the drag action
    start_position: [2]f64 = .{ 0.0, 0.0 },
    // The camera state at the beginning of the drag action
    start_camera: Camera = .{},
    // Inertia
    velocity: [2]f64 = .{ 0.0, 0.0 },
    previous_delta: [2]f64 = .{ 0.0, 0.0 },

    const sensitivity: f32 = 0.01;
    const scroll_sensitivity: f32 = 0.1;
    const inertia: f32 = 0.9;
};

pub const Lighting = struct {
    directions: [2][4]f32 = .{
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
    },
    colors: [2][4]f32 = .{
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
    },
    enable_gamma: u32 = 0,
    padding: [3]u32 = [_]u32{0} ** 3,
};
const App = @This();

allocator: std.mem.Allocator,
window: *zglfw.Window,
gfx: *zgpu.GraphicsContext = undefined,
pipeline: zgpu.RenderPipelineHandle = undefined,
vertex_buffer: zgpu.wgpu.Buffer = undefined,
vertex_count: u32 = 0,
lighting: Lighting = .{},
bind_group: zgpu.BindGroupHandle = undefined,
depth_texture: zgpu.TextureHandle = undefined,
depth_view: zgpu.TextureViewHandle = undefined,
texture: zgpu.TextureHandle = undefined,
texture_view: ?zgpu.TextureViewHandle = null,
my_uniforms: MyUniforms = .{},
camera: Camera = .{},
drag_state: DragState = .{},
uniform_offset: u32 = 0,
lighting_offset: u32 = 0,

pub fn init(allocator: std.mem.Allocator) !*App {
    const app = try createApp(allocator);
    app.createDepthBuffer();
    try app.createTexture(texture_file);
    try app.createGeometry();
    try app.createPipeline();
    app.createUniforms();
    app.createLightUniforms();

    try gui.create(
        allocator,
        app.window,
        app.gfx.device,
        app.gfx.swapchain_descriptor.format,
    );

    return app;
}

pub fn deinit(self: *App) void {
    gui.destroy();
    self.vertex_buffer.release();
    self.cleanDepthBuffer();
    self.gfx.releaseResource(self.texture);
    self.gfx.destroyResource(self.texture);

    self.gfx.destroy(self.allocator);

    zglfw.destroyWindow(self.window);
    zglfw.terminate();
    self.allocator.destroy(self);
}

fn cleanDepthBuffer(self: *App) void {
    self.gfx.releaseResource(self.depth_texture);
    self.gfx.destroyResource(self.depth_texture);
    self.gfx.releaseResource(self.depth_view);
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
                .max_bind_groups = 2,
                .max_uniform_buffers_per_shader_stage = 2,
                .max_uniform_buffer_binding_size = @max(@sizeOf(MyUniforms), @sizeOf(Lighting)),
                .max_dynamic_uniform_buffers_per_pipeline_layout = 1,
                .max_texture_dimension_1d = 2048,
                .max_texture_dimension_2d = 2048,
                .max_texture_array_layers = 1,
                .max_sampled_textures_per_shader_stage = 1,
                .max_samplers_per_shader_stage = 1,
            },
        },
    });

    app.createCallbacks();

    return app;
}

fn createCallbacks(self: *App) void {
    // Get pointer to App to pass to callbacks
    zglfw.setWindowUserPointer(self.window, @ptrCast(self));

    _ = zglfw.setCursorPosCallback(self.window, struct {
        fn cb(window: *zglfw.Window, xpos: f64, ypos: f64) callconv(.C) void {
            // If ImGui is using the mouse, ignore the event
            if (zgui.io.getWantCaptureMouse()) return;

            const app = window.getUserPointer(App) orelse unreachable;
            if (app.drag_state.active) {
                // Handle high DPI displays by scaling the mouse position
                const scale = window.getContentScale();

                const current_mouse_pos: [2]f64 = .{ xpos / scale[0], ypos / scale[1] };
                const delta: [2]f64 = .{
                    (current_mouse_pos[0] - app.drag_state.start_position[0]) * DragState.sensitivity,
                    (current_mouse_pos[1] - app.drag_state.start_position[1]) * DragState.sensitivity,
                };
                app.camera.angles[0] = app.drag_state.start_camera.angles[0] + @as(f32, @floatCast(delta[0]));
                app.camera.angles[1] = app.drag_state.start_camera.angles[1] + @as(f32, @floatCast(delta[1]));
                // Clamp to avoid going too far when orbitting up/down
                app.camera.angles[1] = zmath.clamp(
                    app.camera.angles[1],
                    -std.math.pi / 2.0 + 1e-5,
                    std.math.pi / 2.0 - 1e-5,
                );
                app.updateView();

                app.drag_state.velocity = .{
                    delta[0] - app.drag_state.previous_delta[0],
                    delta[1] - app.drag_state.previous_delta[1],
                };
                app.drag_state.previous_delta = delta;
            }
        }
    }.cb);

    _ = zglfw.setMouseButtonCallback(self.window, struct {
        fn cb(
            window: *zglfw.Window,
            button: zglfw.MouseButton,
            action: zglfw.Action,
            mods: zglfw.Mods,
        ) callconv(.C) void {
            // If ImGui is using the mouse, ignore the event
            if (zgui.io.getWantCaptureMouse()) return;

            const app = window.getUserPointer(App) orelse unreachable;
            _ = mods;

            if (button == .left) {
                switch (action) {
                    .press => {
                        app.drag_state.active = true;
                        const cursor_pos = app.window.getCursorPos();
                        app.drag_state.start_position = .{ cursor_pos[0], cursor_pos[1] };
                        app.drag_state.start_camera = app.camera;
                    },
                    .release => {
                        app.drag_state.active = false;
                    },
                    else => {},
                }
            }
        }
    }.cb);

    _ = zglfw.setScrollCallback(self.window, struct {
        fn cb(window: *zglfw.Window, x_offset: f64, y_offset: f64) callconv(.C) void {
            const app = window.getUserPointer(App) orelse unreachable;
            _ = x_offset;

            app.camera.zoom += @as(f32, @floatCast(y_offset)) * DragState.scroll_sensitivity;
            app.camera.zoom = zmath.clamp(app.camera.zoom, -2.0, 2.0);
            app.updateView();
        }
    }.cb);
}

pub fn isRunning(self: *App) bool {
    return !self.window.shouldClose() and self.window.getKey(.escape) != .press;
}

fn createWindow() !*zglfw.Window {
    try zglfw.init();
    zglfw.windowHint(.client_api, .no_api);
    zglfw.windowHint(.resizable, true);

    const window = try zglfw.createWindow(640, 480, "Learn WebGPU", null);

    return window;
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

fn updatePerspective(self: *App) void {
    self.my_uniforms.projection = zmath.perspectiveFovLh(
        toRadians(45.0),
        @as(f32, @floatFromInt(self.gfx.swapchain_descriptor.width)) / @as(f32, @floatFromInt(self.gfx.swapchain_descriptor.height)),
        0.01,
        100,
    );
}

fn updateView(self: *App) void {
    const cx = @cos(self.camera.angles[0]);
    const sx = @sin(self.camera.angles[0]);
    const cy = @cos(self.camera.angles[1]);
    const sy = @sin(self.camera.angles[1]);

    const zoom_factor = std.math.exp(-self.camera.zoom);
    const position = zmath.Vec{
        cx * cy * zoom_factor,
        sx * cy * zoom_factor,
        sy * zoom_factor,
        1.0,
    };

    self.my_uniforms.view = zmath.lookAtLh(
        position,
        zmath.Vec{ 0.0, 0.0, 0.0, 1.0 },
        zmath.Vec{ 0.0, 0.0, 1.0, 1.0 },
    );
}

fn createUniforms(self: *App) void {
    self.updatePerspective();

    self.my_uniforms.model = zmath.identity();
    self.updateView();
}

fn createLightUniforms(self: *App) void {
    self.lighting.directions[0] = zmath.Vec{ 0.5, -0.9, 0.1, 0.0 };
    self.lighting.directions[1] = zmath.Vec{ 0.2, 0.4, 0.3, 0.0 };
    self.lighting.colors[0] = zmath.Vec{ 1.0, 0.9, 0.6, 1.0 };
    self.lighting.colors[1] = zmath.Vec{ 0.6, 0.9, 1.0, 1.0 };

    const lighting_mem = self.gfx.uniformsAllocate(Lighting, 1);
    lighting_mem.slice[0] = self.lighting;
    self.lighting_offset = lighting_mem.offset;
}

pub fn update(self: *App) !void {
    zglfw.pollEvents();

    gui.update(self);

    self.updateDragInertia();

    const time = @as(f32, @floatCast(self.gfx.stats.time));
    self.my_uniforms.time = time;

    // Allocate and update the MyUniforms struct
    const uni_mem = self.gfx.uniformsAllocate(MyUniforms, 1);
    uni_mem.slice[0] = self.my_uniforms;

    const lighting_mem = self.gfx.uniformsAllocate(Lighting, 1);
    lighting_mem.slice[0] = self.lighting;

    // Store offsets for use in drawing
    self.uniform_offset = uni_mem.offset;
    self.lighting_offset = lighting_mem.offset;
}

pub fn draw(self: *App) !void {
    const depth_view = self.gfx.lookupResource(self.depth_view) orelse unreachable;
    const view = self.gfx.swapchain.getCurrentTextureView();
    defer view.release();

    const encoder = self.gfx.device.createCommandEncoder(null);
    defer encoder.release();

    try self.drawModel(encoder, view, depth_view);
    try gui.draw(encoder, view);

    const command_buffer = encoder.finish(null);
    defer command_buffer.release();

    self.gfx.submit(&.{command_buffer});

    if (self.gfx.present() == .swap_chain_resized) {
        self.cleanDepthBuffer();
        self.createDepthBuffer();
        self.updatePerspective();
    }
}

fn drawModel(self: *App, encoder: zgpu.wgpu.CommandEncoder, view: zgpu.wgpu.TextureView, depth_view: zgpu.wgpu.TextureView) !void {
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

    const pass = encoder.beginRenderPass(render_pass_info);
    defer zgpu.endReleasePass(pass);

    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, self.vertex_buffer, 0, self.vertex_buffer.getSize());

    // Pass the dynamic offsets for both uniform buffers
    pass.setBindGroup(
        0,
        bind_group,
        &.{ self.uniform_offset, self.lighting_offset },
    );

    pass.draw(self.vertex_count, 1, 0, 0);
}

fn updateDragInertia(self: *App) void {
    const eps: f32 = 1e-4;
    // Apply inertia only when the user released the click
    if (!self.drag_state.active) {
        // Avoid updating the matrix when the velocity is no longer noticeable
        if (@abs(self.drag_state.velocity[0]) < eps and @abs(self.drag_state.velocity[1]) < eps) {
            return;
        }
        self.camera.angles[0] += @as(f32, @floatCast(self.drag_state.velocity[0]));
        self.camera.angles[1] += @as(f32, @floatCast(self.drag_state.velocity[1]));
        // Clamp to avoid going too far when orbitting up/down
        self.camera.angles[1] = zmath.clamp(
            self.camera.angles[1],
            -std.math.pi / 2.0 + 1e-5,
            std.math.pi / 2.0 - 1e-5,
        );
        // Dampen the velocity so that it decreases exponentially and stops
        // after a few frames
        self.drag_state.velocity[0] *= DragState.inertia;
        self.drag_state.velocity[1] *= DragState.inertia;
        self.updateView();
    }
}

fn createPipeline(self: *App) !void {
    const shader_module = try ResourceManager.loadShaderModule(
        self.allocator,
        "src/resources/shader.wgsl",
        self.gfx.device,
    );
    defer shader_module.release();

    const uniform_bg = zgpu.bufferEntry(
        0,
        .{ .vertex = true, .fragment = true },
        .uniform,
        true,
        0,
    );
    const lighting_uniform_bg = zgpu.bufferEntry(
        3,
        .{ .fragment = true },
        .uniform,
        true,
        0,
    );

    const bind_group_layout = self.gfx.createBindGroupLayout(&.{
        uniform_bg,
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
        lighting_uniform_bg,
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
        .format = depth_format,
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

    // Create 3D model pipeline
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
        .{
            .binding = 3,
            .buffer_handle = self.gfx.uniforms.buffer,
            .offset = 0,
            .size = @sizeOf(Lighting),
        },
    });
}

fn createDepthBuffer(self: *App) void {
    self.depth_texture = self.gfx.createTexture(.{
        .dimension = .tdim_2d,
        .format = depth_format,
        .mip_level_count = 1,
        .sample_count = 1,
        .size = .{
            .width = self.gfx.swapchain_descriptor.width,
            .height = self.gfx.swapchain_descriptor.height,
            .depth_or_array_layers = 1,
        },
        .usage = .{ .render_attachment = true },
        .view_format_count = 1,
        .view_formats = &[_]zgpu.wgpu.TextureFormat{depth_format},
    });

    self.depth_view = self.gfx.createTextureView(self.depth_texture, .{
        .aspect = .depth_only,
        .base_array_layer = 0,
        .array_layer_count = 1,
        .base_mip_level = 0,
        .mip_level_count = 1,
        .dimension = .tvdim_2d,
        .format = depth_format,
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
