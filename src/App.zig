const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const zmath = @import("zmath");
const obj = @import("obj");

const ResourceManager = @import("ResourceManager.zig");

const vertex_text_file = @embedFile("resources/pyramid.txt");

const VertexAttr = struct {
    position: [3]f32,
    normal: [3]f32,
    color: [3]f32,
};

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
            .max_vertex_attributes = 3,
            .max_vertex_buffers = 1,
            .max_buffer_size = 10000 * @sizeOf(VertexAttr),
            .max_vertex_buffer_array_stride = @sizeOf(VertexAttr),
            .max_inter_stage_shader_components = 3,
            .max_bind_groups = 1,
            .max_uniform_buffers_per_shader_stage = 1,
            .max_uniform_buffer_binding_size = 16 * 4 * @sizeOf(f32),
            .max_dynamic_uniform_buffers_per_pipeline_layout = 1,
            .max_texture_dimension_1d = 640,
            .max_texture_dimension_2d = 480,
            .max_texture_array_layers = 1,
            .max_inter_stage_shader_variables = 6,
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
    // Load the OBJ model instead of using the text file
    var obj_model = try obj.parseObj(self.allocator, @embedFile("resources/mammoth.obj"));
    defer obj_model.deinit(self.allocator);

    // Create vertex and index data arrays to hold the converted data
    var point_data = std.ArrayList(f32).init(self.allocator);
    var index_data = std.ArrayList(u32).init(self.allocator);
    defer point_data.deinit();
    defer index_data.deinit();

    // Process the OBJ data and fill the point_data and index_data arrays
    try processObjData(obj_model, &point_data, &index_data);
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
    buffer_desc.size = index_data.items.len * @sizeOf(u32);
    // round size to the nearest multiple of 4
    buffer_desc.size = (buffer_desc.size + 3) & ~@as(u64, 3);
    buffer_desc.usage = .{ .copy_dst = true, .index = true };
    self.index_buffer = self.gfx.device.createBuffer(buffer_desc);

    if (index_data.items.len != buffer_desc.size) {
        // Pad index_data to the nearest multiple of 4
        const padding = buffer_desc.size - (index_data.items.len * @sizeOf(u32));
        const padding_data = try self.allocator.alloc(u32, padding / @sizeOf(u32));
        @memset(padding_data, 0);
        defer self.allocator.free(padding_data);
        try index_data.appendSlice(padding_data);
    }

    // First submit the write operation
    self.gfx.queue.writeBuffer(self.index_buffer, 0, u32, index_data.items);
}

pub fn run(self: *App) !void {
    // Create a proper view matrix using lookAt
    // Position camera further away and at an angle for isometric-like view
    const eye_pos = zmath.f32x4(3.0, 3.0, 5.0, 1.0); // Moved back and up for isometric view
    const target_pos = zmath.f32x4(0.0, 0.0, 0.0, 1.0);
    const up_vec = zmath.f32x4(0.0, 1.0, 0.0, 0.0);

    // Create right-handed view matrix
    self.my_uniforms.view = zmath.lookAtRh(eye_pos, target_pos, up_vec);

    const ratio = 640.0 / 480.0;

    // Projection parameters
    const near = 0.1;
    const far = 20.0; // Increased far plane for larger viewing distance
    // Use the perspectiveFovRh function which creates a right-handed perspective matrix
    // with depth range [0,1] appropriate for WebGPU (unlike OpenGL's [-1,1])
    self.my_uniforms.projection = zmath.perspectiveFovRh(
        std.math.pi / 4.0,
        ratio,
        near,
        far,
    );
    // The above generates a right-handed perspective matrix with depth range [0,1]

    // Main render loop
    while (!self.window.shouldClose()) {
        self.gfx.device.tick();
        zglfw.pollEvents();

        const time = @as(f32, @floatCast(self.gfx.stats.time));
        self.my_uniforms.time = time;

        // Set orbital radius
        const orbit_radius = 2.0;

        // Calculate orbital angle based on time
        const orbit_angle = time * 0.5; // Orbital speed

        // Calculate position for orbit around Z axis (motion in XY plane)
        const orbit_x = orbit_radius * @cos(orbit_angle);
        const orbit_y = orbit_radius * @sin(orbit_angle); // Changed from Z to Y for XY plane motion

        // Start with identity matrix
        var model = zmath.identity();

        // 1. First apply scale
        model = zmath.mul(model, zmath.scaling(0.9, 0.9, 0.9));

        // 2. Apply self-rotation around Z axis (as specified in C code)
        // This is the rotation in XY plane
        model = zmath.mul(model, zmath.rotationZ(time));

        // 3. Apply orientation to make pyramid point up (Y axis)
        model = zmath.mul(model, zmath.rotationX(-std.math.pi / 2.0));

        // 4. Translate to orbital position in XY plane
        model = zmath.mul(model, zmath.translation(orbit_x, orbit_y, 0.0));

        self.my_uniforms.model = model;

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
        pass.setIndexBuffer(self.index_buffer, .uint32, 0, self.index_buffer.getSize());

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

    const normal_attribute = zgpu.wgpu.VertexAttribute{
        .shader_location = 1,
        .format = .float32x3,
        .offset = 3 * @sizeOf(f32),
    };

    const color_attribute = zgpu.wgpu.VertexAttribute{
        .shader_location = 2,
        .format = .float32x3,
        .offset = 3 * @sizeOf(f32),
    };

    const vertex_buffer_layout = zgpu.wgpu.VertexBufferLayout{
        .array_stride = 9 * @sizeOf(f32),
        .attribute_count = 3,
        .attributes = &[_]zgpu.wgpu.VertexAttribute{
            position_attribute,
            normal_attribute,
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

// Helper function to process OBJ data and convert it to our expected format
fn processObjData(model: obj.ObjData, point_data: *std.ArrayList(f32), index_data: *std.ArrayList(u32)) !void {
    // Clear existing data
    point_data.clearRetainingCapacity();
    index_data.clearRetainingCapacity();

    // Process each mesh in the OBJ model
    for (model.meshes) |mesh| {
        var face_start: usize = 0;

        // Process each face in the mesh (using num_vertices to determine faces)
        for (mesh.num_vertices) |num_verts_in_face| {
            // Handle triangles and quads (or faces with more vertices)
            for (1..num_verts_in_face - 1) |i| {
                // For each triangle in the face, process 3 vertices
                // First vertex is always at face_start
                // The other two form the triangle (like a triangle fan)
                const indices_to_process = [_]usize{ face_start, face_start + i, face_start + i + 1 };

                for (indices_to_process) |idx| {
                    const mesh_index = mesh.indices[idx];

                    // Get position data if available
                    var px: f32 = 0.0;
                    var py: f32 = 0.0;
                    var pz: f32 = 0.0;

                    if (mesh_index.vertex) |vertex_idx| {
                        if (vertex_idx * 3 + 2 < model.vertices.len) {
                            // OBJ uses Y-up convention, but our code uses Z-up
                            // So we need to convert: (x, y, z) -> (x, z, -y)
                            px = model.vertices[vertex_idx * 3]; // x stays the same
                            // Swap y and z and negate new y (previously z) for orientation
                            py = -model.vertices[vertex_idx * 3 + 2]; // new y = old z
                            pz = model.vertices[vertex_idx * 3 + 1]; // new z = -old y
                        }
                    }

                    // Get normal data if available
                    var nx: f32 = 0.0;
                    var ny: f32 = 0.0;
                    var nz: f32 = 0.0;

                    if (mesh_index.normal) |normal_idx| {
                        if (normal_idx * 3 + 2 < model.normals.len) {
                            // Also transform normal vectors using the same conversion
                            nx = model.normals[normal_idx * 3]; // x stays the same
                            ny = -model.normals[normal_idx * 3 + 2]; // new y = old z
                            nz = model.normals[normal_idx * 3 + 1]; // new z = -old y
                        }
                    }

                    // Use white as default color (you could use materials if needed)
                    const r: f32 = 1.0;
                    const g: f32 = 1.0;
                    const b: f32 = 1.0;

                    // Add position, normal, and color to the point data
                    try point_data.appendSlice(&[_]f32{ px, py, pz, nx, ny, nz, r, g, b });

                    // Add index - we're creating a new vertex for each vertex, so indices are sequential
                    try index_data.append(@as(u32, @intCast(point_data.items.len / 9 - 1)));
                }
            }

            // Move to the next face
            face_start += num_verts_in_face;
        }
    }
}
