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
texture: zgpu.TextureHandle = undefined,
texture_view: zgpu.TextureViewHandle = undefined,
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
            .max_inter_stage_shader_components = 6,
            .max_bind_groups = 1,
            .max_uniform_buffers_per_shader_stage = 1,
            .max_uniform_buffer_binding_size = 16 * 4 * @sizeOf(f32),
            .max_dynamic_uniform_buffers_per_pipeline_layout = 1,
            .max_texture_dimension_1d = 480,
            .max_texture_dimension_2d = 640,
            .max_texture_array_layers = 1,
            .max_sampled_textures_per_shader_stage = 1,
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
    self.gfx.releaseResource(self.texture);
    self.gfx.destroyResource(self.texture);

    self.gfx.destroy(self.allocator);

    zglfw.destroyWindow(self.window);
    zglfw.terminate();
    self.allocator.destroy(self);
}

fn initializeBuffers(self: *App) !void {
    // Load the OBJ model instead of using the text file
    var obj_model = try obj.parseObj(self.allocator, @embedFile("resources/plane.obj"));
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

    // Create right-handed view matrix
    self.my_uniforms.view = zmath.scaling(1.0, 1.0, 1.0);

    self.my_uniforms.projection = zmath.orthographicRh(2, 2, -1, 1);

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

    // Consistently get texture size from where we defined it
    const texture: zgpu.wgpu.Texture = self.gfx.lookupResource(self.texture) orelse unreachable;
    const texture_width: u32 = 512;
    const texture_height: u32 = 512;

    const texure_pixels = try makePixels(self.allocator, texture_width, texture_height);
    defer self.allocator.free(texure_pixels);

    const destination = zgpu.wgpu.ImageCopyTexture{
        .texture = texture,
        .mip_level = 0,
        .origin = .{ .x = 0, .y = 0, .z = 0 },
        .aspect = .all,
    };

    const data_layout = zgpu.wgpu.TextureDataLayout{
        .offset = 0,
        .bytes_per_row = 4 * texture_width,
        .rows_per_image = texture_height,
    };

    const copy_size = zgpu.wgpu.Extent3D{
        .width = texture_width,
        .height = texture_height,
        .depth_or_array_layers = 1,
    };

    self.gfx.queue.writeTexture(
        destination,
        data_layout,
        copy_size,
        u8,
        texure_pixels,
    );

    const pipeline = self.gfx.lookupResource(self.pipeline) orelse unreachable;
    const bind_group = self.gfx.lookupResource(self.bind_group) orelse unreachable;

    // Main render loop
    while (!self.window.shouldClose()) {
        self.gfx.device.tick();
        zglfw.pollEvents();

        const time = @as(f32, @floatCast(self.gfx.stats.time));
        self.my_uniforms.time = time;

        self.my_uniforms.model = zmath.identity();

        // Allocate and update the entire uniform struct
        const uni_mem = self.gfx.uniformsAllocate(MyUniforms, 1);
        uni_mem.slice[0] = self.my_uniforms;

        const view = self.gfx.swapchain.getCurrentTextureView();
        defer view.release();

        const color_attachment = [_]zgpu.wgpu.RenderPassColorAttachment{.{
            .view = view,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0.05, .g = 0.05, .b = 0.05, .a = 1.0 },
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
        .{
            .binding = 1,
            .visibility = .{ .fragment = true },
            .texture = .{
                .sample_type = .float,
                .view_dimension = .tvdim_2d,
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

    const texture_desc = zgpu.wgpu.TextureDescriptor{
        .dimension = .tdim_2d,
        .format = .rgba8_unorm,
        .mip_level_count = 1,
        .sample_count = 1,
        .size = .{
            .width = self.gfx.swapchain_descriptor.width,
            .height = self.gfx.swapchain_descriptor.height,
            .depth_or_array_layers = 1,
        },
        .usage = .{ .texture_binding = true, .copy_dst = true },
        .view_format_count = 0,
        .view_formats = null,
    };

    self.texture = self.gfx.createTexture(texture_desc);

    self.texture_view = self.gfx.createTextureView(self.texture, .{
        .aspect = .all,
        .base_array_layer = 0,
        .array_layer_count = 1,
        .base_mip_level = 0,
        .mip_level_count = 1,
        .dimension = .tvdim_2d,
        .format = .rgba8_unorm,
    });

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
    });
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

/// Creates a slice of pixels for texture data
/// Caller is responsible for freeing the returned slice with allocator.free()
fn makePixels(allocator: std.mem.Allocator, width: u32, height: u32) ![]u8 {
    // Allocate memory for the pixel data (4 bytes per pixel: R, G, B, A)
    const pixel_count = width * height;
    const pixels = try allocator.alloc(u8, 4 * pixel_count);

    // Fill the pixel data
    for (0..width) |i_unsigned| {
        for (0..height) |j_unsigned| {
            // Convert to signed to avoid underflow
            const i: i32 = @intCast(i_unsigned);
            const j: i32 = @intCast(j_unsigned);

            // Calculate the pixel index in the buffer
            const pixel_index = 4 * (j_unsigned * width + i_unsigned);

            // Set RGBA values using signed arithmetic
            pixels[pixel_index + 0] = if (@mod(@divFloor(i, 16), 2) == @mod(@divFloor(j, 16), 2)) 255 else 0;
            pixels[pixel_index + 1] = if (@mod(@divFloor(i - j, 16), 2) == 0) 255 else 0;
            pixels[pixel_index + 2] = if (@mod(@divFloor(i + j, 16), 2) == 0) 255 else 0;
            pixels[pixel_index + 3] = 255; // A - fully opaque
        }
    }

    return pixels;
}
