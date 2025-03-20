const std = @import("std");
const zgpu = @import("zgpu");
const obj = @import("obj");
const jpeg = @import("zjpeg");

const ResourceManager = @This();

pub const VertexAttr = struct {
    position: [3]f32,
    normal: [3]f32,
    color: [3]f32,
    uv: [2]f32,
};

pub fn loadGeometryFromObj(
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.ArrayList(VertexAttr) {
    // open file
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    // read file
    const obj_file_contents = try file.readToEndAlloc(allocator, 1024 * 1024 * 10);
    defer allocator.free(obj_file_contents);

    // Load the OBJ model instead of using the text file
    var obj_model = try obj.parseObj(allocator, obj_file_contents);
    defer obj_model.deinit(allocator);

    var vertex_data = std.ArrayList(VertexAttr).init(allocator);
    errdefer vertex_data.deinit();

    // Process each mesh in the OBJ model
    for (obj_model.meshes) |mesh| {
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
                        if (vertex_idx * 3 + 2 < obj_model.vertices.len) {
                            // OBJ uses Y-up convention, but our code uses Z-up
                            px = obj_model.vertices[vertex_idx * 3];
                            py = -obj_model.vertices[vertex_idx * 3 + 2];
                            pz = obj_model.vertices[vertex_idx * 3 + 1];
                        }
                    }

                    // Get normal data if available
                    var nx: f32 = 0.0;
                    var ny: f32 = 0.0;
                    var nz: f32 = 0.0;

                    if (mesh_index.normal) |normal_idx| {
                        if (normal_idx * 3 + 2 < obj_model.normals.len) {
                            nx = obj_model.normals[normal_idx * 3];
                            ny = -obj_model.normals[normal_idx * 3 + 2];
                            nz = obj_model.normals[normal_idx * 3 + 1];
                        }
                    }

                    // Use white as default color
                    const r: f32 = 1.0;
                    const g: f32 = 1.0;
                    const b: f32 = 1.0;

                    // Get texture coordinates if available - THIS IS CRITICAL
                    var u: f32 = 0.0;
                    var v: f32 = 0.0;

                    if (mesh_index.tex_coord) |uv_idx| {
                        if (uv_idx * 2 + 1 < obj_model.tex_coords.len) {
                            // OBJ format stores UV with bottom-left origin (0,0)
                            // Make sure U is clamped to [0,1] range
                            u = @max(0.0, @min(1.0, obj_model.tex_coords[uv_idx * 2]));
                            // Flip V coordinate as OBJ format uses bottom-left origin
                            // and we want top-left origin for WebGPU
                            v = 1.0 - @max(0.0, @min(1.0, obj_model.tex_coords[uv_idx * 2 + 1]));

                            // Debug print - uncomment if needed
                            // std.debug.print("UV: ({d}, {d})\n", .{ u, v });
                        }
                    }

                    // Add position, normal, color and UV to the point data
                    try vertex_data.append(VertexAttr{
                        .position = [3]f32{ px, py, pz },
                        .normal = [3]f32{ nx, ny, nz },
                        .color = [3]f32{ r, g, b },
                        .uv = [2]f32{ u, v },
                    });
                }
            }

            // Move to the next face
            face_start += num_verts_in_face;
        }
    }

    return vertex_data;
}

pub fn loadShaderModule(al: std.mem.Allocator, path: []const u8, device: zgpu.wgpu.Device) !zgpu.wgpu.ShaderModule {
    // open file
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    // read file
    const contents = try file.readToEndAllocOptions(
        al,
        1024 * 16,
        null,
        @alignOf(u8),
        0,
    );
    defer al.free(contents);

    return zgpu.createWgslShaderModule(device, contents, null);
}

pub fn loadTexture(
    allocator: std.mem.Allocator,
    gfx: *zgpu.GraphicsContext,
    path: []const u8,
    texture_view: *?zgpu.TextureViewHandle,
) !zgpu.TextureHandle {
    const image = try jpeg.load(allocator, path);
    defer image.free(allocator);

    const bounds = image.bounds();
    const width: u32 = @intCast(bounds.dX());
    const height: u32 = @intCast(bounds.dY());

    const texture_pixels = try image.rgbaPixels(allocator);
    defer allocator.free(texture_pixels);

    const mip_level_count = bitWidth(@max(width, height));

    const texture_desc = zgpu.wgpu.TextureDescriptor{
        .dimension = .tdim_2d,
        .format = .rgba8_unorm,
        .mip_level_count = mip_level_count,
        .sample_count = 1,
        .size = .{
            .width = width,
            .height = height,
            .depth_or_array_layers = 1,
        },
        .usage = .{ .texture_binding = true, .copy_dst = true },
        .view_format_count = 0,
        .view_formats = null,
    };

    const texture = gfx.createTexture(texture_desc);

    if (texture_view.*) |_| {
        // do nothing if exists
    } else {
        texture_view.* = gfx.createTextureView(texture, .{
            .aspect = .all,
            .base_array_layer = 0,
            .array_layer_count = 1,
            .base_mip_level = 0,
            .mip_level_count = texture_desc.mip_level_count,
            .dimension = .tvdim_2d,
            .format = texture_desc.format,
        });
    }

    writeMipMaps(
        allocator,
        gfx,
        texture,
        texture_desc.size,
        texture_pixels,
    );

    return texture;
}

fn writeMipMaps(
    allocator: std.mem.Allocator,
    gfx: *zgpu.GraphicsContext,
    texture_handle: zgpu.TextureHandle,
    texture_size: zgpu.wgpu.Extent3D,
    texture_pixels: []u8,
) void {
    const texture = gfx.lookupResource(texture_handle) orelse unreachable;

    // Arguments telling which part of the texture to upload
    var destination = zgpu.wgpu.ImageCopyTexture{
        .texture = texture,
        .mip_level = 0,
        .origin = .{ .x = 0, .y = 0, .z = 0 },
        .aspect = .all,
    };

    var mip_level_size = texture_size;
    var previous_level_pixels: ?[]u8 = null;

    // Calculate number of mip levels based on the largest dimension
    const max_dimension = @max(texture_size.width, texture_size.height);
    const mip_level_count = bitWidth(max_dimension);

    var level: u32 = 0;
    while (level < mip_level_count) : (level += 1) {
        // Calculate dimensions for this mip level
        const width = texture_size.width >> @intCast(level);
        const height = texture_size.height >> @intCast(level);

        // Calculate bytes per row with proper alignment (256-byte alignment for WebGPU)
        const bytes_per_row = (4 * width + 255) & ~@as(u32, 255);

        // Allocate space for current mip level with proper row alignment
        const row_pitch = bytes_per_row;
        const buffer_size = row_pitch * height;
        const pixels = allocator.alloc(u8, buffer_size) catch break;
        defer allocator.free(pixels);

        // Clear the buffer first
        @memset(pixels, 0);

        if (level == 0) {
            // For the first level, copy the input texture data row by row to handle alignment
            var y: usize = 0;
            while (y < height) : (y += 1) {
                const src_offset = y * width * 4;
                const dst_offset = y * row_pitch;
                const row_bytes = width * 4;
                @memcpy(pixels[dst_offset..][0..row_bytes], texture_pixels[src_offset..][0..row_bytes]);
            }
        } else {
            // Generate mip level data from previous level
            const prev_width = texture_size.width >> @intCast(level - 1);
            for (0..height) |j| {
                for (0..width) |i| {
                    const dst_offset = j * row_pitch + i * 4;

                    // Calculate source pixels from previous level
                    const src_x = i * 2;
                    const src_y = j * 2;
                    const prev_row_pitch = (4 * prev_width + 255) & ~@as(u32, 255);

                    const p00_idx = src_y * prev_row_pitch + src_x * 4;
                    const p01_idx = src_y * prev_row_pitch + (src_x + 1) * 4;
                    const p10_idx = (src_y + 1) * prev_row_pitch + src_x * 4;
                    const p11_idx = (src_y + 1) * prev_row_pitch + (src_x + 1) * 4;

                    // Average each color component
                    inline for (0..4) |component| {
                        const sum = @as(u16, previous_level_pixels.?[p00_idx + component]) +
                            @as(u16, previous_level_pixels.?[p01_idx + component]) +
                            @as(u16, previous_level_pixels.?[p10_idx + component]) +
                            @as(u16, previous_level_pixels.?[p11_idx + component]);
                        pixels[dst_offset + component] = @truncate(sum / 4);
                    }
                }
            }
        }

        // Upload the mip level to GPU
        destination.mip_level = level;

        // describes the layout of the data in the buffer
        const data_layout = zgpu.wgpu.TextureDataLayout{
            .offset = 0,
            .bytes_per_row = bytes_per_row,
            .rows_per_image = height,
        };

        mip_level_size.width = width;
        mip_level_size.height = height;

        gfx.queue.writeTexture(
            destination,
            data_layout,
            mip_level_size,
            u8,
            pixels,
        );

        // Update for next iteration
        if (previous_level_pixels) |prev_pixels| {
            allocator.free(prev_pixels);
        }
        previous_level_pixels = allocator.dupe(u8, pixels) catch break;
    }

    // Clean up
    if (previous_level_pixels) |prev_pixels| {
        allocator.free(prev_pixels);
    }
}

fn bitWidth(m: u32) u32 {
    if (m == 0) return 0;
    var width: u32 = 0;
    var value = m;
    while (value > 0) : (width += 1) {
        value >>= 1;
    }
    return width;
}
