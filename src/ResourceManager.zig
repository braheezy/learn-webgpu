const std = @import("std");
const zgpu = @import("zgpu");

const ResourceManager = @This();

pub fn loadGeometry(
    data: []const u8,
    point_data: *std.ArrayList(f32),
    index_data: *std.ArrayList(u16),
    dimensions: usize,
) !void {
    // Clear existing data
    point_data.clearRetainingCapacity();
    index_data.clearRetainingCapacity();

    const Section = enum {
        none,
        points,
        indices,
    };
    var current_section: Section = .none;

    // Create a buffered reader
    var buf_reader = std.io.fixedBufferStream(data);
    var reader = buf_reader.reader();
    var line_buf: [1024]u8 = undefined;

    // Read the file line by line
    while (try reader.readUntilDelimiterOrEof(&line_buf, '\n')) |line| {
        // Trim whitespace and CR
        const trimmed = std.mem.trimRight(u8, line, &[_]u8{ ' ', '\r' });
        if (trimmed.len == 0) continue;

        // Skip comments
        if (trimmed[0] == '#') continue;

        // Check section headers
        if (std.mem.eql(u8, trimmed, "[points]")) {
            current_section = .points;
            continue;
        }
        if (std.mem.eql(u8, trimmed, "[indices]")) {
            current_section = .indices;
            continue;
        }

        // Process data based on current section
        var iterator = std.mem.tokenizeScalar(u8, trimmed, ' ');
        switch (current_section) {
            .points => {
                // Read 5 float values (x, y, z, r, g, b)
                for (0..dimensions + 3) |_| {
                    const token = iterator.next() orelse return error.InvalidFormat;
                    const value = try std.fmt.parseFloat(f32, token);
                    try point_data.append(value);
                }
            },
            .indices => {
                // Read 3 index values
                inline for (0..3) |_| {
                    const token = iterator.next() orelse return error.InvalidFormat;
                    const value = try std.fmt.parseInt(u16, token, 10);
                    try index_data.append(value);
                }
            },
            .none => continue,
        }
    }

    std.debug.print("point_data.items.len: {d}\n", .{point_data.items.len});
    std.debug.print("index_data.items.len: {d}\n", .{index_data.items.len});
}

pub fn loadShaderModule(al: std.mem.Allocator, path: []const u8, device: zgpu.wgpu.Device) !zgpu.wgpu.ShaderModule {
    // open file
    std.debug.print("path: {s}\n", .{path});
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
