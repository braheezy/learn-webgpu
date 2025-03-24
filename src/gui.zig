const std = @import("std");
const zgui = @import("zgui");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");
const zmath = @import("zmath");
const math_utils = @import("math_utils.zig");

const App = @import("App.zig");
const Lighting = App.Lighting;

var gui_counter: i32 = 0;
var gui_f: f32 = 0.0;
var gui_show_demo_window: bool = true;
var gui_show_another_window: bool = false;
var gui_clear_color: [4]f32 = .{ 0.45, 0.55, 0.60, 1.00 };

pub fn create(
    allocator: std.mem.Allocator,
    window: *zglfw.Window,
    device: zgpu.wgpu.Device,
    swapchain_format: zgpu.wgpu.TextureFormat,
) !void {
    zgui.init(allocator);
    zgui.backend.init(
        window,
        device,
        @intFromEnum(swapchain_format),
        @intFromEnum(zgpu.wgpu.TextureFormat.undef),
    );

    zgui.io.setConfigFlags(.{
        .dpi_enable_scale_fonts = true,
        .dpi_enable_scale_viewport = true,
    });
}

pub fn update(
    app: *App,
) void {
    const width = app.gfx.swapchain_descriptor.width;
    const height = app.gfx.swapchain_descriptor.height;
    const lighting = &app.lighting;

    // Start ImGui frame before any rendering
    zgui.backend.newFrame(
        width,
        height,
    );

    if (zgui.begin("Lighting", .{})) {
        zgui.text("This is some useful text.", .{});
        _ = zgui.colorEdit3("Color: #0", .{ .col = lighting.colors[0][0..3] });
        _ = dragDirection("Direction: #0", &lighting.directions[0]);
        _ = zgui.colorEdit3("Color: #1", .{ .col = lighting.colors[1][0..3] });
        _ = zgui.dragFloat3("Direction: #1", .{ .v = lighting.directions[1][0..3] });
        _ = zgui.sliderFloat("Hardness", .{ .v = &lighting.hardness, .min = 0.0, .max = 100.0 });
        _ = zgui.sliderFloat("K Diffuse", .{ .v = &lighting.kd, .min = 0.0, .max = 1.0 });
        _ = zgui.sliderFloat("K Specular", .{ .v = &lighting.ks, .min = 0.0, .max = 1.0 });

        var result: bool = lighting.enable_gamma != 0;
        if (zgui.checkbox("Enable Gamma Correction", .{ .v = &result })) {
            lighting.enable_gamma = if (result) 1 else 0;
        }
        const framerate = zgui.io.getFramerate();
        zgui.text("Application average {d:.3} ms/frame ({d:.1} FPS)", .{ 1000.0 / framerate, framerate });
    }
    zgui.end();
}

pub fn draw(encoder: zgpu.wgpu.CommandEncoder, view: zgpu.wgpu.TextureView) !void {
    const gui_pass = zgpu.beginRenderPassSimple(
        encoder,
        .load,
        view,
        null,
        null,
        null,
    );
    defer zgpu.endReleasePass(gui_pass);

    zgui.backend.draw(gui_pass);
}

pub fn destroy() void {
    zgui.backend.deinit();
    zgui.deinit();
}

pub fn dragDirection(label: [:0]const u8, direction: *[4]f32) bool {
    // Extract the 3D vector part
    const cartesian = [3]f32{ direction[0], direction[1], direction[2] };

    // Convert to polar coordinates (radians) and then to degrees
    var angles = math_utils.degreesVec(math_utils.polar(cartesian));

    // Create the drag input
    const changed = zgui.dragFloat2(label, .{ .v = &angles });

    if (changed) {
        // Convert back to radians and then to cartesian
        const new_cartesian = math_utils.euclidean(math_utils.radiansVec(angles));

        direction[0] = new_cartesian[0];
        direction[1] = new_cartesian[1];
        direction[2] = new_cartesian[2];
        // direction[3] remains unchanged
    }

    return changed;
}
