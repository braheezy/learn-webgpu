const std = @import("std");
const zgui = @import("zgui");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");

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
    width: u32,
    height: u32,
) void {
    // Start ImGui frame before any rendering
    zgui.backend.newFrame(
        width,
        height,
    );

    if (zgui.begin("Hello, world!", .{})) {
        zgui.text("This is some useful text.", .{});
        _ = zgui.checkbox("Demo Window", .{ .v = &gui_show_demo_window });
        _ = zgui.checkbox("Another Window", .{ .v = &gui_show_another_window });

        _ = zgui.sliderFloat("float", .{
            .v = &gui_f,
            .min = 0.0,
            .max = 1.0,
        });
        _ = zgui.colorEdit3("clear color", .{ .col = gui_clear_color[0..3] });

        if (zgui.button("Button", .{})) {
            gui_counter += 1;
        }
        zgui.sameLine(.{});
        zgui.text("counter = {d}", .{gui_counter});

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
