const std = @import("std");
const zglfw = @import("zglfw");
const zgpu = @import("zgpu");

const App = @This();

allocator: std.mem.Allocator,
window: *zglfw.Window,
gfx: *zgpu.GraphicsContext,

pub fn init(allocator: std.mem.Allocator) !*App {
    try zglfw.init();
    zglfw.windowHint(.client_api, .no_api);
    zglfw.windowHint(.resizable, false);

    const window = try zglfw.createWindow(640, 480, "Learn WebGPU", null);

    const gctx = try zgpu.GraphicsContext.create(
        allocator,
        .{
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
        },
        .{}, // default context creation options
    );

    const app = try allocator.create(App);
    app.* = App{
        .allocator = allocator,
        .window = window,
        .gfx = gctx,
    };
    return app;
}

pub fn deinit(self: *App) void {
    self.gfx.destroy(self.allocator);

    zglfw.destroyWindow(self.window);
    zglfw.terminate();
    self.allocator.destroy(self);
}

pub fn run(self: *App) void {
    while (!self.window.shouldClose()) {
        zglfw.pollEvents();

        // render your things here

        self.window.swapBuffers();
    }
}
