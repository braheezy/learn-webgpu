const std = @import("std");

const App = @import("App.zig");

pub fn main(init: std.process.Init) !void {
    const app = try App.init(init.gpa, init.io);
    defer app.deinit();

    while (app.isRunning()) {
        try app.update();
        try app.draw();
    }
}
