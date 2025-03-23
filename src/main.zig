const std = @import("std");

const App = @import("App.zig");

pub fn main() !void {
    // Memory allocation setup
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer if (gpa.deinit() == .leak) {
        std.process.exit(1);
    };

    const app = try App.init(allocator);
    defer app.deinit();

    while (app.isRunning()) {
        try app.update();
        try app.draw();
    }
}
