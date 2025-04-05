const std = @import("std");
const assert = std.debug.assert;
const zgpu_options = @import("zgpu_options");

const wgpu = @import("wgpu");

pub const WindowProvider = struct {
    window: *anyopaque,
    fn_getTime: *const fn () f64,
    fn_getFramebufferSize: *const fn (window: *const anyopaque) [2]u32,
    fn_getWin32Window: *const fn (window: *const anyopaque) ?*anyopaque = undefined,
    fn_getX11Display: *const fn () ?*anyopaque = undefined,
    fn_getX11Window: *const fn (window: *const anyopaque) u32 = undefined,
    fn_getWaylandDisplay: ?*const fn () ?*anyopaque = null,
    fn_getWaylandSurface: ?*const fn (window: *const anyopaque) ?*anyopaque = null,
    fn_getCocoaWindow: *const fn (window: *const anyopaque) ?*anyopaque = undefined,

    fn getTime(self: WindowProvider) f64 {
        return self.fn_getTime();
    }

    fn getFramebufferSize(self: WindowProvider) [2]u32 {
        return self.fn_getFramebufferSize(self.window);
    }

    fn getWin32Window(self: WindowProvider) ?*anyopaque {
        return self.fn_getWin32Window(self.window);
    }

    fn getX11Display(self: WindowProvider) ?*anyopaque {
        return self.fn_getX11Display();
    }

    fn getX11Window(self: WindowProvider) u32 {
        return self.fn_getX11Window(self.window);
    }

    fn getWaylandDisplay(self: WindowProvider) ?*anyopaque {
        if (self.fn_getWaylandDisplay) |f| {
            return f();
        } else {
            return @as(?*anyopaque, null);
        }
    }

    fn getWaylandSurface(self: WindowProvider) ?*anyopaque {
        if (self.fn_getWaylandSurface) |f| {
            return f(self.window);
        } else {
            return @as(?*anyopaque, null);
        }
    }

    fn getCocoaWindow(self: WindowProvider) ?*anyopaque {
        return self.fn_getCocoaWindow(self.window);
    }
};

pub const GraphicsContextOptions = struct {
    present_mode: wgpu.PresentMode = .fifo,
    required_features: []const wgpu.FeatureName = &.{},
    required_limits: ?*const wgpu.RequiredLimits = null,
};

pub const GraphicsContext = struct {
    window_provider: WindowProvider,

    instance: *wgpu.Instance,

    pub fn create(
        allocator: std.mem.Allocator,
        window_provider: WindowProvider,
        options: GraphicsContextOptions,
    ) !*GraphicsContext {
        const instance = wgpu.Instance.create(null) orelse return error.NoGraphicsInstance;

        const adapter = adapter: {
            const Response = struct {
                status: wgpu.RequestAdapterStatus = .unknown,
                adapter: wgpu.Adapter = undefined,
            };

            const callback = (struct {
                fn callback(
                    status: wgpu.RequestAdapterStatus,
                    adapter: wgpu.Adapter,
                    message: ?[*:0]const u8,
                    userdata: ?*anyopaque,
                ) callconv(.C) void {
                    _ = message;
                    const response = @as(*Response, @ptrCast(@alignCast(userdata)));
                    response.status = status;
                    response.adapter = adapter;
                }
            }).callback;

            var response = Response{};
            instance.requestAdapter(
                .{ .power_preference = .high_performance },
                callback,
                @ptrCast(&response),
            );

            if (response.status != .success) {
                std.log.err("Failed to request GPU adapter (status: {s}).", .{@tagName(response.status)});
                return error.NoGraphicsAdapter;
            }
            break :adapter response.adapter;
        };
        errdefer adapter.release();

        var properties: wgpu.AdapterProperties = undefined;
        properties.next_in_chain = null;
        adapter.getProperties(&properties);

        std.log.info("[zgpu] High-performance device has been selected:", .{});
        std.log.info("[zgpu]   Name: {s}", .{properties.name});
        std.log.info("[zgpu]   Driver: {s}", .{properties.driver_description});
        std.log.info("[zgpu]   Adapter type: {s}", .{@tagName(properties.adapter_type)});
        std.log.info("[zgpu]   Backend type: {s}", .{@tagName(properties.backend_type)});

        const device = device: {
            const Response = struct {
                status: wgpu.RequestDeviceStatus = .unknown,
                device: wgpu.Device = undefined,
            };

            const callback = (struct {
                fn callback(
                    status: wgpu.RequestDeviceStatus,
                    device: wgpu.Device,
                    message: ?[*:0]const u8,
                    userdata: ?*anyopaque,
                ) callconv(.C) void {
                    _ = message;
                    const response = @as(*Response, @ptrCast(@alignCast(userdata)));
                    response.status = status;
                    response.device = device;
                }
            }).callback;

            var toggles: [2][*:0]const u8 = undefined;
            var num_toggles: usize = 0;
            if (zgpu_options.dawn_skip_validation) {
                toggles[num_toggles] = "skip_validation";
                num_toggles += 1;
            }
            if (zgpu_options.dawn_allow_unsafe_apis) {
                toggles[num_toggles] = "allow_unsafe_apis";
                num_toggles += 1;
            }
            const dawn_toggles = wgpu.DawnTogglesDescriptor{
                .chain = .{ .next = null, .struct_type = .dawn_toggles_descriptor },
                .enabled_toggles_count = num_toggles,
                .enabled_toggles = &toggles,
            };

            var response = Response{};
            adapter.requestDevice(
                wgpu.DeviceDescriptor{
                    .next_in_chain = @ptrCast(&dawn_toggles),
                    .required_features_count = options.required_features.len,
                    .required_features = options.required_features.ptr,
                    .required_limits = @ptrCast(options.required_limits),
                },
                callback,
                @ptrCast(&response),
            );

            if (response.status != .success) {
                std.log.err("Failed to request GPU device (status: {s}).", .{@tagName(response.status)});
                return error.NoGraphicsDevice;
            }
            break :device response.device;
        };
        errdefer device.release();

        device.setUncapturedErrorCallback(logUnhandledError, null);

        const surface = createSurfaceForWindow(instance, window_provider);
        errdefer surface.release();

        const framebuffer_size = window_provider.getFramebufferSize();

        const swapchain_descriptor = wgpu.SwapChainDescriptor{
            .label = "zig-gamedev-gctx-swapchain",
            .usage = .{ .render_attachment = true },
            .format = swapchain_format,
            .width = @intCast(framebuffer_size[0]),
            .height = @intCast(framebuffer_size[1]),
            .present_mode = options.present_mode,
        };
        const swapchain = device.createSwapChain(surface, swapchain_descriptor);
        errdefer swapchain.release();

        const gctx = try allocator.create(GraphicsContext);
        gctx.* = .{
            .window_provider = window_provider,
            .native_instance = if (emscripten) null else native_instance,
            .instance = instance,
            .device = device,
            .queue = device.getQueue(),
            .surface = surface,
            .swapchain = swapchain,
            .swapchain_descriptor = swapchain_descriptor,
            .buffer_pool = BufferPool.init(allocator, zgpu_options.buffer_pool_size),
            .texture_pool = TexturePool.init(allocator, zgpu_options.texture_pool_size),
            .texture_view_pool = TextureViewPool.init(allocator, zgpu_options.texture_view_pool_size),
            .sampler_pool = SamplerPool.init(allocator, zgpu_options.sampler_pool_size),
            .render_pipeline_pool = RenderPipelinePool.init(allocator, zgpu_options.render_pipeline_pool_size),
            .compute_pipeline_pool = ComputePipelinePool.init(allocator, zgpu_options.compute_pipeline_pool_size),
            .bind_group_pool = BindGroupPool.init(allocator, zgpu_options.bind_group_pool_size),
            .bind_group_layout_pool = BindGroupLayoutPool.init(allocator, zgpu_options.bind_group_layout_pool_size),
            .pipeline_layout_pool = PipelineLayoutPool.init(allocator, zgpu_options.pipeline_layout_pool_size),
            .mipgens = std.AutoHashMap(wgpu.TextureFormat, MipgenResources).init(allocator),
        };

        uniformsInit(gctx);
        return gctx;
    }
};

fn logUnhandledError(
    err_type: wgpu.ErrorType,
    message: ?[*:0]const u8,
    userdata: ?*anyopaque,
) callconv(.C) void {
    _ = userdata;
    switch (err_type) {
        .no_error => std.log.info("[zgpu] No error: {?s}", .{message}),
        .validation => std.log.err("[zgpu] Validation: {?s}", .{message}),
        .out_of_memory => std.log.err("[zgpu] Out of memory: {?s}", .{message}),
        .device_lost => std.log.err("[zgpu] Device lost: {?s}", .{message}),
        .internal => std.log.err("[zgpu] Internal error: {?s}", .{message}),
        .unknown => std.log.err("[zgpu] Unknown error: {?s}", .{message}),
    }

    // Exit the process for easier debugging.
    if (@import("builtin").mode == .Debug)
        std.process.exit(1);
}

fn createSurfaceForWindow(instance: wgpu.Instance, window_provider: WindowProvider) wgpu.Surface {
    const os_tag = @import("builtin").target.os.tag;

    const descriptor = switch (os_tag) {
        .windows => wgpu.SurfaceDescriptor{
            .windows_hwnd = .{
                .label = "basic surface",
                .hinstance = std.os.windows.kernel32.GetModuleHandleW(null).?,
                .hwnd = window_provider.getWin32Window().?,
            },
        },
        .macos => macos: {
            const ns_window = window_provider.getCocoaWindow().?;
            const ns_view = msgSend(ns_window, "contentView", .{}, *anyopaque); // [nsWindow contentView]

            // Create a CAMetalLayer that covers the whole window that will be passed to CreateSurface.
            msgSend(ns_view, "setWantsLayer:", .{true}, void); // [view setWantsLayer:YES]
            const layer = msgSend(objc.objc_getClass("CAMetalLayer"), "layer", .{}, ?*anyopaque); // [CAMetalLayer layer]
            if (layer == null) @panic("failed to create Metal layer");
            msgSend(ns_view, "setLayer:", .{layer.?}, void); // [view setLayer:layer]

            // Use retina if the window was created with retina support.
            const scale_factor = msgSend(ns_window, "backingScaleFactor", .{}, f64); // [ns_window backingScaleFactor]
            msgSend(layer.?, "setContentsScale:", .{scale_factor}, void); // [layer setContentsScale:scale_factor]

            break :macos SurfaceDescriptor{
                .metal_layer = .{
                    .label = "basic surface",
                    .layer = layer.?,
                },
            };
        },
        .emscripten => SurfaceDescriptor{
            .canvas_html = .{
                .label = "basic surface",
                .selector = "#canvas", // TODO: can this be somehow exposed through api?
            },
        },
        else => if (isLinuxDesktopLike(os_tag)) linux: {
            if (window_provider.getWaylandDisplay()) |wl_display| {
                break :linux SurfaceDescriptor{
                    .wayland = .{
                        .label = "basic surface",
                        .display = wl_display,
                        .surface = window_provider.getWaylandSurface().?,
                    },
                };
            } else {
                break :linux SurfaceDescriptor{
                    .xlib = .{
                        .label = "basic surface",
                        .display = window_provider.getX11Display().?,
                        .window = window_provider.getX11Window(),
                    },
                };
            }
        } else unreachable,
    };

    return switch (descriptor) {
        .metal_layer => |src| blk: {
            var desc: wgpu.SurfaceDescriptorFromMetalLayer = undefined;
            desc.chain.next = null;
            desc.chain.struct_type = .surface_descriptor_from_metal_layer;
            desc.layer = src.layer;
            break :blk instance.createSurface(.{
                .next_in_chain = @ptrCast(&desc),
                .label = if (src.label) |l| l else null,
            });
        },
        .windows_hwnd => |src| blk: {
            var desc: wgpu.SurfaceDescriptorFromWindowsHWND = undefined;
            desc.chain.next = null;
            desc.chain.struct_type = .surface_descriptor_from_windows_hwnd;
            desc.hinstance = src.hinstance;
            desc.hwnd = src.hwnd;
            break :blk instance.createSurface(.{
                .next_in_chain = @ptrCast(&desc),
                .label = if (src.label) |l| l else null,
            });
        },
        .xlib => |src| blk: {
            var desc: wgpu.SurfaceDescriptorFromXlibWindow = undefined;
            desc.chain.next = null;
            desc.chain.struct_type = .surface_descriptor_from_xlib_window;
            desc.display = src.display;
            desc.window = src.window;
            break :blk instance.createSurface(.{
                .next_in_chain = @ptrCast(&desc),
                .label = if (src.label) |l| l else null,
            });
        },
        .wayland => |src| blk: {
            var desc: wgpu.SurfaceDescriptorFromWaylandSurface = undefined;
            desc.chain.next = null;
            desc.chain.struct_type = .surface_descriptor_from_wayland_surface;
            desc.display = src.display;
            desc.surface = src.surface;
            break :blk instance.createSurface(.{
                .next_in_chain = @ptrCast(&desc),
                .label = if (src.label) |l| l else null,
            });
        },
        .canvas_html => |src| blk: {
            var desc: wgpu.SurfaceDescriptorFromCanvasHTMLSelector = .{
                .chain = .{ .struct_type = .surface_descriptor_from_canvas_html_selector, .next = null },
                .selector = src.selector,
            };
            break :blk instance.createSurface(.{
                .next_in_chain = @as(*const wgpu.ChainedStruct, @ptrCast(&desc)),
                .label = if (src.label) |l| l else null,
            });
        },
    };
}

const objc = struct {
    const SEL = ?*opaque {};
    const Class = ?*opaque {};

    extern fn sel_getUid(str: [*:0]const u8) SEL;
    extern fn objc_getClass(name: [*:0]const u8) Class;
    extern fn objc_msgSend() void;
};

fn msgSend(obj: anytype, sel_name: [:0]const u8, args: anytype, comptime ReturnType: type) ReturnType {
    const args_meta = @typeInfo(@TypeOf(args)).@"struct".fields;

    const FnType = switch (args_meta.len) {
        0 => *const fn (@TypeOf(obj), objc.SEL) callconv(.C) ReturnType,
        1 => *const fn (@TypeOf(obj), objc.SEL, args_meta[0].type) callconv(.C) ReturnType,
        2 => *const fn (
            @TypeOf(obj),
            objc.SEL,
            args_meta[0].type,
            args_meta[1].type,
        ) callconv(.C) ReturnType,
        3 => *const fn (
            @TypeOf(obj),
            objc.SEL,
            args_meta[0].type,
            args_meta[1].type,
            args_meta[2].type,
        ) callconv(.C) ReturnType,
        4 => *const fn (
            @TypeOf(obj),
            objc.SEL,
            args_meta[0].type,
            args_meta[1].type,
            args_meta[2].type,
            args_meta[3].type,
        ) callconv(.C) ReturnType,
        else => @compileError("[zgpu] Unsupported number of args"),
    };

    const func = @as(FnType, @ptrCast(&objc.objc_msgSend));
    const sel = objc.sel_getUid(sel_name.ptr);

    return @call(.never_inline, func, .{ obj, sel } ++ args);
}
