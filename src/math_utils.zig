const std = @import("std");

/// Converts a 3D Cartesian vector to spherical coordinates
/// Returns [theta, phi] where:
/// - theta: azimuthal angle in the x-y plane from the x-axis (in radians)
/// - phi: polar angle from the z-axis (in radians)
pub fn polar(cartesian: [3]f32) [2]f32 {
    // Calculate theta (azimuthal angle in x-y plane)
    const theta = std.math.atan2(cartesian[1], cartesian[0]);

    // Calculate r (length of the vector)
    const r = std.math.sqrt(cartesian[0] * cartesian[0] +
        cartesian[1] * cartesian[1] +
        cartesian[2] * cartesian[2]);

    // Calculate phi (polar angle from z-axis)
    // Handle the case when r is zero to avoid division by zero
    var phi: f32 = 0.0;
    if (r > 0.0) {
        phi = std.math.acos(cartesian[2] / r);
    }

    return .{ theta, phi };
}

/// Converts spherical coordinates to a 3D Cartesian vector
/// Input: [theta, phi] where:
/// - theta: azimuthal angle in the x-y plane from the x-axis (in radians)
/// - phi: polar angle from the z-axis (in radians)
/// Returns a unit vector [x, y, z]
pub fn euclidean(angles: [2]f32) [3]f32 {
    const theta = angles[0];
    const phi = angles[1];

    return .{
        std.math.sin(phi) * std.math.cos(theta),
        std.math.sin(phi) * std.math.sin(theta),
        std.math.cos(phi),
    };
}

/// Converts radians to degrees
pub fn degrees(rad: f32) f32 {
    return rad * 180.0 / std.math.pi;
}

/// Converts degrees to radians
pub fn radians(deg: f32) f32 {
    return deg * std.math.pi / 180.0;
}

/// Converts an array of angles from radians to degrees
pub fn degreesVec(rad_vec: [2]f32) [2]f32 {
    return .{ degrees(rad_vec[0]), degrees(rad_vec[1]) };
}

/// Converts an array of angles from degrees to radians
pub fn radiansVec(deg_vec: [2]f32) [2]f32 {
    return .{ radians(deg_vec[0]), radians(deg_vec[1]) };
}

pub fn toRadians(deg: f32) f32 {
    return deg * std.math.pi / 180.0;
}
