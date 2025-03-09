struct VertexInput {
  @location(0) position: vec3f,
  @location(1) color: vec3f,
};
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

// Specify which binding index the uniform is attached to
// The memory location of the uniform is given by a pair of a *bind group* and a *binding*
struct MyUniforms {
    color: vec4f,
    time: f32,
};

// Instead of the simple uTime variable, our uniform variable is a struct
@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let ratio = 640.0 / 480.0;

    // We now move the scene depending on the time!
    var offset = vec2f(0, 0);

    let angle = uMyUniforms.time; // you can multiply it go rotate faster
    let alpha = cos(angle);
    let beta = sin(angle);
    var position = vec3f(
        in.position.x,
        alpha * in.position.y + beta * in.position.z,
        alpha * in.position.z - beta * in.position.y,
    );
    out.position = vec4f(position.x, position.y * ratio, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f  {
    // We multiply the scene's color with our global uniform (this is one
    // possible use of the color uniform, among many others).
    let color = in.color * uMyUniforms.color.rgb;

    // Gamma-correction
    let linear_color = pow(color, vec3f(2.2));
    return vec4f(linear_color, 1.0);
};
