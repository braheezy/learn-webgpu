struct VertexInput {
  @location(0) position: vec2f,
  @location(1) color: vec3f,
};
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

// Specify which binding index the uniform is attached to
// The memory location of the uniform is given by a pair of a *bind group* and a *binding*
@group(0) @binding(0) var<uniform> uTime: f32;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
    let ratio = 640.0 / 480.0;

    // We now move the scene depending on the time!
    var offset = vec2f(-0.6875, -0.463);
    offset += 0.3 * vec2f(cos(uTime), sin(uTime));

    out.position = vec4f(in.position.x + offset.x, (in.position.y + offset.y) * ratio, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f  {
    return vec4f(in.color, 1.0);
};
