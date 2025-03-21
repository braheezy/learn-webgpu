struct VertexInput {
	@location(0) position: vec3f,
	@location(1) normal: vec3f,
	@location(2) color: vec3f,
	@location(3) uv: vec2f, // new attribute
};

struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) color: vec3f,
	@location(1) normal: vec3f,
	@location(2) uv: vec2f, // <--- Add a texture coordinate output
};

/**
 * A structure holding the value of our uniforms
 */
struct MyUniforms {
    projectionMatrix: mat4x4f,
    viewMatrix: mat4x4f,
    modelMatrix: mat4x4f,
    color: vec4f,
    time: f32,
};

@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;

// The texture binding
@group(0) @binding(1) var gradientTexture: texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;
	out.position = uMyUniforms.projectionMatrix * uMyUniforms.viewMatrix * uMyUniforms.modelMatrix * vec4f(in.position, 1.0);
    out.normal = (uMyUniforms.modelMatrix * vec4f(in.normal, 0.0)).xyz;
	out.color = in.color;
	out.uv = in.uv;
	return out;
}


@group(0) @binding(2) var textureSampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
	// We remap UV coords to actual texel coordinates
	let texelCoords = vec2i(in.uv * vec2f(textureDimensions(gradientTexture)));

	// And we fetch a texel from the texture
    let color = textureSample(gradientTexture, textureSampler, in.uv).rgb;

	return vec4f(color, uMyUniforms.color.a);
}
