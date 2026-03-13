// ============================================================
// Test compute shader: writes a checkerboard pattern to a storage buffer.
// Used only in Stage 1 to validate the GPU pipeline end-to-end.
// DELETE THIS FILE after Stage 1 passes.
// ============================================================

struct Uniforms {
  width  : u32,
  height : u32,
  r      : f32,
  _pad   : u32,
}

@group(0) @binding(0) var<uniform>             uniforms : Uniforms;
@group(0) @binding(1) var<storage, read>       dummy_in : array<f32>;
@group(0) @binding(2) var<storage, read_write> buf_out  : array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= uniforms.width || y >= uniforms.height) { return; }

  // Checkerboard: 1.0 on even cells, 0.0 on odd cells
  let isEven = (x + y) % 2u == 0u;
  buf_out[y * uniforms.width + x] = select(0.0, 1.0, isEven);
}
