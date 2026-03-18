// ============================================================
// Grain growth render shader — weighted color blend.
//
// For each canvas pixel (x,y):
//   color = Σᵢ φᵢ(x,y) · colorTable[i]
//
// With the constrained formulation (Σφᵢ=1, φᵢ≥0), this gives:
//   - bulk grain (φᵢ≈1): grain's own color
//   - two-grain boundary (φᵢ≈φⱼ≈0.5): blended color (no black)
//   - triple junction (φᵢ≈1/3): three-way blend (dim but not void)
//
// Buffer layout: phi[g * W² + y * W + x]  (grain-major order)
// Color table:   colorTable[g * 3 + {0,1,2}] = {R, G, B}
// ============================================================

struct RenderUniforms {
  numGrains : u32,
  gridWidth : u32,
  _pad0     : f32,
  _pad1     : f32,
}

override NUM_GRAINS : u32 = 8u;

@group(0) @binding(0) var<uniform>       uniforms   : RenderUniforms;
@group(0) @binding(1) var<storage, read> phi        : array<f32>;
@group(0) @binding(2) var<storage, read> colorTable : array<f32>;  // [numGrains × 3]

struct VertexOut {
  @builtin(position) pos : vec4f,
}

// Full-screen triangle: 3 vertices cover the entire [-1,1]² NDC space.
@vertex
fn vs_main(@builtin(vertex_index) vIdx: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  var out: VertexOut;
  out.pos = vec4f(positions[vIdx], 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let x = u32(fragCoord.x);
  let y = u32(fragCoord.y);
  let N = uniforms.gridWidth;

  if (x >= N || y >= N) {
    return vec4f(0.05, 0.05, 0.05, 1.0);
  }

  let planeSize = N * N;
  let pix = y * N + x;

  // Weighted blend: color = Σᵢ φᵢ · colorTable[i]
  // With Σφᵢ = 1, this is a proper convex combination — no black bands.
  var color = vec3f(0.0);
  for (var g = 0u; g < NUM_GRAINS; g++) {
    let phi_g = phi[g * planeSize + pix];
    let r     = colorTable[g * 3u];
    let gr    = colorTable[g * 3u + 1u];
    let b     = colorTable[g * 3u + 2u];
    color += phi_g * vec3f(r, gr, b);
  }

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
