// ============================================================
// Grain growth render shader.
//
// For each canvas pixel (x,y):
//   1. Find the dominant grain: i* = argmax_i ηᵢ(x,y)
//   2. Look up its RGB color from the color table.
//   3. Modulate brightness by smoothstep(0.1, 0.55, maxEta) so that:
//        bulk grain (maxEta ≈ 1.0) → full brightness
//        triple junction (maxEta ≈ 1/3) → dim but visible
//        fully void (maxEta ≈ 0) → dark
//
// Buffer layout: eta[g * W² + y * W + x]  (grain-major order)
// Color table:   colorTable[g * 3 + {0,1,2}] = {R, G, B}  (linear RGB)
// ============================================================

struct RenderUniforms {
  numGrains : u32,
  gridWidth : u32,
  _pad0     : f32,
  _pad1     : f32,
}

@group(0) @binding(0) var<uniform>       uniforms   : RenderUniforms;
@group(0) @binding(1) var<storage, read> eta        : array<f32>;
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

  // Find dominant grain
  var maxEta = -1.0;
  var best = 0u;
  for (var g = 0u; g < uniforms.numGrains; g++) {
    let e = eta[g * planeSize + pix];
    if (e > maxEta) { maxEta = e; best = g; }
  }

  let r  = colorTable[best * 3u];
  let gr = colorTable[best * 3u + 1u];
  let b  = colorTable[best * 3u + 2u];

  let bright = smoothstep(0.1, 0.55, maxEta);
  return vec4f(r * bright, gr * bright, b * bright, 1.0);
}
