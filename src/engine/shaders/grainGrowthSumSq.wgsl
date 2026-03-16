// ============================================================
// Grain growth — Pass 1: sum of squares.
//
// For each pixel (x,y), computes: sumSq[y*W + x] = Σᵢ ηᵢ(y,x)²
//
// This auxiliary scalar field lets Pass 2 compute the grain-grain
// interaction term as: Σⱼ≠ᵢ ηⱼ² = sumSq − ηᵢ²
// without a nested inner loop over all pairs.
//
// Buffer layout: eta[g * W² + y * W + x]  (grain-major order)
// Units: all dimensionless.
// ============================================================

struct GGUniforms {
  numGrains : u32,
  gridWidth : u32,
  dt        : f32,
  kappa     : f32,
  Wbar      : f32,  // double-well barrier height W
  A         : f32,
  L         : f32,
  _pad      : f32,
}

@group(0) @binding(0) var<uniform>             uniforms : GGUniforms;
@group(0) @binding(1) var<storage, read>       eta      : array<f32>;  // source field
@group(0) @binding(2) var<storage, read_write> sumSq    : array<f32>;  // output

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let N = uniforms.gridWidth;
  if (x >= N || y >= N) { return; }

  let planeSize = N * N;
  let pix = y * N + x;

  var s = 0.0;
  for (var g = 0u; g < uniforms.numGrains; g++) {
    let e = eta[g * planeSize + pix];
    s += e * e;
  }
  sumSq[pix] = s;
}
