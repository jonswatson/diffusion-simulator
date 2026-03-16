// ============================================================
// Grain growth — Pass 2: Allen-Cahn update for all order parameters.
//
// PDE (per grain i):
//   ∂ηᵢ/∂t = L [ κ·∇²ηᵢ
//                − (W/2)·ηᵢ·(1−ηᵢ)·(1−2ηᵢ)        ← bulk double-well
//                − 2A·ηᵢ·(sumSq − ηᵢ²) ]            ← grain-grain repulsion
//
// where sumSq = Σⱼ ηⱼ² was computed by Pass 1.
//
// Discretization:
//   Laplacian: 5-point central difference, dx = 1 (nondimensional).
//   Time integration: explicit Euler.
//   Boundary conditions: periodic (index wrapping).
//
// Dispatch: 3D — workgroup (16,16,1), dispatch z = numGrains.
//   gid.z selects the grain; gid.{x,y} select the pixel.
//
// Buffer layout: eta[g * W² + y * W + x]  (grain-major order).
//
// Stability: dt ≤ 0.2 / (L·(4κ + W/2 + 2A·(N−1)))  [set in physics.ts]
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
@group(0) @binding(1) var<storage, read>       etaIn    : array<f32>;  // source (current step)
@group(0) @binding(2) var<storage, read>       sumSq    : array<f32>;  // from Pass 1
@group(0) @binding(3) var<storage, read_write> etaOut   : array<f32>;  // dest (next step)

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x  = gid.x;
  let y  = gid.y;
  let g  = gid.z;
  let N  = uniforms.gridWidth;
  let NG = uniforms.numGrains;
  if (x >= N || y >= N || g >= NG) { return; }

  let planeSize = N * N;
  let pix = y * N + x;
  let idx = g * planeSize + pix;

  let eta_i = etaIn[idx];

  // Periodic neighbors
  let xp = (x + 1u) % N;
  let xm = (x + N - 1u) % N;
  let yp = (y + 1u) % N;
  let ym = (y + N - 1u) % N;

  // 5-point Laplacian (dx = 1 implicit)
  let lap = etaIn[g * planeSize + y  * N + xp]
          + etaIn[g * planeSize + y  * N + xm]
          + etaIn[g * planeSize + yp * N + x ]
          + etaIn[g * planeSize + ym * N + x ]
          - 4.0 * eta_i;

  // Double-well reaction: (W/2)·η·(1−η)·(1−2η)
  // Derived from d/dη[ W/4·η²(1−η)² ] = W/2·η(1−η)(1−2η)
  let bulk = uniforms.Wbar * 0.5 * eta_i * (1.0 - eta_i) * (1.0 - 2.0 * eta_i);

  // Grain-grain repulsion: 2A·η_i·Σⱼ≠ᵢ ηⱼ²
  let cross   = sumSq[pix] - eta_i * eta_i;
  let interact = 2.0 * uniforms.A * eta_i * cross;

  // Allen-Cahn: ∂ηᵢ/∂t = −L·δF/δηᵢ = L·(κ∇²ηᵢ − bulk − interact)
  etaOut[idx] = eta_i + uniforms.L * uniforms.dt * (uniforms.kappa * lap - bulk - interact);
}
