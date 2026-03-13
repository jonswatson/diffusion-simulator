// ============================================================
// Fick's Second Law — 2D explicit finite difference (FTCS)
//
// PDE:          ∂C/∂t = D∇²C
// Discretized:  C[i,j]^(n+1) = C[i,j]^n + r·(C[i+1,j] + C[i-1,j]
//                                             + C[i,j+1] + C[i,j-1]
//                                             − 4·C[i,j])
//
// where r = D·dt/dx²  (Fourier number)
//
// Stability: r ≤ 0.25 in 2D  →  dt ≤ dx²/(4D)
// We use a safety factor of 0.4, giving r ≈ 0.1
//
// Boundary:  Zero-flux Neumann — ∂C/∂n = 0 at all walls
//            Implemented by clamping neighbor indices (mirror ghost cell)
//
// Units:  C = mole fraction [0, 1]
//         D = interdiffusion coefficient [m²/s]
//         dx = grid spacing [m/pixel]
//         dt = timestep [s]
// ============================================================

struct Uniforms {
  width  : u32,
  height : u32,
  r      : f32,  // Fourier number = D·dt/dx²
  _pad   : u32,
}

@group(0) @binding(0) var<uniform>             uniforms : Uniforms;
@group(0) @binding(1) var<storage, read>       C_in     : array<f32>;
@group(0) @binding(2) var<storage, read_write> C_out    : array<f32>;

fn cellIndex(x: u32, y: u32) -> u32 {
  return y * uniforms.width + x;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= uniforms.width || y >= uniforms.height) { return; }

  // Zero-flux Neumann BCs: mirror ghost cells at boundaries.
  // NOTE: x - 1u when x = 0 wraps to u32::MAX in WGSL, so we use
  // select() instead of clamp() to avoid the unsigned underflow bug.
  let xm = select(x - 1u, 0u, x == 0u);
  let xp = min(x + 1u, uniforms.width  - 1u);
  let ym = select(y - 1u, 0u, y == 0u);
  let yp = min(y + 1u, uniforms.height - 1u);

  let center = C_in[cellIndex(x, y)];

  // 5-point Laplacian stencil
  let laplacian = C_in[cellIndex(xp, y)] + C_in[cellIndex(xm, y)]
                + C_in[cellIndex(x, yp)] + C_in[cellIndex(x, ym)]
                - 4.0 * center;

  C_out[cellIndex(x, y)] = center + uniforms.r * laplacian;
}
