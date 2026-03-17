// ============================================================
// Binary Solidification — Pass 2: Phase field update
//
// Allen-Cahn equation for the solid fraction φ:
//   ∂φ/∂t = Lphi · [κ_φ · ∇²φ − dF/dφ]
//
// Discretization (explicit Euler, dx = 1 nondimensional):
//   φ_new = φ + Lphi · dt · (κ_φ · ∇²φ − dfdphi)
//
// ∇²φ via 5-point Laplacian with zero-flux Neumann BCs (clamped indices):
//   ∇²φ ≈ φ[x+1,y] + φ[x-1,y] + φ[x,y+1] + φ[x,y-1] − 4·φ[x,y]
//
// dfdphi is precomputed by Pass 1 (binarySolidificationAux.wgsl).
//
// Stability: dt ≤ 0.1 / (Lphi · (4·kappa_phi + w))
// Output clamped to [0, 1] to prevent overshoot.
// ============================================================

struct Uniforms {
  width    : u32,
  height   : u32,
  dt       : f32,
  _pad0    : f32,
  kappa_phi: f32,
  w        : f32,
  Lphi     : f32,
  As       : f32,
  Al       : f32,
  cs_eq    : f32,
  cl_eq    : f32,
  Ms       : f32,
  Ml       : f32,
  _pad1    : f32,
  _pad2    : f32,
  _pad3    : f32,
}

@group(0) @binding(0) var<uniform>             uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read>       phi_in    : array<f32>;
@group(0) @binding(2) var<storage, read>       dfdphi_in : array<f32>;
@group(0) @binding(3) var<storage, read_write> phi_out   : array<f32>;

fn idx(x: u32, y: u32) -> u32 {
  return y * uniforms.width + x;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= uniforms.width || y >= uniforms.height) { return; }

  // Zero-flux Neumann: clamp neighbor indices at boundaries
  let xm = select(x - 1u, 0u, x == 0u);
  let xp = min(x + 1u, uniforms.width  - 1u);
  let ym = select(y - 1u, 0u, y == 0u);
  let yp = min(y + 1u, uniforms.height - 1u);

  let phi = phi_in[idx(x, y)];

  // 5-point Laplacian (dx = 1, so no dx² denominator)
  let lap_phi = phi_in[idx(xp, y)] + phi_in[idx(xm, y)]
              + phi_in[idx(x, yp)] + phi_in[idx(x, ym)]
              - 4.0 * phi;

  // Allen-Cahn explicit Euler step
  let phi_new = phi + uniforms.Lphi * uniforms.dt
              * (uniforms.kappa_phi * lap_phi - dfdphi_in[idx(x, y)]);

  // Clamp to physical range [0, 1]
  phi_out[idx(x, y)] = clamp(phi_new, 0.0, 1.0);
}
