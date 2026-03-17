// ============================================================
// Binary Solidification — Pass 3: Composition update
//
// Conserved composition equation:
//   ∂c/∂t = ∇·(M(φ) · ∇μc)
//
// Discretized with face-centered mobility and Neumann BCs.
// Face-centered mobility (arithmetic average at cell faces):
//   M_xp = 0.5 · (M[x,y] + M[x+1,y])   (right face)
//   M_xm = 0.5 · (M[x,y] + M[x-1,y])   (left face)
//   M_yp = 0.5 · (M[x,y] + M[x,y+1])   (top face)
//   M_ym = 0.5 · (M[x,y] + M[x,y-1])   (bottom face)
//
// Face-centered fluxes (dx = 1):
//   Jx_xp = −M_xp · (μc[x+1,y] − μc[x,y])
//   Jx_xm = −M_xm · (μc[x,y] − μc[x-1,y])
//
// Composition update (dx = 1, so no 1/dx² factor):
//   c_new = c + dt · [(Jx_xm − Jx_xp) + (Jy_ym − Jy_yp)]
//
// Note sign convention: flux divergence = inflow − outflow
//   div(J) = −(Jx_xp − Jx_xm) − (Jy_yp − Jy_ym)
//
// Boundary: zero-flux Neumann — at boundaries M_face = 0 by clamping
// (clamped neighbor indices → same μc on both sides → zero flux)
//
// Stability: dt ≤ 0.1 / (4 · Ml · 2 · max(As, Al))
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

@group(0) @binding(0) var<uniform>             uniforms : Uniforms;
@group(0) @binding(1) var<storage, read>       c_in     : array<f32>;
@group(0) @binding(2) var<storage, read>       muc_in   : array<f32>;
@group(0) @binding(3) var<storage, read>       mob_in   : array<f32>;
@group(0) @binding(4) var<storage, read_write> c_out    : array<f32>;

fn idx(x: u32, y: u32) -> u32 {
  return y * uniforms.width + x;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= uniforms.width || y >= uniforms.height) { return; }

  // Clamped neighbor indices for zero-flux Neumann BCs
  let xm = select(x - 1u, 0u, x == 0u);
  let xp = min(x + 1u, uniforms.width  - 1u);
  let ym = select(y - 1u, 0u, y == 0u);
  let yp = min(y + 1u, uniforms.height - 1u);

  let muc_c  = muc_in[idx(x,  y )];
  let muc_xp = muc_in[idx(xp, y )];
  let muc_xm = muc_in[idx(xm, y )];
  let muc_yp = muc_in[idx(x,  yp)];
  let muc_ym = muc_in[idx(x,  ym)];

  let mob_c  = mob_in[idx(x,  y )];
  let mob_xp = mob_in[idx(xp, y )];
  let mob_xm = mob_in[idx(xm, y )];
  let mob_yp = mob_in[idx(x,  yp)];
  let mob_ym = mob_in[idx(x,  ym)];

  // Face-centered mobilities (arithmetic mean)
  // At boundary pixels, clamped neighbor == self → M_face = M_c → flux = 0 (same μ)
  let M_xp = 0.5 * (mob_c + mob_xp);
  let M_xm = 0.5 * (mob_c + mob_xm);
  let M_yp = 0.5 * (mob_c + mob_yp);
  let M_ym = 0.5 * (mob_c + mob_ym);

  // Fluxes: J = −M · ∇μc  (dx = 1)
  let Jx_xp = -M_xp * (muc_xp - muc_c);
  let Jx_xm = -M_xm * (muc_c  - muc_xm);
  let Jy_yp = -M_yp * (muc_yp - muc_c);
  let Jy_ym = -M_ym * (muc_c  - muc_ym);

  // Divergence: ∇·J = (Jx_xm − Jx_xp) + (Jy_ym − Jy_yp)  [flux in − flux out]
  let div_J = (Jx_xm - Jx_xp) + (Jy_ym - Jy_yp);

  c_out[idx(x, y)] = c_in[idx(x, y)] + uniforms.dt * div_J;
}
