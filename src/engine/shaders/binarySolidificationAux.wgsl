// ============================================================
// Binary Solidification — Pass 1: Auxiliary thermodynamics
//
// Coupled Allen-Cahn (φ) + composition (c) field equations.
// State: φ [0=liquid, 1=solid], c [solute mole fraction]
//
// This pass precomputes three per-pixel quantities needed by
// the phase and composition update passes:
//
//   dfdphi  = dF/dφ = h'(φ)·[fs(c) − fl(c)] + w·g'(φ)
//   mu_c    = dF/dc = h(φ)·fs'(c) + (1−h(φ))·fl'(c)
//   mob     = M(φ)  = Ms·h(φ) + Ml·(1−h(φ))
//
// Interpolation functions (Karma-Rappel):
//   h(φ)  = φ²(3−2φ)       — monotone, h(0)=0, h(1)=1
//   h'(φ) = 6φ(1−φ)        — h' in [0, 1.5]
//   g'(φ) = 2φ(1−φ)(1−2φ)  — double-well derivative
//
// Free energy densities (quadratic, each a parabola):
//   fs(c) = As·(c − cs_eq)²,  fs'(c) = 2·As·(c − cs_eq)
//   fl(c) = Al·(c − cl_eq)²,  fl'(c) = 2·Al·(c − cl_eq)
//
// Boundary: zero-flux Neumann (clamped indices) — unused in this pass
// Units: all nondimensional (dx = 1)
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
@group(0) @binding(2) var<storage, read>       c_in      : array<f32>;
@group(0) @binding(3) var<storage, read_write> dfdphi_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> muc_out   : array<f32>;
@group(0) @binding(5) var<storage, read_write> mob_out   : array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= uniforms.width || y >= uniforms.height) { return; }

  let pix = y * uniforms.width + x;
  let phi = clamp(phi_in[pix], 0.0, 1.0);
  let c   = c_in[pix];

  // Interpolation functions
  let h     = phi * phi * (3.0 - 2.0 * phi);
  let h_prime = 6.0 * phi * (1.0 - phi);
  let g_prime = 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi);

  // Bulk free energy densities
  let dc_s = c - uniforms.cs_eq;
  let dc_l = c - uniforms.cl_eq;
  let fs   = uniforms.As * dc_s * dc_s;
  let fl   = uniforms.Al * dc_l * dc_l;

  // df/dφ: driving force for phase field update
  dfdphi_out[pix] = h_prime * (fs - fl) + uniforms.w * g_prime;

  // μc = df/dc: chemical potential drives composition diffusion
  muc_out[pix] = h * (2.0 * uniforms.As * dc_s)
               + (1.0 - h) * (2.0 * uniforms.Al * dc_l);

  // M(φ): composition mobility interpolated between phases
  mob_out[pix] = uniforms.Ms * h + uniforms.Ml * (1.0 - h);
}
