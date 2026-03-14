// ============================================================
// Cahn-Hilliard — Pass 2: Composition update φ from chemical potential μ
//
// Split form of the Cahn-Hilliard equation:
//   ∂φ/∂t = M·∇²μ
//
// This pass reads the old composition and the chemical potential
// computed in Pass 1, then writes the new composition:
//   φ_new = φ_old + M·dt·∇²μ
//
// Boundary: zero-flux Neumann (clamped indices)
//
// Stability: dt ≤ safetyFactor · dx⁴ / (16·M·ε²)
//            for the biharmonic ∇⁴ term from the split scheme.
// ============================================================

struct Uniforms {
  width      : u32,
  height     : u32,
  dt         : f32,
  dx         : f32,
  M          : f32,   // mobility
  epsilon_sq : f32,   // (unused in this pass, shared struct)
  Omega      : f32,   // (unused in this pass, shared struct)
  RT         : f32,   // (unused in this pass, shared struct)
}

@group(0) @binding(0) var<uniform>             uniforms : Uniforms;
@group(0) @binding(1) var<storage, read>       phi_in   : array<f32>;
@group(0) @binding(2) var<storage, read>       mu_in    : array<f32>;
@group(0) @binding(3) var<storage, read_write> phi_out  : array<f32>;

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

  // 5-point Laplacian of μ (divided by dx²)
  let lap_mu = (mu_in[idx(xp, y)] + mu_in[idx(xm, y)]
             + mu_in[idx(x, yp)] + mu_in[idx(x, ym)]
             - 4.0 * mu_in[idx(x, y)]) / (uniforms.dx * uniforms.dx);

  // φ_new = φ_old + M·dt·∇²μ
  phi_out[idx(x, y)] = phi_in[idx(x, y)] + uniforms.M * uniforms.dt * lap_mu;
}
