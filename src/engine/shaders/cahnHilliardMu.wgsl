// ============================================================
// Cahn-Hilliard — Pass 1: Chemical potential μ from composition φ
//
// Split form of the Cahn-Hilliard equation:
//   μ = f'(φ) − ε²·∇²φ
//
// Regular solution free energy:
//   f(φ) = Ω·φ·(1−φ) + RT·[φ·ln(φ) + (1−φ)·ln(1−φ)]
//   f'(φ) = Ω·(1−2φ) + RT·ln(φ/(1−φ))
//
// In dimensionless form (dividing by RT):
//   f'(φ) = Ω_d·(1−2φ) + ln(φ/(1−φ))
// where Ω_d = Ω/(RT). Spinodal instability when Ω_d > 2.
//
// The gradient term ε²·∇²φ penalizes interfaces.
// This pass reads the composition field and writes the chemical potential.
// The second pass (cahnHilliardPhi.wgsl) then advances φ via ∇²μ.
//
// Boundary: zero-flux Neumann (clamped indices)
// ============================================================

struct Uniforms {
  width      : u32,
  height     : u32,
  dt         : f32,
  dx         : f32,
  M          : f32,   // mobility (dimensionless, Arrhenius-scaled)
  epsilon_sq : f32,   // ε² — gradient energy coefficient
  Omega      : f32,   // dimensionless Ω/RT — regular solution parameter
  RT         : f32,   // dimensionless (= 1.0) — for clarity in f'(φ)
}

@group(0) @binding(0) var<uniform>             uniforms : Uniforms;
@group(0) @binding(1) var<storage, read>       phi_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> mu_out   : array<f32>;

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

  // 5-point Laplacian of φ (divided by dx²)
  let lap_phi = (phi_in[idx(xp, y)] + phi_in[idx(xm, y)]
              + phi_in[idx(x, yp)] + phi_in[idx(x, ym)]
              - 4.0 * phi) / (uniforms.dx * uniforms.dx);

  // Clamp φ away from 0 and 1 to avoid log(0) singularity
  let phi_c = clamp(phi, 1e-6, 1.0 - 1e-6);

  // f'(φ) = Ω_d·(1−2φ) + RT_d·ln(φ/(1−φ))   (regular solution)
  let f_prime = uniforms.Omega * (1.0 - 2.0 * phi_c)
              + uniforms.RT * log(phi_c / (1.0 - phi_c));

  // μ = f'(φ) − ε²·∇²φ
  mu_out[idx(x, y)] = f_prime - uniforms.epsilon_sq * lap_phi;
}
