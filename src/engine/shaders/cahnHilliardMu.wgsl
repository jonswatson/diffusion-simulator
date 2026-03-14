// ============================================================
// Cahn-Hilliard — Pass 1: Chemical potential μ from composition φ
//
// Split form of the Cahn-Hilliard equation:
//   μ = f'(φ) − ε²·∇²φ
//
// where:
//   f(φ) = A·φ²·(1−φ)²               (double-well free energy)
//   f'(φ) = 2A·φ·(1−φ)·(1−2φ)        (chemical potential from bulk)
//   ε²·∇²φ                            (gradient energy, penalizes interfaces)
//
// This pass reads the composition field and writes the chemical potential.
// The second pass (cahnHilliardPhi.wgsl) then advances φ via ∇²μ.
//
// Boundary: zero-flux Neumann (clamped indices)
// Units: all dimensionless (normalized for educational use)
// ============================================================

struct Uniforms {
  width      : u32,
  height     : u32,
  dt         : f32,
  dx         : f32,
  M          : f32,   // mobility
  epsilon_sq : f32,   // ε² — gradient energy coefficient
  A          : f32,   // barrier height in double-well
  _pad       : u32,
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

  // f'(φ) = 2A · φ · (1−φ) · (1−2φ)
  let f_prime = 2.0 * uniforms.A * phi * (1.0 - phi) * (1.0 - 2.0 * phi);

  // μ = f'(φ) − ε²·∇²φ
  mu_out[idx(x, y)] = f_prime - uniforms.epsilon_sq * lap_phi;
}
