// ============================================================
// Allen-Cahn multi-order-parameter grain growth
//
// PDE:  ∂η_i/∂t = −L · (∂f/∂η_i − κ·∇²η_i)   for each grain i
//
// Free energy:
//   f = Σ_i  A·η_i²·(1−η_i)²   +  Σ_{i≠j}  B·η_i²·η_j²
//
// Derivatives:
//   ∂f/∂η_i = 2A·η_i·(1−η_i)·(1−2η_i) + 2B·η_i·Σ_{j≠i} η_j²
//
// Buffer layout:
//   Packed flat array: eta[grain * W * H + y * W + x]
//   Two such buffers for ping-pong (etaA, etaB)
//
// Stability:
//   dt ≤ min(dx²/(4·L·κ), 1/(L·2A))  ×  safetyFactor
//
// Boundary: zero-flux Neumann (clamped indices)
// Units: all dimensionless
// ============================================================

struct Uniforms {
  width   : u32,
  height  : u32,
  dt      : f32,
  dx      : f32,
  L       : f32,   // kinetic coefficient
  kappa   : f32,   // gradient energy coefficient
  A       : f32,   // same-phase well depth
  B       : f32,   // cross-coupling strength
  N       : u32,   // number of grains
  _pad1   : u32,
  _pad2   : u32,
  _pad3   : u32,
}

@group(0) @binding(0) var<uniform>             uniforms : Uniforms;
@group(0) @binding(1) var<storage, read>       eta_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> eta_out  : array<f32>;

fn idx(grain: u32, x: u32, y: u32) -> u32 {
  return grain * uniforms.width * uniforms.height + y * uniforms.width + x;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x;
  let y = id.y;
  if (x >= uniforms.width || y >= uniforms.height) { return; }

  // Zero-flux Neumann BCs
  let xm = select(x - 1u, 0u, x == 0u);
  let xp = min(x + 1u, uniforms.width  - 1u);
  let ym = select(y - 1u, 0u, y == 0u);
  let yp = min(y + 1u, uniforms.height - 1u);

  let dx2 = uniforms.dx * uniforms.dx;

  // For each grain order parameter, compute Allen-Cahn update
  for (var i = 0u; i < uniforms.N; i = i + 1u) {
    let eta_i = eta_in[idx(i, x, y)];

    // Laplacian of η_i
    let lap_i = (eta_in[idx(i, xp, y)] + eta_in[idx(i, xm, y)]
              + eta_in[idx(i, x, yp)] + eta_in[idx(i, x, ym)]
              - 4.0 * eta_i) / dx2;

    // Same-phase double-well: ∂f_bulk/∂η_i = 2A·η_i·(1−η_i)·(1−2η_i)
    let df_bulk = 2.0 * uniforms.A * eta_i * (1.0 - eta_i) * (1.0 - 2.0 * eta_i);

    // Cross-coupling: 2B·η_i·Σ_{j≠i} η_j²
    var sum_eta_j_sq = 0.0;
    for (var j = 0u; j < uniforms.N; j = j + 1u) {
      if (j != i) {
        let eta_j = eta_in[idx(j, x, y)];
        sum_eta_j_sq += eta_j * eta_j;
      }
    }
    let df_cross = 2.0 * uniforms.B * eta_i * sum_eta_j_sq;

    // Allen-Cahn: ∂η_i/∂t = −L·(∂f/∂η_i − κ·∇²η_i)
    let rhs = -uniforms.L * (df_bulk + df_cross - uniforms.kappa * lap_i);
    eta_out[idx(i, x, y)] = eta_i + uniforms.dt * rhs;
  }
}
