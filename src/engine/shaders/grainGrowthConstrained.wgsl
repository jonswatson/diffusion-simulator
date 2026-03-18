// ============================================================
// Constrained Allen-Cahn grain growth — single-pass per-pixel solver.
//
// PDE: ∂φᵢ/∂t = −L(μᵢ − λ)
//      λ = (1/N)·Σᵢ μᵢ  (per-pixel Lagrange multiplier)
//      → Σᵢ ∂φᵢ/∂t = 0  (sum is conserved in continuous time)
//
// Free energy:
//   F = ∫ [ Σᵢ κ/2·|∇φᵢ|² + Σᵢ W/4·φᵢ²(1−φᵢ)² + Σᵢ<ⱼ A·φᵢ²φⱼ² ] dV
//
// Chemical potential:
//   μᵢ = δF/δφᵢ = −κ∇²φᵢ + W/2·φᵢ(1−φᵢ)(1−2φᵢ) + 2A·φᵢ·Σⱼ≠ᵢ φⱼ²
//
// Numerical scheme (per pixel, per step):
//   1. Load φᵢ, compute Laplacian ∇²φᵢ (5-point stencil, dx=1)
//   2. Compute μᵢ for all grains; λ = mean(μ)
//   3. φ̃ᵢ = φᵢ − dt·L·(μᵢ − λ)    [explicit Euler, constrained]
//   4. Project (φ̃) onto probability simplex: φᵢ≥0, Σφᵢ=1
//      (Euclidean projection — Duchi et al. 2008)
//
// BCs: periodic (index wrapping).
// Units: nondimensional (dx=1; κ=W=A=L=1 are standard defaults).
// Stability: dt ≤ 0.1 / (L·(4κ + W/2 + 2A·(N−1)))
//
// Buffer layout: phi[g * W² + y * W + x]  (grain-major order, f32)
// Workgroup: 16×16 (2D dispatch, one thread per pixel)
// ============================================================

struct Uniforms {
  numGrains : u32,
  gridWidth : u32,
  dt        : f32,
  kappa     : f32,
  W_dw      : f32,   // double-well barrier coefficient W
  A_inter   : f32,   // grain-grain interaction coefficient A
  L_mob     : f32,   // Allen-Cahn mobility L
  _pad      : f32,
}

override NUM_GRAINS : u32 = 8u;

@group(0) @binding(0) var<uniform>             u      : Uniforms;
@group(0) @binding(1) var<storage, read>       phiIn  : array<f32>;
@group(0) @binding(2) var<storage, read_write> phiOut : array<f32>;

// ---- Euclidean projection onto the probability simplex ----
// Given v ∈ ℝᴺ, computes u = argmin‖u−v‖² s.t. Σuᵢ=1, uᵢ≥0.
// Algorithm: Duchi, Shalev-Shwartz, Singer, Chandra (ICML 2008).
// Runs in-place on *v. Inner arrays sized at compile-time MAX 32.
fn projectSimplex(v: ptr<function, array<f32, 32>>, n: u32) {
  // Step 1: copy v into u and sort descending (O(n²) insertion sort; n ≤ 32)
  var u: array<f32, 32>;
  for (var i = 0u; i < n; i++) { u[i] = (*v)[i]; }
  for (var i = 0u; i < n - 1u; i++) {
    for (var j = i + 1u; j < n; j++) {
      if (u[j] > u[i]) {
        let tmp = u[i]; u[i] = u[j]; u[j] = tmp;
      }
    }
  }

  // Step 2: find ρ = largest j s.t. u[j] + (1 − Σₖ≤j u[k]) / (j+1) > 0
  var cumsum = 0.0;
  var rho = 0u;
  for (var j = 0u; j < n; j++) {
    cumsum += u[j];
    if (u[j] + (1.0 - cumsum) / f32(j + 1u) > 0.0) {
      rho = j;
    }
  }

  // Step 3: θ = (1 − Σₖ₌₀^ρ u[k]) / (ρ + 1)
  var s = 0.0;
  for (var j = 0u; j <= rho; j++) { s += u[j]; }
  let theta = (1.0 - s) / f32(rho + 1u);

  // Step 4: threshold — applies θ to original (unsorted) v
  for (var i = 0u; i < n; i++) {
    (*v)[i] = max((*v)[i] + theta, 0.0);
  }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  let N = u.gridWidth;
  if (x >= N || y >= N) { return; }

  let planeSize = N * N;
  let pix       = y * N + x;

  // Periodic neighbors (wrapping indices)
  let xp = (x + 1u) % N;
  let xm = (x + N - 1u) % N;
  let yp = (y + 1u) % N;
  let ym = (y + N - 1u) % N;

  // Load φ for this pixel across all grains; compute ∇²φᵢ and Σφⱼ²
  var phi  : array<f32, 32>;
  var lap  : array<f32, 32>;
  var sumSq = 0.0;
  for (var g = 0u; g < NUM_GRAINS; g++) {
    let base = g * planeSize;
    let c    = phiIn[base + pix];
    phi[g]   = c;
    sumSq   += c * c;
    // 5-point Laplacian (dx = 1 nondimensional)
    lap[g] = phiIn[base + y  * N + xp]
           + phiIn[base + y  * N + xm]
           + phiIn[base + yp * N + x ]
           + phiIn[base + ym * N + x ]
           - 4.0 * c;
  }

  // Chemical potential μᵢ and λ = mean(μ)
  var mu       : array<f32, 32>;
  var lambdaVal = 0.0;
  for (var g = 0u; g < NUM_GRAINS; g++) {
    let p    = phi[g];
    // Double-well bulk derivative: dg/dφ = W/2·φ(1−φ)(1−2φ)
    let bulk = u.W_dw * 0.5 * p * (1.0 - p) * (1.0 - 2.0 * p);
    // Cross-interaction: 2A·φᵢ·Σⱼ≠ᵢ φⱼ² = 2A·φᵢ·(sumSq − φᵢ²)
    let xtrm = 2.0 * u.A_inter * p * (sumSq - p * p);
    mu[g]     = -u.kappa * lap[g] + bulk + xtrm;
    lambdaVal += mu[g];
  }
  lambdaVal /= f32(NUM_GRAINS);

  // Constrained explicit Euler: φ̃ᵢ = φᵢ − dt·L·(μᵢ − λ)
  var phiNew: array<f32, 32>;
  for (var g = 0u; g < NUM_GRAINS; g++) {
    phiNew[g] = phi[g] - u.dt * u.L_mob * (mu[g] - lambdaVal);
  }

  // Euclidean projection onto probability simplex (φᵢ≥0, Σφᵢ=1)
  projectSimplex(&phiNew, NUM_GRAINS);

  // Write output
  for (var g = 0u; g < NUM_GRAINS; g++) {
    phiOut[g * planeSize + pix] = phiNew[g];
  }
}
