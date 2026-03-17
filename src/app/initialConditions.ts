// ============================================================
// Initial condition generators for all simulation modes.
//
// Spinodal: random noise around a mean composition.
// Grain growth: Voronoi tessellation (sharp IC, relaxes into diffuse profiles).
// Physics tests: planar interface, circular grain, triple junction.
// ============================================================

/**
 * Generate a spinodal decomposition field: uniform random noise
 * centered on meanPhi. All values clamped to [0, 1].
 *
 * The noise must be inside the spinodal region (mean near 0.5)
 * for decomposition to occur with a double-well free energy.
 */
export function generateSpinodalField(
  gridSize: number,
  meanPhi = 0.5,
  noiseAmplitude = 0.05,
): Float32Array {
  const N = gridSize * gridSize;
  const field = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const noise = (Math.random() * 2 - 1) * noiseAmplitude;
    field[i] = Math.max(0, Math.min(1, meanPhi + noise));
  }
  return field;
}

/**
 * Voronoi tessellation IC for grain growth.
 * Returns a flat Float32Array of size [numGrains × gridSize²] in grain-major order.
 * For each pixel, the nearest-seed grain is set to φ = 1; all others remain 0.
 * Simplex constraint Σφᵢ = 1 holds by construction (exactly one grain = 1 per pixel).
 * Interfaces are initially sharp — the solver relaxes them into diffuse tanh profiles
 * during the initial transient before coarsening begins.
 * Periodic distance metric so grains can wrap around domain boundaries.
 */
export function generateVoronoiField(gridSize: number, numGrains: number): Float32Array {
  const N = gridSize;
  const planeSize = N * N;
  const field = new Float32Array(numGrains * planeSize);

  // Random seed positions (uniform over the domain)
  const seedX = new Float32Array(numGrains);
  const seedY = new Float32Array(numGrains);
  for (let g = 0; g < numGrains; g++) {
    seedX[g] = Math.random() * N;
    seedY[g] = Math.random() * N;
  }

  // Assign each pixel to its nearest seed (periodic Euclidean distance)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let minDist = Infinity;
      let nearest = 0;
      for (let g = 0; g < numGrains; g++) {
        let dx = Math.abs(x - seedX[g]);
        let dy = Math.abs(y - seedY[g]);
        if (dx > N / 2) dx = N - dx;
        if (dy > N / 2) dy = N - dy;
        const d2 = dx * dx + dy * dy;
        if (d2 < minDist) { minDist = d2; nearest = g; }
      }
      field[nearest * planeSize + y * N + x] = 1.0;
    }
  }

  return field;
}

/**
 * Planar interface IC for 2-grain physics test.
 * Grain 0: tanh profile rising left-to-right across the grid midpoint.
 * Grain 1: complement (φ₁ = 1 − φ₀).
 * Simplex constraint Σφᵢ = 1 holds by construction.
 *
 * Interface half-width xi (pixels) should match the equilibrium profile
 * ξ_eq = 2√(κ/W) for the Allen-Cahn model (= 2 px for κ=W=1).
 * Using xi slightly larger avoids the initial transient.
 */
export function generatePlanarInterface(gridSize: number, xi = 3.0): Float32Array {
  const N = gridSize;
  const planeSize = N * N;
  const field = new Float32Array(2 * planeSize);

  const midX = N / 2;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const phi0 = 0.5 + 0.5 * Math.tanh((x - midX) / xi);
      field[0 * planeSize + y * N + x] = phi0;
      field[1 * planeSize + y * N + x] = 1.0 - phi0;
    }
  }
  return field;
}

/**
 * Circular grain IC for 2-grain physics test.
 * Grain 0: disk of radius r0 centered at domain center, with a tanh interface.
 * Grain 1: matrix (complement).
 * Simplex constraint Σφᵢ = 1 holds by construction.
 *
 * Expected behavior: Allen-Cahn drives dR/dt = −L·κ/R (isotropic curvature flow),
 * so A(t) = π·R₀² − 2π·L·κ·t → area decreases approximately linearly.
 */
export function generateCircularGrain(
  gridSize: number,
  radius?: number,
  xi = 3.0,
): Float32Array {
  const N = gridSize;
  const planeSize = N * N;
  const field = new Float32Array(2 * planeSize);
  const r0 = radius ?? N / 4;
  const cx = N / 2;
  const cy = N / 2;

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const r = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      const phi0 = 0.5 + 0.5 * Math.tanh((r0 - r) / xi);
      field[0 * planeSize + y * N + x] = phi0;
      field[1 * planeSize + y * N + x] = 1.0 - phi0;
    }
  }
  return field;
}

/**
 * Triple junction IC for 3-grain physics test.
 * Three Voronoi grains with seeds placed at 120°-symmetric positions
 * r = N/3 from the domain center. Sharp IC — solver smooths the interfaces.
 * Simplex constraint Σφᵢ = 1 holds by construction.
 *
 * Expected behavior: interfaces evolve toward equal 120° angles at the
 * triple junction (isotropic equal-energy condition).
 */
export function generateTripleJunction(gridSize: number): Float32Array {
  const N = gridSize;
  const planeSize = N * N;
  const field = new Float32Array(3 * planeSize);
  const cx = N / 2;
  const cy = N / 2;
  const r = N / 3;

  // Seeds at 90°, 210°, 330° (120° spacing, first seed at top)
  const seeds = [90, 210, 330].map(deg => {
    const rad = (deg * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  });

  // Assign each pixel to its nearest seed (non-periodic — junction stays at center)
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let minDist = Infinity;
      let nearest = 0;
      for (let g = 0; g < 3; g++) {
        const dx = x - seeds[g].x;
        const dy = y - seeds[g].y;
        const d2 = dx * dx + dy * dy;
        if (d2 < minDist) { minDist = d2; nearest = g; }
      }
      field[nearest * planeSize + y * N + x] = 1.0;
    }
  }
  return field;
}
