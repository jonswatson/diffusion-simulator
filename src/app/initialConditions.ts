// ============================================================
// Initial condition generators for all simulation modes.
//
// Spinodal: random noise around a mean composition.
// Grain growth: Voronoi tessellation (sharp IC, relaxes into diffuse profiles).
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
 * For each pixel, the nearest-seed grain is set to η = 1; all others remain 0.
 * Interfaces are initially sharp — the solver relaxes them into diffuse profiles
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
