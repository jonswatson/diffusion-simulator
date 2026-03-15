// ============================================================
// Initial condition generators for all simulation modes.
//
// Spinodal: random noise around a mean composition.
// Voronoi: grain assignment from random seed points.
//   - Sharp: η = 0 or 1 (legacy, for tests).
//   - Smooth: tanh-profile boundaries at equilibrium width.
// Colors: evenly-spaced HSL hues for grain display.
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
 * Place N seed points on a gridSize×gridSize domain using
 * Poisson-disk–style rejection sampling (min distance constraint).
 */
function placeSeeds(
  gridSize: number,
  numGrains: number,
): [number, number][] {
  const seeds: [number, number][] = [];
  const minDist = gridSize / (numGrains * 2);

  for (let i = 0; i < numGrains; i++) {
    let x: number, y: number;
    let attempts = 0;
    do {
      x = Math.random() * gridSize;
      y = Math.random() * gridSize;
      attempts++;
      // After 100 attempts, accept any position to avoid infinite loops
      if (attempts > 100) break;
    } while (
      seeds.some(([sx, sy]) => {
        const dx = x - sx;
        const dy = y - sy;
        return Math.sqrt(dx * dx + dy * dy) < minDist;
      })
    );
    seeds.push([x, y]);
  }

  return seeds;
}

/**
 * Generate a Voronoi tessellation with sharp (step-function) boundaries.
 * Returns a packed Float32Array of size numGrains × gridSize².
 * For each cell, the owning grain has η = 1, all others η = 0.
 */
export function generateVoronoiField(
  gridSize: number,
  numGrains: number,
): Float32Array {
  const WH = gridSize * gridSize;
  const packed = new Float32Array(numGrains * WH);

  const seeds = placeSeeds(gridSize, numGrains);

  // Assign each cell to the nearest seed
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      let minDistSq = Infinity;
      let owner = 0;
      for (let i = 0; i < numGrains; i++) {
        const dx = x - seeds[i][0];
        const dy = y - seeds[i][1];
        const distSq = dx * dx + dy * dy;
        if (distSq < minDistSq) {
          minDistSq = distSq;
          owner = i;
        }
      }
      packed[owner * WH + y * gridSize + x] = 1.0;
    }
  }

  return packed;
}

/**
 * Generate a Voronoi tessellation with smooth tanh-profile boundaries.
 *
 * The Allen-Cahn equilibrium profile between two grains is a tanh:
 *   η_i(d) = 0.5·(1 + tanh(d / (ξ/√2)))
 * where d is the signed distance from the grain boundary.
 *
 * By starting with the equilibrium profile, we avoid the dramatic
 * interface-thickening transient that occurs when sharp (step-function)
 * boundaries relax to the equilibrium width.
 *
 * Partition of unity: η_owner + η_second = 1 everywhere.
 */
export function generateSmoothVoronoiField(
  gridSize: number,
  numGrains: number,
  xi_px: number,
): Float32Array {
  const WH = gridSize * gridSize;
  const packed = new Float32Array(numGrains * WH);

  const seeds = placeSeeds(gridSize, numGrains);

  // Tanh profile width parameter: ξ/√2
  const w = xi_px / Math.SQRT2;

  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      // Find nearest and second-nearest seeds
      let d1 = Infinity, d2 = Infinity;
      let owner = 0, second = 0;

      for (let i = 0; i < numGrains; i++) {
        const ddx = x - seeds[i][0];
        const ddy = y - seeds[i][1];
        const dist = Math.sqrt(ddx * ddx + ddy * ddy);
        if (dist < d1) {
          d2 = d1; second = owner;
          d1 = dist; owner = i;
        } else if (dist < d2) {
          d2 = dist; second = i;
        }
      }

      // Signed distance from boundary: positive inside owner grain
      const halfGap = (d2 - d1) / 2;

      // Smooth tanh transition at equilibrium width
      const etaOwner = 0.5 * (1.0 + Math.tanh(halfGap / w));
      const etaSecond = 1.0 - etaOwner;

      packed[owner  * WH + y * gridSize + x] = etaOwner;
      packed[second * WH + y * gridSize + x] = etaSecond;
      // All other grains remain 0.0
    }
  }

  return packed;
}

/**
 * Generate evenly-spaced HSL colors for grain display.
 * Returns a Float32Array of size numGrains × 4 (vec4f per grain).
 * RGB values in [0, 1], alpha channel = 1.0 (padding).
 */
export function generateGrainColors(numGrains: number): Float32Array {
  const colors = new Float32Array(numGrains * 4);
  for (let i = 0; i < numGrains; i++) {
    const hue = (i / numGrains) * 360;
    const [r, g, b] = hslToRgb(hue, 0.7, 0.6);
    colors[i * 4 + 0] = r;
    colors[i * 4 + 1] = g;
    colors[i * 4 + 2] = b;
    colors[i * 4 + 3] = 1.0;
  }
  return colors;
}

/** Convert HSL to linear RGB. H in degrees, S and L in [0,1]. */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;

  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }

  return [r + m, g + m, b + m];
}
