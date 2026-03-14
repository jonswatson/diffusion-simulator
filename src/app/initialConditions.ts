// ============================================================
// Initial condition generators for all simulation modes.
//
// Spinodal: random noise around a mean composition.
// Voronoi: sharp grain assignment from random seed points.
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
 * Generate a Voronoi tessellation for grain growth.
 * Returns a packed Float32Array of size numGrains × gridSize².
 * For each cell, the owning grain has η = 1, all others η = 0.
 *
 * Seed points are placed randomly with rejection (min distance)
 * to avoid degenerate configurations with overlapping seeds.
 */
export function generateVoronoiField(
  gridSize: number,
  numGrains: number,
): Float32Array {
  const WH = gridSize * gridSize;
  const packed = new Float32Array(numGrains * WH);

  // Place seed points with some minimum spacing
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
