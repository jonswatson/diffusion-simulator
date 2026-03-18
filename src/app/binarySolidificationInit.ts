// ============================================================
// Initial condition generators for binary solidification mode.
//
// All functions return a Float32Array of length 2·N (N = gridWidth²).
// Layout: first N values = φ (phase field), second N values = c (composition).
//
// φ profile uses tanh to create a smooth diffuse interface of width xi pixels:
//   φ(r) = 0.5 · (1 + tanh((r_solid - r_dist) / xi))
// where r_solid is the distance from the solid center/boundary.
//
// Composition is set to equilibrium values in each phase:
//   c = cs_eq in solid (φ ≈ 1), c = c0 in liquid (φ ≈ 0)
// Interpolated smoothly across the interface.
// ============================================================

/** Build a 2N Float32Array from separate phi and c arrays. */
function pack(phi: Float32Array, c: Float32Array): Float32Array {
  const N = phi.length;
  const result = new Float32Array(2 * N);
  result.set(phi);
  result.set(c, N);
  return result;
}

/**
 * Circular solid seed surrounded by liquid.
 * Solid fraction φ = 1 inside radius, 0 outside, tanh interface of width xi.
 * Composition set to equilibrium values interpolated by φ.
 *
 * @param gridWidth  Grid size (square, in pixels)
 * @param seedRadius Seed radius in pixels (default = gridWidth/6)
 * @param c0         Mean composition in the liquid far from the seed
 * @param cs_eq      Equilibrium solid composition
 * @param cl_eq      Equilibrium liquid composition
 * @param xi         Interface half-width in pixels (default 3)
 */
export function generateSolidSeed(
  gridWidth: number,
  seedRadius = gridWidth / 6,
  c0 = 0.6,
  cs_eq = 0.3,
  _cl_eq = 0.7,
  xi = 3.0,
): Float32Array {
  const N = gridWidth * gridWidth;
  const phi = new Float32Array(N);
  const c   = new Float32Array(N);
  const cx = gridWidth / 2;
  const cy = gridWidth / 2;

  for (let y = 0; y < gridWidth; y++) {
    for (let x = 0; x < gridWidth; x++) {
      const r = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
      const p = 0.5 * (1.0 + Math.tanh((seedRadius - r) / xi));
      phi[y * gridWidth + x] = p;
      // Composition interpolated between equilibrium values by phi
      c[y * gridWidth + x] = p * cs_eq + (1.0 - p) * c0;
    }
  }
  return pack(phi, c);
}

/**
 * Flat solid-liquid interface at the horizontal midplane.
 * Solid (φ=1) on the left half, liquid (φ=0) on the right half.
 * Composition set to equilibrium values in each phase.
 *
 * @param gridWidth Grid size (square, in pixels)
 * @param cs_eq     Equilibrium solid composition
 * @param cl_eq     Equilibrium liquid composition
 * @param xi        Interface half-width in pixels
 */
export function generatePlanarSolidInterface(
  gridWidth: number,
  cs_eq = 0.3,
  cl_eq = 0.7,
  xi = 3.0,
): Float32Array {
  const N = gridWidth * gridWidth;
  const phi = new Float32Array(N);
  const c   = new Float32Array(N);
  const x0  = gridWidth / 2; // interface position (center column)

  for (let y = 0; y < gridWidth; y++) {
    for (let x = 0; x < gridWidth; x++) {
      // Solid on left side (x < x0)
      const p = 0.5 * (1.0 + Math.tanh((x0 - x) / xi));
      phi[y * gridWidth + x] = p;
      c[y * gridWidth + x]   = p * cs_eq + (1.0 - p) * cl_eq;
    }
  }
  return pack(phi, c);
}

/**
 * Multiple solid seeds scattered at random positions.
 * Composition set to equilibrium values interpolated by φ.
 *
 * @param gridWidth  Grid size (square, in pixels)
 * @param numSeeds   Number of seeds
 * @param seedRadius Radius of each seed in pixels
 * @param c0         Nominal liquid composition
 * @param cs_eq      Equilibrium solid composition
 * @param xi         Interface half-width in pixels
 */
export function generateMultiSeed(
  gridWidth: number,
  numSeeds = 4,
  seedRadius = gridWidth / 10,
  c0 = 0.6,
  cs_eq = 0.3,
  xi = 3.0,
): Float32Array {
  const N = gridWidth * gridWidth;
  const phi = new Float32Array(N);
  const c   = new Float32Array(N);

  // Deterministic pseudo-random seed positions (Fibonacci spiral on grid)
  const seeds: Array<[number, number]> = [];
  const goldenAngle = 2.399963; // radians
  const maxRadius = gridWidth * 0.4;
  for (let i = 0; i < numSeeds; i++) {
    const r = maxRadius * Math.sqrt((i + 0.5) / numSeeds);
    const theta = i * goldenAngle;
    seeds.push([
      gridWidth / 2 + r * Math.cos(theta),
      gridWidth / 2 + r * Math.sin(theta),
    ]);
  }

  for (let y = 0; y < gridWidth; y++) {
    for (let x = 0; x < gridWidth; x++) {
      // φ = max over all seeds (union of solid regions)
      let p = 0.0;
      for (const [sx, sy] of seeds) {
        const r = Math.sqrt((x - sx) ** 2 + (y - sy) ** 2);
        p = Math.max(p, 0.5 * (1.0 + Math.tanh((seedRadius - r) / xi)));
      }
      phi[y * gridWidth + x] = p;
      c[y * gridWidth + x]   = p * cs_eq + (1.0 - p) * c0;
    }
  }
  return pack(phi, c);
}

/**
 * Uniform liquid with sinusoidal composition perturbation (no solid).
 * Useful for testing composition diffusion without phase change.
 *
 * @param gridWidth Grid size (square, in pixels)
 * @param c0        Mean composition
 * @param amplitude Amplitude of composition wave
 */
export function generateUniformLiquid(
  gridWidth: number,
  c0 = 0.6,
  amplitude = 0.05,
): Float32Array {
  const N = gridWidth * gridWidth;
  const phi = new Float32Array(N); // all liquid (φ = 0)
  const c   = new Float32Array(N);
  const k = 2.0 * Math.PI / gridWidth; // one wavelength across the domain

  for (let y = 0; y < gridWidth; y++) {
    for (let x = 0; x < gridWidth; x++) {
      c[y * gridWidth + x] = c0 + amplitude * Math.sin(k * x) * Math.cos(k * y);
    }
  }
  return pack(phi, c);
}
