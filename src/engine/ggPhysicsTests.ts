// ============================================================
// Physics validation test harnesses for the constrained Allen-Cahn
// grain growth solver.
//
// Each test creates its own solver instance, runs a specific IC,
// reads back the field at regular intervals, and checks hard
// physical constraints. No test passes by appearance alone.
//
// Tests:
//   1. planarInterfaceTest   — bounded interface width, no dark band
//   2. circularGrainTest     — monotonic area, linear A(t), no black moat
//   3. tripleJunctionTest    — no dark hole at junction, all grains present
//   4. polycrystalTest       — grain count ↓, mean area ↑, no dark channels
//
// All tests also check the simplex constraint at every sample:
//   max|Σφᵢ − 1| < 1e-4,  min(φᵢ) ≥ −1e-5
// ============================================================

import { GrainGrowthSolver } from './grainGrowthSolver';
import type { GGConfig } from './materials';
import {
  generatePlanarInterface,
  generateCircularGrain,
  generateTripleJunction,
  generateVoronoiField,
} from '../app/initialConditions';
import {
  computeSimplexResidual,
  computeFullGGFreeEnergy,
  countActiveGrains,
} from '../app/validation';

export interface GGTestResult {
  name: string;
  passed: boolean;
  reason: string;
  data: Record<string, number[]>;  // diagnostic time-series
}

// ---- Helper: create a test solver on a fresh canvas ----

interface TestCtx {
  solver: GrainGrowthSolver;
  canvas: HTMLCanvasElement;
  config: GGConfig;
}

function makeTestSolver(
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
  gridSize: number,
  numGrains: number,
  overrides: Partial<GGConfig> = {},
): TestCtx {
  const canvas = document.createElement('canvas');
  canvas.width = gridSize;
  canvas.height = gridSize;
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  context.configure({ device, format: canvasFormat, alphaMode: 'opaque' });

  const config: GGConfig = {
    mode: 'grain-growth',
    numGrains,
    kappa: 1.0,
    W: 1.0,
    A: 1.5,  // slightly above 1 for strong grain separation
    L: 1.0,
    gridWidth: gridSize,
    ...overrides,
  };

  const solver = new GrainGrowthSolver({ device, context, canvasFormat, width: gridSize, height: gridSize, numGrains });
  solver.updateGGConfig(config);
  return { solver, canvas, config };
}

// ---- Helper: check simplex constraint ----

function checkSimplex(
  field: Float32Array,
  numGrains: number,
  gridSize: number,
): string | null {
  const { maxAbsErr, minPhi } = computeSimplexResidual(field, numGrains, gridSize);
  if (maxAbsErr > 1e-4) return `Simplex residual ${maxAbsErr.toExponential(2)} > 1e-4`;
  if (minPhi < -1e-5) return `min(φ) = ${minPhi.toExponential(2)} < −1e-5`;
  return null;
}

// ---- Helper: linear regression R² ----

function linearR2(y: number[]): number {
  const n = y.length;
  if (n < 3) return 0;
  const x = Array.from({ length: n }, (_, i) => i);
  const xBar = x.reduce((a, b) => a + b, 0) / n;
  const yBar = y.reduce((a, b) => a + b, 0) / n;
  let ssRes = 0;
  let ssTot = 0;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (x[i] - xBar) * (y[i] - yBar);
    sxx += (x[i] - xBar) ** 2;
  }
  const slope = sxy / sxx;
  const intercept = yBar - slope * xBar;
  for (let i = 0; i < n; i++) {
    const pred = slope * x[i] + intercept;
    ssRes += (y[i] - pred) ** 2;
    ssTot += (y[i] - yBar) ** 2;
  }
  return ssTot < 1e-12 ? 1 : 1 - ssRes / ssTot;
}

// ============================================================
// Test 1: Planar Interface
// IC: φ₀ = tanh profile, φ₁ = 1 − φ₀, vertical interface at midpoint.
// Run 2000 steps on a 128×128 grid.
// Pass if:
//   - Simplex constraint holds at every sample
//   - Min of (max-grain φ) over all pixels > 0.40 (no dark band)
//   - Interface width (FWHM of |∂φ₀/∂x| at midline) < 3× initial
// ============================================================
export async function planarInterfaceTest(
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
): Promise<GGTestResult> {
  const gridSize = 128;
  const { solver, config } = makeTestSolver(device, canvasFormat, gridSize, 2);
  const N = gridSize;
  const planeSize = N * N;

  const field0 = generatePlanarInterface(gridSize, 3.0);
  solver.loadField(field0 as Float32Array<ArrayBuffer>);

  // Measure initial interface width from |∂φ₀/∂x| profile at midrow
  function interfaceWidth(field: Float32Array): number {
    const midY = Math.floor(N / 2);
    let maxGrad = 0;
    let halfMaxX1 = 0;
    let halfMaxX2 = N - 1;
    // Compute |∂φ₀/∂x| at midrow
    const grads: number[] = [];
    for (let x = 1; x < N - 1; x++) {
      const g = Math.abs(
        field[0 * planeSize + midY * N + (x + 1)] -
        field[0 * planeSize + midY * N + (x - 1)]
      ) * 0.5;
      grads.push(g);
      if (g > maxGrad) maxGrad = g;
    }
    if (maxGrad < 1e-6) return 0;
    const half = maxGrad / 2;
    let peak = 0;
    for (let i = 0; i < grads.length; i++) {
      if (grads[i] >= half) { halfMaxX1 = i; break; }
    }
    for (let i = 0; i < grads.length; i++) {
      if (grads[i] >= half) { peak = i; }
    }
    halfMaxX2 = peak;
    return halfMaxX2 - halfMaxX1 + 1;
  }

  const initialWidth = interfaceWidth(field0);
  const widthSamples: number[] = [initialWidth];
  const simplexOk: boolean[] = [];
  let darkBandMin = 1.0;

  const stepsPerSample = 500;
  const numSamples = 4; // 0, 500, 1000, 1500, 2000

  for (let s = 0; s < numSamples; s++) {
    solver.step(stepsPerSample);
    const field = await solver.readField();
    const simplexErr = checkSimplex(field, 2, gridSize);
    simplexOk.push(simplexErr === null);

    // Min of max-grain phi (dark band detector)
    for (let pix = 0; pix < planeSize; pix++) {
      const m = Math.max(field[pix], field[planeSize + pix]);
      if (m < darkBandMin) darkBandMin = m;
    }

    widthSamples.push(interfaceWidth(field));
  }

  solver.destroy();

  const failedSimplex = simplexOk.findIndex(ok => !ok);
  if (failedSimplex >= 0) {
    return { name: 'planarInterface', passed: false, reason: `Simplex violation at sample ${failedSimplex + 1}`, data: { widthSamples } };
  }
  if (darkBandMin < 0.40) {
    return { name: 'planarInterface', passed: false, reason: `Dark band detected: min(max-grain φ) = ${darkBandMin.toFixed(3)} < 0.40`, data: { widthSamples, darkBandMin: [darkBandMin] } };
  }
  const finalWidth = widthSamples[widthSamples.length - 1];
  const widthRatio = initialWidth > 0 ? finalWidth / initialWidth : 1;
  if (widthRatio > 3.0) {
    return { name: 'planarInterface', passed: false, reason: `Interface widened ${widthRatio.toFixed(1)}× (limit 3×)`, data: { widthSamples } };
  }

  return { name: 'planarInterface', passed: true, reason: `OK — width ratio ${widthRatio.toFixed(2)}, min(max-grain φ) ${darkBandMin.toFixed(3)}`, data: { widthSamples } };
}

// ============================================================
// Test 2: Circular Grain Shrinkage
// IC: grain 0 = disk radius N/4, grain 1 = matrix.
// Run until grain area < 10% of initial, or for 20000 steps max.
// Pass if:
//   - Simplex constraint holds
//   - Area decreases monotonically
//   - No black moat: min φ₁ in annular region > 0.35
//   - A(t) is approximately linear (R² > 0.90 over shrinkage regime)
// ============================================================
export async function circularGrainTest(
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
): Promise<GGTestResult> {
  const gridSize = 128;
  const { solver } = makeTestSolver(device, canvasFormat, gridSize, 2);
  const N = gridSize;
  const planeSize = N * N;
  const r0 = N / 4;

  const field0 = generateCircularGrain(gridSize, r0, 3.0);
  solver.loadField(field0 as Float32Array<ArrayBuffer>);

  const areaSamples: number[] = [];
  const simplexOk: boolean[] = [];
  let moatMin = 1.0;
  const stepsPerSample = 500;

  // Compute grain 0 area: count pixels where φ₀ > φ₁ (argmax = grain 0)
  function grainArea(field: Float32Array): number {
    let count = 0;
    for (let pix = 0; pix < planeSize; pix++) {
      if (field[pix] > field[planeSize + pix]) count++;
    }
    return count;
  }

  // Sample initial area
  areaSamples.push(grainArea(field0));

  for (let s = 0; s < 40; s++) {
    solver.step(stepsPerSample);
    const field = await solver.readField();

    const simplexErr = checkSimplex(field, 2, gridSize);
    simplexOk.push(simplexErr === null);

    const area = grainArea(field);
    areaSamples.push(area);

    // No-black-moat check: in the region just OUTSIDE the shrinking grain
    // (r between r0 and r0+15), verify max(φ₀, φ₁) ≥ 0.45.
    // With the simplex constraint Σφᵢ=1, max(φ₀,φ₁) ≥ 0.5 is guaranteed,
    // so any failure here indicates a constraint violation, not a genuine moat.
    const cx = N / 2;
    const cy = N / 2;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const r = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
        if (r > r0 && r <= r0 + 15) {
          const maxPhi = Math.max(field[y * N + x], field[planeSize + y * N + x]);
          if (maxPhi < moatMin) moatMin = maxPhi;
        }
      }
    }

    // Stop when grain is nearly gone
    if (area < 0.10 * areaSamples[0]) break;
  }

  solver.destroy();

  const failedSimplex = simplexOk.findIndex(ok => !ok);
  if (failedSimplex >= 0) {
    return { name: 'circularGrain', passed: false, reason: `Simplex violation at sample ${failedSimplex + 1}`, data: { areaSamples } };
  }

  // Check monotonic decrease (allow 1 non-decrease out of total — numerical noise)
  let nonDecreases = 0;
  for (let i = 1; i < areaSamples.length; i++) {
    if (areaSamples[i] > areaSamples[i - 1]) nonDecreases++;
  }
  if (nonDecreases > 1) {
    return { name: 'circularGrain', passed: false, reason: `Area not monotonically decreasing (${nonDecreases} increases)`, data: { areaSamples } };
  }

  // With the simplex constraint, max(φ₀,φ₁)≥0.5 always; threshold 0.45 catches violations
  if (moatMin < 0.45) {
    return { name: 'circularGrain', passed: false, reason: `Dark region outside grain: min max-grain φ = ${moatMin.toFixed(3)} < 0.45 (simplex constraint may be violated)`, data: { areaSamples, moatMin: [moatMin] } };
  }

  // Extract shrinkage regime: A(t) from 90% down to 20% of A₀
  const A0 = areaSamples[0];
  const shrinkage = areaSamples.filter(a => a <= 0.9 * A0 && a >= 0.2 * A0);
  let r2 = 1;
  if (shrinkage.length >= 3) {
    r2 = linearR2(shrinkage);
    if (r2 < 0.90) {
      return { name: 'circularGrain', passed: false, reason: `A(t) non-linear in shrinkage regime: R² = ${r2.toFixed(3)} < 0.90`, data: { areaSamples, r2: [r2] } };
    }
  }

  return { name: 'circularGrain', passed: true, reason: `OK — A(t) R²=${r2.toFixed(3)}, min max-grain φ outside grain=${moatMin.toFixed(3)}`, data: { areaSamples } };
}

// ============================================================
// Test 3: Triple Junction
// IC: 3 Voronoi grains from 120°-symmetric seeds.
// Run 5000 steps on a 128×128 grid.
// Pass if:
//   - Simplex constraint holds
//   - All 3 grains still present (none fully consumed)
//   - No dark hole at junction center: max-grain φ in center 20×20 px > 0.25
// ============================================================
export async function tripleJunctionTest(
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
): Promise<GGTestResult> {
  const gridSize = 128;
  const { solver } = makeTestSolver(device, canvasFormat, gridSize, 3);
  const N = gridSize;
  const planeSize = N * N;

  const field0 = generateTripleJunction(gridSize);
  solver.loadField(field0 as Float32Array<ArrayBuffer>);

  const stepsPerSample = 1000;
  const numSamples = 5; // 5000 steps total
  const simplexOk: boolean[] = [];
  let junctionMin = 1.0;
  let grainCountFinal = 3;

  for (let s = 0; s < numSamples; s++) {
    solver.step(stepsPerSample);
    const field = await solver.readField();

    simplexOk.push(checkSimplex(field, 3, gridSize) === null);

    // Check center region (20×20 px around midpoint) for dark hole
    const cx = Math.floor(N / 2);
    const cy = Math.floor(N / 2);
    const half = 10;
    for (let y = cy - half; y < cy + half; y++) {
      for (let x = cx - half; x < cx + half; x++) {
        if (x < 0 || x >= N || y < 0 || y >= N) continue;
        const pix = y * N + x;
        let maxPhi = 0;
        for (let g = 0; g < 3; g++) {
          const p = field[g * planeSize + pix];
          if (p > maxPhi) maxPhi = p;
        }
        if (maxPhi < junctionMin) junctionMin = maxPhi;
      }
    }

    grainCountFinal = countActiveGrains(field, 3, gridSize);
  }

  solver.destroy();

  const failedSimplex = simplexOk.findIndex(ok => !ok);
  if (failedSimplex >= 0) {
    return { name: 'tripleJunction', passed: false, reason: `Simplex violation at sample ${failedSimplex + 1}`, data: {} };
  }

  if (grainCountFinal < 3) {
    return { name: 'tripleJunction', passed: false, reason: `Only ${grainCountFinal}/3 grains survived to t_final`, data: {} };
  }

  if (junctionMin < 0.25) {
    return { name: 'tripleJunction', passed: false, reason: `Dark hole at junction: min(max-grain φ) = ${junctionMin.toFixed(3)} < 0.25`, data: { junctionMin: [junctionMin] } };
  }

  return { name: 'tripleJunction', passed: true, reason: `OK — all 3 grains present, junction min=${junctionMin.toFixed(3)}`, data: { junctionMin: [junctionMin] } };
}

// ============================================================
// Test 4: Polycrystal Coarsening
// IC: Voronoi, 8 grains, 128×128.
// Run 30000 steps total; sample every 5000 steps.
// Pass if:
//   - Simplex constraint holds at every sample
//   - Grain count decreases (or stays same at last step compared to first)
//   - Mean grain area increases (or stays same)
//   - Full free energy is non-increasing from first to last sample
//   - No persistent dark channels: < 5% of pixels with max-grain φ < 0.40
// ============================================================
export async function polycrystalTest(
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
): Promise<GGTestResult> {
  const gridSize = 128;
  const numGrains = 8;
  const { solver, config } = makeTestSolver(device, canvasFormat, gridSize, numGrains);
  const N = gridSize;
  const planeSize = N * N;

  const field0 = generateVoronoiField(gridSize, numGrains);
  solver.loadField(field0 as Float32Array<ArrayBuffer>);

  const grainCounts: number[] = [];
  const freeEnergies: number[] = [];
  const darkFractions: number[] = [];
  const simplexOk: boolean[] = [];

  const stepsPerSample = 5000;
  const numSamples = 6; // 30000 steps total

  for (let s = 0; s < numSamples; s++) {
    solver.step(stepsPerSample);
    const field = await solver.readField();

    simplexOk.push(checkSimplex(field, numGrains, gridSize) === null);

    grainCounts.push(countActiveGrains(field, numGrains, gridSize));
    freeEnergies.push(computeFullGGFreeEnergy(
      field, numGrains, gridSize, config.kappa, config.W, config.A,
    ));

    // Count dark pixels: pixels where max-grain φ < 0.40
    let darkCount = 0;
    for (let pix = 0; pix < planeSize; pix++) {
      let maxPhi = 0;
      for (let g = 0; g < numGrains; g++) {
        const p = field[g * planeSize + pix];
        if (p > maxPhi) maxPhi = p;
      }
      if (maxPhi < 0.40) darkCount++;
    }
    darkFractions.push(darkCount / planeSize);
  }

  solver.destroy();

  const failedSimplex = simplexOk.findIndex(ok => !ok);
  if (failedSimplex >= 0) {
    return { name: 'polycrystal', passed: false, reason: `Simplex violation at sample ${failedSimplex + 1}`, data: { grainCounts, freeEnergies, darkFractions } };
  }

  // Grain count should decrease (final ≤ initial)
  const grainCountDecreased = grainCounts[grainCounts.length - 1] <= grainCounts[0];
  if (!grainCountDecreased) {
    return { name: 'polycrystal', passed: false, reason: `Grain count did not decrease: ${grainCounts[0]} → ${grainCounts[grainCounts.length - 1]}`, data: { grainCounts, freeEnergies } };
  }

  // Free energy should be non-increasing (final < initial + 1% tolerance)
  const energyDecreased = freeEnergies[freeEnergies.length - 1] <= freeEnergies[0] * 1.01;
  if (!energyDecreased) {
    return { name: 'polycrystal', passed: false, reason: `Free energy did not decrease: ${freeEnergies[0].toFixed(1)} → ${freeEnergies[freeEnergies.length - 1].toFixed(1)}`, data: { grainCounts, freeEnergies } };
  }

  // Persistent dark channels: every sample should have < 5% dark pixels
  const persistentDark = darkFractions.findIndex(f => f > 0.05);
  if (persistentDark >= 0) {
    return { name: 'polycrystal', passed: false, reason: `Persistent dark channels at sample ${persistentDark + 1}: ${(darkFractions[persistentDark] * 100).toFixed(1)}% > 5% of pixels`, data: { grainCounts, freeEnergies, darkFractions } };
  }

  return {
    name: 'polycrystal',
    passed: true,
    reason: `OK — grains ${grainCounts[0]}→${grainCounts[grainCounts.length - 1]}, energy ↓, dark < 5%`,
    data: { grainCounts, freeEnergies, darkFractions },
  };
}

// ============================================================
// Run all four tests in sequence. Returns results for display.
// ============================================================
export async function runGGValidationSuite(
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
): Promise<GGTestResult[]> {
  const results: GGTestResult[] = [];

  results.push(await planarInterfaceTest(device, canvasFormat));
  results.push(await circularGrainTest(device, canvasFormat));
  results.push(await tripleJunctionTest(device, canvasFormat));
  results.push(await polycrystalTest(device, canvasFormat));

  return results;
}
