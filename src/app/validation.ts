// ============================================================
// Validation utilities for the diffusion engine.
//
// Mass conservation: sum concentration at load, compare periodically.
// Analytical validation: midline erfc comparison for 1D step IC.
// Grain growth: simplex residual, min/max phi, full free energy, grain count.
// Mode-aware: analytical check only for Fick mode.
// ============================================================

import type { DiffusionEngine, EngineState, SimMode } from '../engine/types';
import { analyticalConcentration } from '../engine/physics';

export interface ValidationState {
  initialMass: number;
  lastMassError: number;       // fractional: |current - initial| / initial
  lastAnalyticalRMS: number;   // RMS error vs erfc solution on midline
  lastCheckStep: number;
  // Grain growth diagnostics
  lastFreeEnergy: number;      // full free energy (gradient + bulk + interaction)
  lastGrainCount: number;      // grains with at least one dominant pixel
  lastSimplexResidual: number; // max |Σᵢ φᵢ(x,y) − 1| over all pixels
  lastMinPhi: number;          // global min of any φᵢ (should be ≥ 0)
  lastMaxPhi: number;          // global max of any φᵢ (should be ≤ 1)
}

/** Compute the total "mass" (sum of all concentration values) from a field. */
export function fieldSum(field: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < field.length; i++) sum += field[i];
  return sum;
}

/**
 * Compare the numerical midline (row at y = gridHeight/2) to the
 * analytical erfc solution for a 1D step-function IC.
 *
 * Returns the RMS error across all grid points on that row.
 */
export function midlineRMSError(
  field: Float32Array,
  gridWidth: number,
  state: EngineState,
): number {
  const midY = Math.floor(gridWidth / 2);
  const x0_m = (gridWidth - 1) / 2.0 * state.dx; // physical midpoint

  let sumSqErr = 0;
  for (let x = 0; x < gridWidth; x++) {
    const x_m = x * state.dx;
    const numerical = field[midY * gridWidth + x];
    const analytical = analyticalConcentration(x_m, x0_m, state.diffusivity, state.time);
    const err = numerical - analytical;
    sumSqErr += err * err;
  }

  return Math.sqrt(sumSqErr / gridWidth);
}

/**
 * Simplex constraint residual for the Allen-Cahn grain growth field.
 * Returns max|Σᵢ φᵢ(x,y) − 1|, min(φᵢ), max(φᵢ) over all pixels.
 *
 * The constrained solver should keep these values near 0, ≥0, ≤1 respectively.
 */
export function computeSimplexResidual(
  field: Float32Array,
  numGrains: number,
  gridSize: number,
): { maxAbsErr: number; minPhi: number; maxPhi: number } {
  const planeSize = gridSize * gridSize;
  let maxAbsErr = 0;
  let minPhi = Infinity;
  let maxPhi = -Infinity;

  for (let pix = 0; pix < planeSize; pix++) {
    let sumPhi = 0;
    for (let g = 0; g < numGrains; g++) {
      const phi = field[g * planeSize + pix];
      sumPhi += phi;
      if (phi < minPhi) minPhi = phi;
      if (phi > maxPhi) maxPhi = phi;
    }
    const err = Math.abs(sumPhi - 1.0);
    if (err > maxAbsErr) maxAbsErr = err;
  }

  return { maxAbsErr, minPhi, maxPhi };
}

/**
 * Full free energy for the constrained Allen-Cahn grain growth model.
 *
 * F = Σ_pix [ Σᵢ κ/2·|∇φᵢ|² + Σᵢ W/4·φᵢ²(1−φᵢ)² + Σᵢ<ⱼ A·φᵢ²φⱼ² ]
 *
 * Gradient term uses central differences (periodic wrap at boundaries).
 * A non-increasing F is required for thermodynamically consistent evolution.
 */
export function computeFullGGFreeEnergy(
  field: Float32Array,
  numGrains: number,
  gridSize: number,
  kappa: number,
  W: number,
  A: number,
): number {
  const N = gridSize;
  const planeSize = N * N;
  let F = 0;

  for (let y = 0; y < N; y++) {
    const yp = (y + 1) % N;
    const ym = (y + N - 1) % N;
    for (let x = 0; x < N; x++) {
      const xp = (x + 1) % N;
      const xm = (x + N - 1) % N;

      for (let i = 0; i < numGrains; i++) {
        const base = i * planeSize;
        const c = field[base + y * N + x];

        // Gradient energy: κ/2 · (|∂φ/∂x|² + |∂φ/∂y|²) — central difference
        const dpdx = (field[base + y * N + xp] - field[base + y * N + xm]) * 0.5;
        const dpdy = (field[base + yp * N + x] - field[base + ym * N + x]) * 0.5;
        F += (kappa / 2) * (dpdx * dpdx + dpdy * dpdy);

        // Double-well bulk: W/4·φᵢ²(1−φᵢ)²
        F += (W / 4) * c * c * (1 - c) * (1 - c);

        // Grain-grain interaction: A·φᵢ²·φⱼ² (summed for j > i only)
        for (let j = i + 1; j < numGrains; j++) {
          const cj = field[j * planeSize + y * N + x];
          F += A * c * c * cj * cj;
        }
      }
    }
  }

  return F;
}

/**
 * Count grains that are dominant (argmax φᵢ > 0.5) in at least one pixel.
 * A grain drops from this count once it has been fully consumed.
 */
export function countActiveGrains(
  field: Float32Array,
  numGrains: number,
  gridSize: number,
): number {
  const planeSize = gridSize * gridSize;
  const present = new Set<number>();
  for (let pix = 0; pix < planeSize; pix++) {
    let maxPhi = 0;
    let best = 0;
    for (let g = 0; g < numGrains; g++) {
      const e = field[g * planeSize + pix];
      if (e > maxPhi) { maxPhi = e; best = g; }
    }
    if (maxPhi > 0.5) present.add(best);
  }
  return present.size;
}

/** Create a validation state tracker. Call setInitialMass after loadField. */
export function createValidation(): ValidationState {
  return {
    initialMass: 0,
    lastMassError: 0,
    lastAnalyticalRMS: 0,
    lastCheckStep: 0,
    lastFreeEnergy: 0,
    lastGrainCount: 0,
    lastSimplexResidual: 0,
    lastMinPhi: 0,
    lastMaxPhi: 1,
  };
}

export interface GGValidationParams {
  numGrains: number;
  kappa: number;
  W: number;
  A: number;
}

/**
 * Run periodic validation checks. Call every frame; internally throttles
 * to only check every 500 steps.
 *
 * The readField callback is async (GPU readback) so this returns a promise.
 * Mode parameter controls which checks are run.
 */
export async function runValidationChecks(
  validation: ValidationState,
  solver: DiffusionEngine,
  gridWidth: number,
  mode: SimMode = 'fick',
  ggParams?: GGValidationParams,
): Promise<void> {
  const state = solver.state;
  const stepsSinceCheck = state.stepsRun - validation.lastCheckStep;

  // Only check every 500 steps to avoid GPU readback overhead
  if (stepsSinceCheck < 500 || state.stepsRun === 0) return;

  validation.lastCheckStep = state.stepsRun;

  const field = await solver.readField();

  if (mode === 'grain-growth' && ggParams) {
    const { maxAbsErr, minPhi, maxPhi } = computeSimplexResidual(
      field, ggParams.numGrains, gridWidth,
    );
    validation.lastSimplexResidual = maxAbsErr;
    validation.lastMinPhi = minPhi;
    validation.lastMaxPhi = maxPhi;
    validation.lastFreeEnergy = computeFullGGFreeEnergy(
      field, ggParams.numGrains, gridWidth, ggParams.kappa, ggParams.W, ggParams.A,
    );
    validation.lastGrainCount = countActiveGrains(field, ggParams.numGrains, gridWidth);
  } else {
    // Mass conservation: total concentration/composition is conserved
    const currentMass = fieldSum(field);
    if (validation.initialMass > 0) {
      validation.lastMassError = Math.abs(currentMass - validation.initialMass) / validation.initialMass;
    }

    // Analytical comparison (only valid for Fick default step-function IC)
    if (mode === 'fick' && state.time > 0) {
      validation.lastAnalyticalRMS = midlineRMSError(field, gridWidth, state);
    }
  }
}
