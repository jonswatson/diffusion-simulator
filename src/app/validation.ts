// ============================================================
// Validation utilities for the diffusion engine.
//
// Mass conservation: sum concentration at load, compare periodically.
// Analytical validation: midline erfc comparison for 1D step IC.
// Grain growth: bulk free energy and grain count diagnostics.
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
  lastFreeEnergy: number;      // bulk + interaction free energy (no gradient term)
  lastGrainCount: number;      // grains with at least one dominant pixel
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
 * Bulk + interaction free energy for the Allen-Cahn multi-grain model.
 * F_bulk = Σ_pix Σ_i  W/4·ηᵢ²·(1−ηᵢ)²
 * F_inter = Σ_pix Σ_{i<j} A·ηᵢ²·ηⱼ²
 * Gradient term omitted (needs finite differences; expensive on CPU).
 * This quantity still decreases monotonically after the initial transient.
 */
export function grainBulkFreeEnergy(
  field: Float32Array,
  numGrains: number,
  gridSize: number,
  W: number,
  A: number,
): number {
  const planeSize = gridSize * gridSize;
  let F = 0;
  for (let pix = 0; pix < planeSize; pix++) {
    for (let i = 0; i < numGrains; i++) {
      const ei = field[i * planeSize + pix];
      F += (W / 4) * ei * ei * (1 - ei) * (1 - ei);
      for (let j = i + 1; j < numGrains; j++) {
        const ej = field[j * planeSize + pix];
        F += A * ei * ei * ej * ej;
      }
    }
  }
  return F;
}

/**
 * Count grains that are dominant (maxEta > 0.5) in at least one pixel.
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
    let maxEta = 0;
    let best = 0;
    for (let g = 0; g < numGrains; g++) {
      const e = field[g * planeSize + pix];
      if (e > maxEta) { maxEta = e; best = g; }
    }
    if (maxEta > 0.5) present.add(best);
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
  };
}

export interface GGValidationParams {
  numGrains: number;
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
    validation.lastFreeEnergy = grainBulkFreeEnergy(
      field, ggParams.numGrains, gridWidth, ggParams.W, ggParams.A,
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
