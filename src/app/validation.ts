// ============================================================
// Validation utilities for the diffusion engine.
//
// Mass conservation: sum concentration at load, compare periodically.
// Analytical validation: midline erfc comparison for 1D step IC.
// Mode-aware: analytical check only for Fick mode.
// ============================================================

import type { DiffusionEngine, EngineState, SimMode } from '../engine/types';
import { analyticalConcentration } from '../engine/physics';

export interface ValidationState {
  initialMass: number;
  lastMassError: number;       // fractional: |current - initial| / initial
  lastAnalyticalRMS: number;   // RMS error vs erfc solution on midline
  lastCheckStep: number;
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

/** Create a validation state tracker. Call setInitialMass after loadField. */
export function createValidation(): ValidationState {
  return {
    initialMass: 0,
    lastMassError: 0,
    lastAnalyticalRMS: 0,
    lastCheckStep: 0,
  };
}

/**
 * Run periodic validation checks. Call every frame; internally throttles
 * to only check every 500 steps.
 *
 * The readField callback is async (GPU readback) so this returns a promise.
 * Mode parameter controls whether analytical (erfc) comparison is run.
 */
export async function runValidationChecks(
  validation: ValidationState,
  solver: DiffusionEngine,
  gridWidth: number,
  mode: SimMode = 'fick',
): Promise<void> {
  const state = solver.state;
  const stepsSinceCheck = state.stepsRun - validation.lastCheckStep;

  // Only check every 500 steps to avoid GPU readback overhead
  if (stepsSinceCheck < 500 || state.stepsRun === 0) return;

  validation.lastCheckStep = state.stepsRun;

  const field = await solver.readField();

  // Mass conservation (applies to all modes, but for GG use first W*H slice)
  if (mode === 'grain-growth') {
    // For grain growth, mass check on the first eta slice (informational)
    const sliceLen = gridWidth * gridWidth;
    const slice = field.subarray(0, sliceLen);
    const currentMass = fieldSum(slice);
    if (validation.initialMass > 0) {
      validation.lastMassError = Math.abs(currentMass - validation.initialMass) / validation.initialMass;
    }
  } else {
    const currentMass = fieldSum(field);
    if (validation.initialMass > 0) {
      validation.lastMassError = Math.abs(currentMass - validation.initialMass) / validation.initialMass;
    }
  }

  // Analytical comparison (only valid for Fick default step-function IC)
  if (mode === 'fick' && state.time > 0) {
    validation.lastAnalyticalRMS = midlineRMSError(field, gridWidth, state);
  }
}
