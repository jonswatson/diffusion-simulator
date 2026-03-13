import { describe, it, expect } from 'vitest';
import { fieldSum, midlineRMSError, createValidation } from '../../src/app/validation';
import type { EngineState } from '../../src/engine/solver';

describe('fieldSum', () => {
  it('sums a uniform field correctly', () => {
    const field = new Float32Array(100).fill(0.5);
    expect(fieldSum(field)).toBeCloseTo(50.0, 5);
  });

  it('returns 0 for an empty field', () => {
    expect(fieldSum(new Float32Array(0))).toBe(0);
  });

  it('returns 0 for a zero field', () => {
    const field = new Float32Array(256);
    expect(fieldSum(field)).toBe(0);
  });
});

describe('midlineRMSError', () => {
  it('returns 0 for a perfect step function at t=0', () => {
    // At t=0, the analytical solution is a step function.
    // If our field is exactly that step function, RMS should be ~0.
    const N = 64;
    const field = new Float32Array(N * N);
    const cx = (N - 1) / 2.0;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        field[y * N + x] = x < cx ? 1.0 : 0.0;
      }
    }

    const state: EngineState = {
      time: 0,
      diffusivity: 1e-12,
      dx: 100e-6 / N,
      dt: 0.001,
      r: 0.1,
      stepsRun: 0,
    };

    // At t=0, analytical is also a step function
    const rms = midlineRMSError(field, N, state);
    // The transition pixel at cx might differ slightly
    expect(rms).toBeLessThan(0.2);
  });

  it('returns small error for erfc-like profile', () => {
    // Create a field that approximates the analytical solution
    const N = 128;
    const field = new Float32Array(N * N);
    const D = 1e-12;
    const t = 1e6; // enough time for the interface to spread
    const dx = 100e-6 / N;
    const x0_m = (N - 1) / 2.0 * dx;

    const midY = Math.floor(N / 2);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        // Fill all rows with the 1D analytical solution
        const x_m = x * dx;
        const arg = (x_m - x0_m) / (2.0 * Math.sqrt(D * t));
        // erfc approximation
        const absArg = Math.abs(arg);
        const tVal = 1.0 / (1.0 + 0.3275911 * absArg);
        const poly = tVal * (0.254829592 + tVal * (-0.284496736 + tVal * (1.421413741 + tVal * (-1.453152027 + tVal * 1.061405429))));
        const erfc = arg >= 0 ? poly * Math.exp(-arg * arg) : 2.0 - poly * Math.exp(-arg * arg);
        field[y * N + x] = 0.5 * erfc;
      }
    }

    const state: EngineState = {
      time: t,
      diffusivity: D,
      dx,
      dt: 0.001,
      r: 0.1,
      stepsRun: 1000,
    };

    const rms = midlineRMSError(field, N, state);
    expect(rms).toBeLessThan(0.001); // Should be essentially 0
  });
});

describe('createValidation', () => {
  it('initializes with zero state', () => {
    const v = createValidation();
    expect(v.initialMass).toBe(0);
    expect(v.lastMassError).toBe(0);
    expect(v.lastAnalyticalRMS).toBe(0);
    expect(v.lastCheckStep).toBe(0);
  });
});
