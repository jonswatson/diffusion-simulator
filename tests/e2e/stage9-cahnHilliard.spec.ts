import { test, expect } from '@playwright/test';
import { waitForEngine } from './helpers';

// Dimensionless Cahn-Hilliard config: domainSize = gridWidth → dx = 1.0
// ε = 0.5 keeps most Fourier modes inside the spinodal (k_crit = √(A/ε²) = 2),
// so phase separation is fast and observable in a small number of steps.
function makeCHConfig(W: number) {
  return {
    mode: 'cahn-hilliard' as const,
    mobility: 1.0,
    epsilon: 0.5,
    barrierHeight: 1.0,
    domainSize_m: W, // dx = W/W = 1.0 (dimensionless)
    gridWidth: W,
  };
}

test.describe('Stage 9: Cahn-Hilliard Solver', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForEngine(page);
  });

  test('CH solver can be instantiated and runs without error', async ({ page }) => {
    const result = await page.evaluate(async (cfg) => {
      const sim = (window as any).__sim;
      const { device, CahnHilliardSolver, canvasFormat, context } = sim;

      const W = cfg.gridWidth;
      const chSolver = new CahnHilliardSolver({ device, context, canvasFormat, width: W, height: W });
      chSolver.updateCHConfig(cfg);

      // Generate spinodal noise field (mean 0.5, small perturbation)
      const field = new Float32Array(W * W);
      for (let i = 0; i < field.length; i++) {
        field[i] = 0.5 + 0.05 * (Math.random() * 2 - 1);
      }
      chSolver.loadField(field);

      // Step 200 times in batches
      const batchSize = 50;
      for (let remaining = 200; remaining > 0; remaining -= batchSize) {
        chSolver.step(Math.min(batchSize, remaining));
      }
      await device.queue.onSubmittedWorkDone();

      const result = await chSolver.readField();
      chSolver.destroy();

      return {
        length: result.length,
        mode: chSolver.mode,
        min: Math.min(...Array.from(result)),
        max: Math.max(...Array.from(result)),
      };
    }, makeCHConfig(64));

    expect(result.length).toBe(64 * 64);
    expect(result.mode).toBe('cahn-hilliard');
    expect(result.min).toBeGreaterThanOrEqual(-0.1); // slight undershoot OK for explicit
    expect(result.max).toBeLessThanOrEqual(1.1);
  });

  test('mass is conserved during spinodal decomposition (< 0.5%)', async ({ page }) => {
    const result = await page.evaluate(async (cfg) => {
      const sim = (window as any).__sim;
      const { device, CahnHilliardSolver, canvasFormat, context } = sim;

      const W = cfg.gridWidth;
      const chSolver = new CahnHilliardSolver({ device, context, canvasFormat, width: W, height: W });
      chSolver.updateCHConfig(cfg);

      const field = new Float32Array(W * W);
      let initMass = 0;
      for (let i = 0; i < field.length; i++) {
        field[i] = 0.5 + 0.05 * (Math.random() * 2 - 1);
        initMass += field[i];
      }
      chSolver.loadField(field);

      // Step 200 times
      const batchSize = 50;
      for (let remaining = 200; remaining > 0; remaining -= batchSize) {
        chSolver.step(Math.min(batchSize, remaining));
      }
      await device.queue.onSubmittedWorkDone();

      const finalField = await chSolver.readField();
      let finalMass = 0;
      for (let i = 0; i < finalField.length; i++) finalMass += finalField[i];

      chSolver.destroy();

      return {
        initMass,
        finalMass,
        drift: Math.abs(finalMass - initMass) / initMass,
      };
    }, makeCHConfig(64));

    expect(result.drift).toBeLessThan(0.005); // < 0.5%
  });

  test('pixel variance increases after stepping (phase separation)', async ({ page }) => {
    const result = await page.evaluate(async (cfg) => {
      const sim = (window as any).__sim;
      const { device, CahnHilliardSolver, canvasFormat, context } = sim;

      const W = cfg.gridWidth;
      const chSolver = new CahnHilliardSolver({ device, context, canvasFormat, width: W, height: W });
      chSolver.updateCHConfig(cfg);

      // Small noise around 0.5 — must be inside the spinodal region
      const field = new Float32Array(W * W);
      for (let i = 0; i < field.length; i++) {
        field[i] = 0.5 + 0.05 * (Math.random() * 2 - 1);
      }
      chSolver.loadField(field);

      // Compute initial variance
      const initField = await chSolver.readField();
      const N = initField.length;
      let initMean = 0;
      for (let i = 0; i < N; i++) initMean += initField[i];
      initMean /= N;
      let initVar = 0;
      for (let i = 0; i < N; i++) {
        const d = initField[i] - initMean;
        initVar += d * d;
      }
      initVar /= N;

      // Step many times to allow phase separation
      const batchSize = 200;
      for (let remaining = 2000; remaining > 0; remaining -= batchSize) {
        chSolver.step(Math.min(batchSize, remaining));
      }
      await device.queue.onSubmittedWorkDone();

      // Compute final variance
      const finalField = await chSolver.readField();
      let finalMean = 0;
      for (let i = 0; i < N; i++) finalMean += finalField[i];
      finalMean /= N;
      let finalVar = 0;
      for (let i = 0; i < N; i++) {
        const d = finalField[i] - finalMean;
        finalVar += d * d;
      }
      finalVar /= N;

      chSolver.destroy();

      return { initVar, finalVar };
    }, makeCHConfig(64));

    // Phase separation should increase variance as domains form
    expect(result.finalVar).toBeGreaterThan(result.initVar);
  });
});
