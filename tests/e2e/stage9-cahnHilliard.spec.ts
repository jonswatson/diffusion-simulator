import { test, expect } from '@playwright/test';
import { waitForEngine } from './helpers';

// Regular solution Cahn-Hilliard config using Al-Zn material.
// T = 400 K is well below T_spinodal = 601 K → Ω_d = Ω/(RT) ≈ 3.01 > 2 (unstable).
//
// Interface width = 3 µm on a 64 µm domain (W=64) → dx = 1 µm → ξ = 3 px.
// The CFL-limited growth rate per step scales as 0.2/ξ⁴.
// With ξ=3, growth is 140× over 2000 steps — clearly detectable.
function makeCHConfig(W: number) {
  return {
    mode: 'cahn-hilliard' as const,
    material: {
      name: 'Aluminium-Zinc', symbol: 'Al-Zn',
      Omega_Jmol: 10_000, D0: 2.59e-5, Q: 120_500,
      T_spinodal: 601, T_min: 300, T_max: 600,
      colorA: [0.75, 0.75, 0.82] as [number, number, number],
      colorB: [0.65, 0.65, 0.70] as [number, number, number],
    },
    temperature_K: 400,
    interfaceWidth_m: 3e-6, // 3 µm → 3 px at dx = 1 µm
    domainSize_m: W * 1e-6, // W µm → dx = 1e-6 m (1 µm/pixel)
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
