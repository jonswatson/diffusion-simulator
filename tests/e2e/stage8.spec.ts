import { test, expect } from '@playwright/test';
import { waitForEngine, stepEngine, readField, getEngineState } from './helpers';

test.describe('Stage 8: Validation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForEngine(page);
  });

  test('mass conservation: drift < 0.1% after 1000 steps', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__sim.engine;

      // Record initial mass
      const initField: Float32Array = await engine.readField();
      let initMass = 0;
      for (let i = 0; i < initField.length; i++) initMass += initField[i];

      // Step 1000 times in batches
      const batchSize = 100;
      for (let remaining = 1000; remaining > 0; remaining -= batchSize) {
        engine.step(Math.min(batchSize, remaining));
      }
      await engine.device.queue.onSubmittedWorkDone();

      // Read final mass
      const finalField: Float32Array = await engine.readField();
      let finalMass = 0;
      for (let i = 0; i < finalField.length; i++) finalMass += finalField[i];

      return {
        initMass,
        finalMass,
        drift: Math.abs(finalMass - initMass) / initMass,
      };
    });

    expect(result.drift).toBeLessThan(0.001); // < 0.1%
  });

  test('analytical validation: RMS error < 5% on midline after 500 steps', async ({ page }) => {
    const rms = await page.evaluate(async () => {
      const engine = (window as any).__sim.engine;
      const W = 256;

      // Step 500 times
      const batchSize = 100;
      for (let remaining = 500; remaining > 0; remaining -= batchSize) {
        engine.step(Math.min(batchSize, remaining));
      }
      await engine.device.queue.onSubmittedWorkDone();

      const field: Float32Array = await engine.readField();
      const state = engine.state;
      const midY = Math.floor(W / 2);
      const x0_m = (W - 1) / 2.0 * state.dx;

      // erfc approximation
      function erfcApprox(x: number): number {
        const t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
        const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        const result = poly * Math.exp(-x * x);
        return x >= 0 ? result : 2.0 - result;
      }

      let sumSqErr = 0;
      for (let x = 0; x < W; x++) {
        const x_m = x * state.dx;
        const numerical = field[midY * W + x];
        const analytical = 0.5 * erfcApprox((x_m - x0_m) / (2.0 * Math.sqrt(state.diffusivity * state.time)));
        const err = numerical - analytical;
        sumSqErr += err * err;
      }

      return Math.sqrt(sumSqErr / W);
    });

    expect(rms).toBeLessThan(0.05); // < 5% RMS error
  });

  test('validation panel is visible in info panel', async ({ page }) => {
    const massError = page.locator('#info-mass-error');
    const rmsError = page.locator('#info-rms-error');

    await expect(massError).toBeVisible();
    await expect(rmsError).toBeVisible();
  });

  test('temperature out-of-range warning fires correctly', async ({ page }) => {
    // Set temperature below Cu-in-Al range (400–900 K)
    await page.locator('#sl-temperature').fill('350');
    await page.locator('#sl-temperature').dispatchEvent('input');
    await page.waitForTimeout(100);

    const warnings = page.locator('#warnings .warning');
    const count = await warnings.count();
    expect(count).toBeGreaterThan(0);

    const text = await warnings.first().textContent();
    expect(text).toContain('outside the validated range');
  });
});
