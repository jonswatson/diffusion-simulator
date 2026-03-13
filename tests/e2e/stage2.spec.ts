import { test, expect } from '@playwright/test';
import { waitForEngine, readField, stepEngine, fieldMass } from './helpers';

test.describe('Stage 2: Diffusion Shader', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForEngine(page);
  });

  test('initial field has ~50% total mass (symmetric IC)', async ({ page }) => {
    const field = await readField(page);
    const mass = fieldMass(field);
    const expected = field.length * 0.5;
    expect(mass / expected).toBeCloseTo(1.0, 2);
  });

  test('all initial values are in [0, 1]', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const field: Float32Array = await (window as any).__sim.engine.readField();
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < field.length; i++) {
        if (field[i] < min) min = field[i];
        if (field[i] > max) max = field[i];
      }
      return { min, max };
    });
    expect(result.min).toBeGreaterThanOrEqual(-0.001);
    expect(result.max).toBeLessThanOrEqual(1.001);
  });

  test('diffusion spreads the interface — left half mean decreases after 500 steps', async ({ page }) => {
    // The left half is initially ~1.0. After diffusion, cells near the interface lose
    // concentration to the right half, so the left-half mean should decrease.
    const result = await page.evaluate(async () => {
      const engine = (window as any).__sim.engine;
      const device = (window as any).__sim.device;

      const before: Float32Array = await engine.readField();
      const N = Math.round(Math.sqrt(before.length));
      const half = Math.floor(N / 2);

      let leftSumBefore = 0;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < half; x++) leftSumBefore += before[y * N + x];
      }
      const leftMeanBefore = leftSumBefore / (N * half);

      engine.step(500);
      await device.queue.onSubmittedWorkDone();

      const after: Float32Array = await engine.readField();
      let leftSumAfter = 0;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < half; x++) leftSumAfter += after[y * N + x];
      }
      const leftMeanAfter = leftSumAfter / (N * half);

      return { leftMeanBefore, leftMeanAfter };
    });

    expect(result.leftMeanAfter).toBeLessThan(result.leftMeanBefore);
  });

  test('mass is conserved after 500 steps (< 0.1%)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__sim.engine;
      const device = (window as any).__sim.device;

      const before: Float32Array = await engine.readField();
      const massBefore = before.reduce((a: number, b: number) => a + b, 0);

      engine.step(500);
      await device.queue.onSubmittedWorkDone();

      const after: Float32Array = await engine.readField();
      const massAfter = after.reduce((a: number, b: number) => a + b, 0);

      return { massBefore, massAfter };
    });

    const relativeError = Math.abs(result.massAfter - result.massBefore) / result.massBefore;
    expect(relativeError).toBeLessThan(0.001);
  });

  test('midpoint cell remains ≈ 0.5 throughout (symmetry)', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__sim.engine;
      const device = (window as any).__sim.device;

      const before: Float32Array = await engine.readField();
      const N = Math.round(Math.sqrt(before.length));
      const midIdx = Math.floor(N / 2) * N + Math.floor(N / 2);
      const midBefore = before[midIdx];

      engine.step(1000);
      await device.queue.onSubmittedWorkDone();

      const after: Float32Array = await engine.readField();
      const midAfter = after[midIdx];

      return { midBefore, midAfter };
    });

    expect(result.midBefore).toBeCloseTo(0.5, 1);
    expect(result.midAfter).toBeCloseTo(0.5, 1);
  });

  test('boundary cells do not exceed [0, 1] — zero-flux BC holds', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const engine = (window as any).__sim.engine;
      const device = (window as any).__sim.device;

      engine.step(1000);
      await device.queue.onSubmittedWorkDone();

      const field: Float32Array = await engine.readField();
      const N = Math.round(Math.sqrt(field.length));

      let min = Infinity, max = -Infinity;
      for (let i = 0; i < N; i++) {
        const vals = [
          field[0 * N + i],         // top
          field[(N - 1) * N + i],   // bottom
          field[i * N + 0],         // left
          field[i * N + (N - 1)],   // right
        ];
        for (const v of vals) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      return { min, max };
    });

    expect(result.min).toBeGreaterThanOrEqual(-0.001);
    expect(result.max).toBeLessThanOrEqual(1.001);
  });
});
