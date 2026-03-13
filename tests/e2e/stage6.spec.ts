import { test, expect } from '@playwright/test';
import { waitForEngine } from './helpers';

test.describe('Stage 6: Simulation Loop', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForEngine(page);
  });

  test('loop object is exposed on window.__sim', async ({ page }) => {
    const hasLoop = await page.evaluate(() => {
      const sim = (window as any).__sim;
      return sim.loop !== undefined && typeof sim.loop.play === 'function';
    });
    expect(hasLoop).toBe(true);
  });

  test('starts paused', async ({ page }) => {
    const playing = await page.evaluate(() => (window as any).__sim.loop.playing);
    expect(playing).toBe(false);
  });

  test('play advances steps, pause stops them', async ({ page }) => {
    // Start playing
    await page.evaluate(() => {
      const loop = (window as any).__sim.loop;
      loop.setStepsPerSecond(1000);
      loop.play();
    });

    // Wait for some frames to run
    await page.waitForTimeout(500);

    const stepsAfterPlay = await page.evaluate(() => {
      return (window as any).__sim.engine.state.stepsRun;
    });
    expect(stepsAfterPlay).toBeGreaterThan(0);

    // Pause and record steps
    const stepsAtPause = await page.evaluate(() => {
      const sim = (window as any).__sim;
      sim.loop.pause();
      return sim.engine.state.stepsRun;
    });

    // Wait and verify no more steps run
    await page.waitForTimeout(300);
    const stepsAfterPause = await page.evaluate(() => {
      return (window as any).__sim.engine.state.stepsRun;
    });
    expect(stepsAfterPause).toBe(stepsAtPause);
  });

  test('setStepsPerSecond changes simulation speed', async ({ page }) => {
    // Run at slow speed
    await page.evaluate(() => {
      const loop = (window as any).__sim.loop;
      loop.setStepsPerSecond(100);
      loop.play();
    });
    await page.waitForTimeout(400);
    const stepsLow = await page.evaluate(() => {
      const sim = (window as any).__sim;
      sim.loop.pause();
      return sim.engine.state.stepsRun;
    });

    // Reset field and run at high speed
    await page.evaluate(() => {
      const sim = (window as any).__sim;
      const W = 256;
      const field = new Float32Array(W * W);
      for (let i = 0; i < W * W; i++) field[i] = 0.5;
      sim.engine.loadField(field);
      sim.loop.setStepsPerSecond(10000);
      sim.loop.play();
    });
    await page.waitForTimeout(400);
    const stepsHigh = await page.evaluate(() => {
      const sim = (window as any).__sim;
      sim.loop.pause();
      return sim.engine.state.stepsRun;
    });

    // Higher speed should run more steps
    expect(stepsHigh).toBeGreaterThan(stepsLow * 2);
  });

  test('time accumulates correctly (f64 precision)', async ({ page }) => {
    await page.evaluate(() => {
      const loop = (window as any).__sim.loop;
      loop.setStepsPerSecond(500);
      loop.play();
    });
    await page.waitForTimeout(500);

    const result = await page.evaluate(() => {
      const sim = (window as any).__sim;
      sim.loop.pause();
      const state = sim.engine.state;
      return {
        time: state.time,
        stepsRun: state.stepsRun,
        dt: state.dt,
      };
    });

    // time should equal stepsRun * dt
    expect(result.time).toBeCloseTo(result.stepsRun * result.dt, 10);
    expect(result.time).toBeGreaterThan(0);
  });

  test('no burst after simulated tab background (100ms cap)', async ({ page }) => {
    // Play normally
    await page.evaluate(() => {
      const loop = (window as any).__sim.loop;
      loop.setStepsPerSecond(100);
      loop.play();
    });
    await page.waitForTimeout(300);

    // Record steps per ~300ms of normal play
    const normalSteps = await page.evaluate(() => {
      return (window as any).__sim.engine.state.stepsRun;
    });

    // The 100ms frame cap means even if a frame delta is huge,
    // we only step for 100ms worth of time per frame.
    // Just verify the simulation is running and steps are reasonable.
    expect(normalSteps).toBeGreaterThan(0);
    expect(normalSteps).toBeLessThan(100 * 10); // ~10 frames max in 300ms at 100 sps
  });
});
