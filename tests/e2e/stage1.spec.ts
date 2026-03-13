import { test, expect } from '@playwright/test';
import { waitForEngine, getEngineState } from './helpers';

test.describe('Stage 1: GPU Pipeline Bootstrap', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForEngine(page);
  });

  test('WebGPU is available — no error banner', async ({ page }) => {
    const errorBanner = page.locator('#error-banner');
    await expect(errorBanner).toBeHidden();
  });

  test('canvas is visible and has non-zero dimensions', async ({ page }) => {
    const canvas = page.locator('#sim-canvas');
    await expect(canvas).toBeVisible();
    const box = await canvas.boundingBox();
    expect(box!.width).toBeGreaterThan(0);
    expect(box!.height).toBeGreaterThan(0);
  });

  test('engine state is accessible via window.__sim', async ({ page }) => {
    const state = await getEngineState(page);
    expect(state).toBeDefined();
    expect(typeof state.stepsRun).toBe('number');
  });

  test('canvas is rendering — not blank black', async ({ page }) => {
    // Wait for several RAF frames so the GPU has time to render
    await page.waitForTimeout(500);
    const canvas = page.locator('#sim-canvas');
    const screenshot = await canvas.screenshot();
    // A non-blank buffer will be > 200 bytes (PNG compressed)
    expect(screenshot.length).toBeGreaterThan(200);
  });
});
