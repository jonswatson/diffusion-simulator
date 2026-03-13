import { test, expect } from '@playwright/test';
import { waitForEngine, getEngineState } from './helpers';

test.describe('Stage 7: UI Controls', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForEngine(page);
    // Wait for UI to render
    await page.waitForTimeout(200);
  });

  test('play/pause buttons toggle simulation', async ({ page }) => {
    const btnPlay = page.locator('#btn-play');
    const btnPause = page.locator('#btn-pause');

    // Initially play enabled, pause disabled
    await expect(btnPlay).toBeEnabled();
    await expect(btnPause).toBeDisabled();

    // Click play
    await btnPlay.click();
    await expect(btnPlay).toBeDisabled();
    await expect(btnPause).toBeEnabled();

    await page.waitForTimeout(300);
    const steps = await page.evaluate(() => (window as any).__sim.engine.state.stepsRun);
    expect(steps).toBeGreaterThan(0);

    // Click pause
    await btnPause.click();
    await expect(btnPlay).toBeEnabled();
    await expect(btnPause).toBeDisabled();
  });

  test('reset button stops and resets simulation', async ({ page }) => {
    // Play briefly
    await page.locator('#btn-play').click();
    await page.waitForTimeout(300);

    // Reset
    await page.locator('#btn-reset').click();
    const state = await getEngineState(page);
    expect(state.stepsRun).toBe(0);
    expect(state.time).toBe(0);
  });

  test('temperature slider updates D(T)', async ({ page }) => {
    const state1 = await getEngineState(page);
    const D1 = state1.diffusivity;

    // Change temperature to a higher value
    await page.locator('#sl-temperature').fill('1200');
    await page.locator('#sl-temperature').dispatchEvent('input');
    await page.waitForTimeout(100);

    const state2 = await getEngineState(page);
    // Higher temperature = higher diffusivity (Arrhenius)
    expect(state2.diffusivity).toBeGreaterThan(D1);

    // Temperature display should update
    const tempText = await page.locator('#val-temperature').textContent();
    expect(tempText).toBe('1200');
  });

  test('material select changes physics', async ({ page }) => {
    const state1 = await getEngineState(page);

    await page.locator('#sel-material').selectOption('Ni-in-Cu');
    await page.waitForTimeout(100);

    const state2 = await getEngineState(page);
    // Different material = different D
    expect(state2.diffusivity).not.toBeCloseTo(state1.diffusivity, 20);
  });

  test('domain size select changes dx', async ({ page }) => {
    const state1 = await getEngineState(page);

    await page.locator('#sel-domain').selectOption('200e-6');
    await page.waitForTimeout(100);

    const state2 = await getEngineState(page);
    // Larger domain = larger dx
    expect(state2.dx).toBeGreaterThan(state1.dx);
  });

  test('info panel shows simulation values', async ({ page }) => {
    // Wait for the first frame to populate values
    await page.waitForTimeout(500);

    const dText = await page.locator('#info-D').textContent();
    expect(dText).toContain('m²/s');

    const dtText = await page.locator('#info-dt').textContent();
    expect(dtText).toContain('s');

    const rText = await page.locator('#info-r').textContent();
    expect(parseFloat(rText!)).toBeGreaterThan(0);
  });

  test('speed slider changes stepsPerSecond', async ({ page }) => {
    // Set to maximum speed (10^5)
    await page.locator('#sl-speed').fill('5');
    await page.locator('#sl-speed').dispatchEvent('input');

    const speedText = await page.locator('#val-speed').textContent();
    expect(speedText).toBe('100,000');
  });

  test('temperature out-of-range shows warning', async ({ page }) => {
    // Cu-in-Al valid range is 400–900 K. Set to 300 K.
    await page.locator('#sl-temperature').fill('300');
    await page.locator('#sl-temperature').dispatchEvent('input');
    await page.waitForTimeout(100);

    const warning = page.locator('#warnings .warning');
    await expect(warning.first()).toBeVisible();
    const text = await warning.first().textContent();
    expect(text).toContain('outside the validated range');
  });
});
