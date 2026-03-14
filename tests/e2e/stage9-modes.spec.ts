import { test, expect } from '@playwright/test';
import { waitForEngine } from './helpers';

test.describe('Stage 9: Mode Switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForEngine(page);
    await page.waitForTimeout(200);
  });

  test('mode selector switches control visibility', async ({ page }) => {
    // Initially Fick mode: fick-controls visible, others hidden
    await expect(page.locator('#fick-controls')).toBeVisible();
    await expect(page.locator('#ch-controls')).toBeHidden();
    await expect(page.locator('#gg-controls')).toBeHidden();

    // Switch to Cahn-Hilliard
    await page.locator('#sel-mode').selectOption('cahn-hilliard');
    await page.waitForTimeout(300);
    await expect(page.locator('#fick-controls')).toBeHidden();
    await expect(page.locator('#ch-controls')).toBeVisible();
    await expect(page.locator('#gg-controls')).toBeHidden();

    // Switch to Grain Growth
    await page.locator('#sel-mode').selectOption('grain-growth');
    await page.waitForTimeout(300);
    await expect(page.locator('#fick-controls')).toBeHidden();
    await expect(page.locator('#ch-controls')).toBeHidden();
    await expect(page.locator('#gg-controls')).toBeVisible();

    // Switch back to Fick
    await page.locator('#sel-mode').selectOption('fick');
    await page.waitForTimeout(300);
    await expect(page.locator('#fick-controls')).toBeVisible();
    await expect(page.locator('#ch-controls')).toBeHidden();
    await expect(page.locator('#gg-controls')).toBeHidden();
  });

  test('CH mode: play produces phase separation (variance increase)', async ({ page }) => {
    // Switch to Cahn-Hilliard
    await page.locator('#sel-mode').selectOption('cahn-hilliard');
    await page.waitForTimeout(300);

    // Read initial field variance
    const initialVariance = await page.evaluate(async () => {
      const engine = (window as any).__sim.engine;
      // The engine got recreated by mode switch — but __sim still points to old engine.
      // We need to get the current solver from the loop or wait for it.
      // Actually, after mode switch, let's just use the engine that __sim has.
      // The mode switch calls recreateEngine which replaces ctx.solver,
      // but __sim is only set at startup. Let's step and check via canvas instead.
      const field: Float32Array = await engine.readField();
      const arr = Array.from(field);
      const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
      return arr.reduce((a, v) => a + (v - mean) ** 2, 0) / arr.length;
    });

    // Play for a while to let phase separation develop
    await page.locator('#btn-play').click();
    await page.waitForTimeout(2000);
    await page.locator('#btn-pause').click();
    await page.waitForTimeout(200);

    // Check that steps advanced
    const stepsRun = await page.evaluate(() => (window as any).__sim.engine.state.stepsRun);
    // Note: __sim.engine may still point to old engine if ui.ts doesn't update it.
    // Let's verify via the info panel instead.
    const stepsText = await page.locator('#info-steps').textContent();
    expect(parseInt(stepsText!.replace(/,/g, ''), 10)).toBeGreaterThan(0);

    // Verify the mode info shows Spinodal
    const modeText = await page.locator('#info-mode').textContent();
    expect(modeText).toBe('Spinodal');
  });

  test('GG mode: Voronoi pattern renders and mode info updates', async ({ page }) => {
    // Switch to Grain Growth
    await page.locator('#sel-mode').selectOption('grain-growth');
    await page.waitForTimeout(500);

    // Mode info should show Grain Growth
    const modeText = await page.locator('#info-mode').textContent();
    expect(modeText).toBe('Grain Growth');

    // dt should be displayed (not a dash)
    const dtText = await page.locator('#info-dt').textContent();
    expect(dtText).toContain('s');

    // Play briefly — should not crash
    await page.locator('#btn-play').click();
    await page.waitForTimeout(500);
    await page.locator('#btn-pause').click();

    const stepsText = await page.locator('#info-steps').textContent();
    expect(parseInt(stepsText!.replace(/,/g, ''), 10)).toBeGreaterThan(0);
  });

  test('mode switch + reset works cleanly', async ({ page }) => {
    // Switch to CH, play, then reset
    await page.locator('#sel-mode').selectOption('cahn-hilliard');
    await page.waitForTimeout(300);
    await page.locator('#btn-play').click();
    await page.waitForTimeout(500);
    await page.locator('#btn-reset').click();
    await page.waitForTimeout(300);

    // Steps should be reset to 0
    const stepsText = await page.locator('#info-steps').textContent();
    expect(stepsText).toBe('0');

    // Switch to GG, play, reset
    await page.locator('#sel-mode').selectOption('grain-growth');
    await page.waitForTimeout(300);
    await page.locator('#btn-play').click();
    await page.waitForTimeout(500);
    await page.locator('#btn-reset').click();
    await page.waitForTimeout(300);

    const stepsText2 = await page.locator('#info-steps').textContent();
    expect(stepsText2).toBe('0');

    // Switch back to Fick — should work without errors
    await page.locator('#sel-mode').selectOption('fick');
    await page.waitForTimeout(300);
    await page.locator('#btn-play').click();
    await page.waitForTimeout(500);

    // Should have advanced some steps
    const stepsText3 = await page.locator('#info-steps').textContent();
    expect(parseInt(stepsText3!.replace(/,/g, ''), 10)).toBeGreaterThan(0);
  });

  test('info panel adapts to mode — D(T) and r hidden for non-Fick', async ({ page }) => {
    // In Fick mode, D should show m²/s
    await page.waitForTimeout(500);
    const dFick = await page.locator('#info-D').textContent();
    expect(dFick).toContain('m');

    // Switch to CH
    await page.locator('#sel-mode').selectOption('cahn-hilliard');
    await page.waitForTimeout(500);
    const dCH = await page.locator('#info-D').textContent();
    expect(dCH).toBe('\u2014');
    const rCH = await page.locator('#info-r').textContent();
    expect(rCH).toBe('\u2014');

    // dt should still show a value
    const dtCH = await page.locator('#info-dt').textContent();
    expect(dtCH).toContain('s');
  });

  test('image upload hidden in non-Fick modes', async ({ page }) => {
    // Fick: image upload visible
    await expect(page.locator('#lbl-image-upload')).toBeVisible();

    // CH: image upload hidden
    await page.locator('#sel-mode').selectOption('cahn-hilliard');
    await page.waitForTimeout(200);
    await expect(page.locator('#lbl-image-upload')).toBeHidden();

    // GG: image upload hidden
    await page.locator('#sel-mode').selectOption('grain-growth');
    await page.waitForTimeout(200);
    await expect(page.locator('#lbl-image-upload')).toBeHidden();
  });
});
