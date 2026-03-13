import { Page } from '@playwright/test';

/** Wait until the engine is initialized and attached to window.__sim */
export async function waitForEngine(page: Page): Promise<void> {
  await page.waitForFunction(() => (window as any).__sim?.engine !== undefined, { timeout: 10_000 });
}

/** Read the current engine state (CPU-side values only — no GPU readback) */
export async function getEngineState(page: Page) {
  return page.evaluate(() => {
    const eng = (window as any).__sim.engine;
    return { stepsRun: eng.stepsRun };
  });
}

/** Read the full concentration field from GPU to CPU via engine.readField() */
export async function readField(page: Page): Promise<number[]> {
  return page.evaluate(async () => {
    const field: Float32Array = await (window as any).__sim.engine.readField();
    return Array.from(field);
  });
}

/** Dispatch n physics steps then wait briefly for GPU to flush */
export async function stepEngine(page: Page, n: number): Promise<void> {
  await page.evaluate((n: number) => (window as any).__sim.engine.step(n), n);
  await page.waitForTimeout(100);
}

/** Sum all values in the field (total mass) */
export function fieldMass(field: number[]): number {
  return field.reduce((a, b) => a + b, 0);
}
