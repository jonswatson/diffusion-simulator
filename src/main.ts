import './style.css';
import { Solver, MATERIALS } from './engine';
import { generateDefaultField } from './app/imageLoader';
import { createLoop } from './app/loop';
import { initUI } from './app/ui';
import type { UIContext, FrameCallback } from './app/ui';

/** Request a WebGPU device, preferring the discrete GPU on dual-GPU laptops. */
async function initGPU(): Promise<GPUDevice> {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported in this browser.');
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) {
    throw new Error('No suitable GPU adapter found.');
  }

  const device = await adapter.requestDevice({ label: 'diffusion-sim' });

  device.lost.then((info) => {
    showError(`GPU device lost (${info.reason}): ${info.message}. Please reload.`);
  });

  return device;
}

function showError(message: string): void {
  const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
  const banner = document.getElementById('error-banner') as HTMLDivElement;
  canvas.hidden = true;
  banner.hidden = false;
  banner.querySelector('p')!.textContent = message;
}

async function main(): Promise<void> {
  try {
    const device = await initGPU();

    const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
    const context = canvas.getContext('webgpu')!;
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: canvasFormat });

    const W = canvas.width;
    const H = canvas.height;
    const solver = new Solver({ device, context, canvasFormat, width: W, height: H });

    // Configure physics: Cu-in-Al at 873K (600°C), 100µm domain
    const material = MATERIALS['Cu-in-Al'];
    solver.updateConfig({
      material,
      temperature_K: 873,
      domainSize_m: 100e-6,
      gridWidth: W,
    });
    solver.updateMaterialColors([0.75, 0.75, 0.82], material.color);

    // Default initial condition: cosine-smoothed step function
    const field = generateDefaultField(W) as Float32Array<ArrayBuffer>;
    solver.loadField(field);

    // Frame callback — set by initUI
    let onFrame: FrameCallback = () => {};

    // Create the simulation loop
    const loop = createLoop(solver, (info) => onFrame(info));

    // UIContext is a mutable ref — recreateEngine updates ctx.solver so
    // all UI closures see the new engine without being re-wired.
    const ctx: UIContext = {
      solver,
      loop,
      canvas,
      recreateEngine: async (gridWidth: number) => {
        canvas.width = gridWidth;
        canvas.height = gridWidth;
        context.configure({ device, format: canvasFormat });
        const newSolver = new Solver({ device, context, canvasFormat, width: gridWidth, height: gridWidth });
        ctx.solver = newSolver;
        loop.setEngine(newSolver);

        if (import.meta.env.DEV) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (window as any).__sim = { device, engine: newSolver, loop };
        }
      },
    };

    // Wire up UI controls
    onFrame = initUI(ctx);

    console.log('GPU ready:', device.label);
    console.log(`Grid: ${W}×${H}, canvas format: ${canvasFormat}`);

    // Expose for Playwright E2E tests (stripped from production builds)
    if (import.meta.env.DEV) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__sim = { device, engine: solver, loop };
    }
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
  }
}

main();
