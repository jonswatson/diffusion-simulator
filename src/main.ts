import './style.css';
import { Solver, MATERIALS } from './engine';

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

    // Default initial condition: left half = 1.0 (B), right half = 0.0 (A)
    // Cosine-smoothed 5%-wide interface to avoid numerical ringing
    const field = new Float32Array(W * H) as Float32Array<ArrayBuffer>;
    const cx = W / 2;
    const blendW = Math.max(3, Math.floor(W * 0.05));
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const d = x - cx;
        if (d < -blendW) {
          field[y * W + x] = 1.0;
        } else if (d > blendW) {
          field[y * W + x] = 0.0;
        } else {
          field[y * W + x] = 0.5 * (1 - Math.sin((Math.PI * d) / (2 * blendW)));
        }
      }
    }
    solver.loadField(field);
    solver.render();

    console.log('GPU ready:', device.label);
    console.log(`Grid: ${W}×${H}, canvas format: ${canvasFormat}`);

    // Expose for Playwright E2E tests (stripped from production builds)
    if (import.meta.env.DEV) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__sim = { device, engine: solver };
    }
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
  }
}

main();
