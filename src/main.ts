import './style.css';

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
    console.log('GPU ready:', device.label);
    console.log('Max buffer size:', device.limits.maxBufferSize);
    console.log('Max compute workgroups/dim:', device.limits.maxComputeWorkgroupsPerDimension);

    // Expose for Playwright E2E tests (stripped from production builds)
    if (import.meta.env.DEV) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__sim = { device };
    }
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
  }
}

main();
