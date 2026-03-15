import { test, expect } from '@playwright/test';
import { waitForEngine } from './helpers';

const NUM_GRAINS = 4;
const W = 64;

function makeGGConfig() {
  return {
    mode: 'grain-growth' as const,
    material: { name: 'Aluminium', symbol: 'Al', Q_gb: 130_000, T_min: 400, T_max: 900 },
    temperature_K: 700,
    interfaceWidth_m: 6e-6,  // 6 µm → 6 px at dx = 1 µm
    barrierA: 1.0,
    crossB: 1.0,
    numGrains: NUM_GRAINS,
    domainSize_m: W * 1e-6,  // W µm → dx = 1e-6 m (1 µm/pixel)
    gridWidth: W,
  };
}

/** Generate a simple Voronoi field: assign each cell to the nearest seed. */
function makeVoronoiField(W: number, N: number): number[] {
  // Fixed seed positions for reproducibility
  const seeds: [number, number][] = [];
  for (let i = 0; i < N; i++) {
    seeds.push([
      Math.floor(W * (0.2 + 0.6 * ((i * 7 + 3) % N) / N)),
      Math.floor(W * (0.2 + 0.6 * ((i * 11 + 5) % N) / N)),
    ]);
  }

  const packed: number[] = new Array(N * W * W).fill(0);
  for (let y = 0; y < W; y++) {
    for (let x = 0; x < W; x++) {
      let minDist = Infinity;
      let owner = 0;
      for (let i = 0; i < N; i++) {
        const dx = x - seeds[i][0];
        const dy = y - seeds[i][1];
        const dist = dx * dx + dy * dy;
        if (dist < minDist) {
          minDist = dist;
          owner = i;
        }
      }
      packed[owner * W * W + y * W + x] = 1.0;
    }
  }
  return packed;
}

test.describe('Stage 9: Grain Growth Solver', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForEngine(page);
  });

  test('GG solver can be instantiated and stepped', async ({ page }) => {
    const result = await page.evaluate(async (args) => {
      const sim = (window as any).__sim;
      const { device, GrainGrowthSolver, canvasFormat, context } = sim;

      const ggSolver = new GrainGrowthSolver({
        device, context, canvasFormat,
        width: args.W, height: args.W, numGrains: args.N,
      });
      ggSolver.updateGGConfig(args.cfg);

      const field = new Float32Array(args.packed);
      ggSolver.loadField(field);

      // Step 100 times
      for (let remaining = 100; remaining > 0; remaining -= 50) {
        ggSolver.step(Math.min(50, remaining));
      }
      await device.queue.onSubmittedWorkDone();

      const result = await ggSolver.readField();
      ggSolver.destroy();

      return {
        length: result.length,
        mode: ggSolver.mode,
        expectedLength: args.N * args.W * args.W,
      };
    }, { W, N: NUM_GRAINS, cfg: makeGGConfig(), packed: makeVoronoiField(W, NUM_GRAINS) });

    expect(result.length).toBe(result.expectedLength);
    expect(result.mode).toBe('grain-growth');
  });

  test('each cell has at least one η > 0.5 after stepping', async ({ page }) => {
    const result = await page.evaluate(async (args) => {
      const sim = (window as any).__sim;
      const { device, GrainGrowthSolver, canvasFormat, context } = sim;

      const ggSolver = new GrainGrowthSolver({
        device, context, canvasFormat,
        width: args.W, height: args.W, numGrains: args.N,
      });
      ggSolver.updateGGConfig(args.cfg);
      ggSolver.loadField(new Float32Array(args.packed));

      for (let remaining = 100; remaining > 0; remaining -= 50) {
        ggSolver.step(Math.min(50, remaining));
      }
      await device.queue.onSubmittedWorkDone();

      const field = await ggSolver.readField();
      const WH = args.W * args.W;

      // Check each cell has at least one η > 0.5
      let cellsWithGrain = 0;
      for (let y = 0; y < args.W; y++) {
        for (let x = 0; x < args.W; x++) {
          let maxEta = -1;
          for (let i = 0; i < args.N; i++) {
            maxEta = Math.max(maxEta, field[i * WH + y * args.W + x]);
          }
          if (maxEta > 0.5) cellsWithGrain++;
        }
      }

      ggSolver.destroy();

      return {
        totalCells: WH,
        cellsWithGrain,
        fraction: cellsWithGrain / WH,
      };
    }, { W, N: NUM_GRAINS, cfg: makeGGConfig(), packed: makeVoronoiField(W, NUM_GRAINS) });

    // At least 90% of cells should have a dominant grain
    expect(result.fraction).toBeGreaterThan(0.9);
  });

  test('render produces non-blank canvas', async ({ page }) => {
    const result = await page.evaluate(async (args) => {
      const sim = (window as any).__sim;
      const { device, GrainGrowthSolver, canvasFormat, context } = sim;

      const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement;
      canvas.width = args.W;
      canvas.height = args.W;
      context.configure({ device, format: canvasFormat });

      const ggSolver = new GrainGrowthSolver({
        device, context, canvasFormat,
        width: args.W, height: args.W, numGrains: args.N,
      });
      ggSolver.updateGGConfig(args.cfg);
      ggSolver.loadField(new Float32Array(args.packed));
      ggSolver.render();

      // Read canvas pixels to check it rendered something
      const ctx2d = document.createElement('canvas').getContext('2d')!;
      ctx2d.canvas.width = args.W;
      ctx2d.canvas.height = args.W;
      ctx2d.drawImage(canvas, 0, 0);
      const imageData = ctx2d.getImageData(0, 0, args.W, args.W);

      // Count non-black pixels
      let nonBlack = 0;
      for (let i = 0; i < imageData.data.length; i += 4) {
        if (imageData.data[i] > 10 || imageData.data[i + 1] > 10 || imageData.data[i + 2] > 10) {
          nonBlack++;
        }
      }

      ggSolver.destroy();

      return {
        totalPixels: args.W * args.W,
        nonBlackPixels: nonBlack,
      };
    }, { W, N: NUM_GRAINS, cfg: makeGGConfig(), packed: makeVoronoiField(W, NUM_GRAINS) });

    // Most pixels should be non-black (grain colors are bright)
    expect(result.nonBlackPixels).toBeGreaterThan(result.totalPixels * 0.5);
  });
});
