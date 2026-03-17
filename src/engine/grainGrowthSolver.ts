// ============================================================
// Constrained Allen-Cahn grain growth solver.
//
// Single-pass explicit Euler per pixel:
//   1. Compute μᵢ = δF/δφᵢ for every grain at every pixel.
//   2. Compute λ = mean(μ) per pixel (Lagrange multiplier).
//   3. φ̃ᵢ = φᵢ − dt·L·(μᵢ − λ)   (constrained update, Σφ conserved)
//   4. Project φ̃ onto probability simplex (Duchi et al.).
//
// Constraint guarantees:
//   - Σᵢ φᵢ(x,y) = 1 exactly after projection
//   - φᵢ ≥ 0 everywhere after projection
//   → No void phase, no dark band at grain boundaries.
//
// Ping-pong: phiA ↔ phiB with bind group swap each step.
// Periodic boundary conditions (index wrapping in WGSL).
// Buffer layout: phi[g * W² + y * W + x]  (grain-major order, f32).
// ============================================================

import { createUniformBuffer, dispatchSize, BYTES_PER_CELL } from './buffers';
import { computeGGDt } from './physics';
import type { DiffusionEngine, SimMode, EngineState } from './types';
import type { GGConfig } from './materials';
import constrainedWGSL from './shaders/grainGrowthConstrained.wgsl?raw';
import renderWGSL from './shaders/renderGrain.wgsl?raw';

export interface GGSolverConfig {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
  width: number;
  height: number;
  numGrains: number;
}

/** Compute uniform struct layout: numGrains(u32) gridWidth(u32) dt kappa W A L _pad = 32 bytes */
const UNIFORM_SIZE = 32;

/** Render uniform struct: numGrains(u32) gridWidth(u32) _pad0 _pad1 = 16 bytes */
const RENDER_UNIFORM_SIZE = 16;

/** Evenly-spaced HSL hues, converted to linear RGB [0,1]. */
function makeColorTable(numGrains: number): Float32Array {
  const table = new Float32Array(numGrains * 3);
  for (let i = 0; i < numGrains; i++) {
    const h = (i / numGrains) * 360;
    const s = 0.75;
    const l = 0.55;
    // HSL → RGB (CSS algorithm)
    const a = s * Math.min(l, 1 - l);
    const f = (n: number): number => {
      const k = (n + h / 30) % 12;
      return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    };
    table[i * 3]     = f(0);
    table[i * 3 + 1] = f(8);
    table[i * 3 + 2] = f(4);
  }
  return table;
}

export class GrainGrowthSolver implements DiffusionEngine {
  readonly mode: SimMode = 'grain-growth';
  readonly device: GPUDevice;

  private context: GPUCanvasContext;
  private width: number;
  private height: number;
  private numGrains: number;

  // Ping-pong order-parameter buffers: [numGrains × W × H] f32
  private phiA: GPUBuffer;
  private phiB: GPUBuffer;

  private uniformBuffer: GPUBuffer;
  private renderUniformBuffer: GPUBuffer;
  private colorTableBuffer: GPUBuffer;

  // Single compute pipeline (constrained Allen-Cahn, single pass)
  private computePipeline: GPUComputePipeline;

  // Two bind groups — one per ping-pong direction
  private bindGroupAtoB: GPUBindGroup;  // reads phiA → writes phiB
  private bindGroupBtoA: GPUBindGroup;  // reads phiB → writes phiA

  // Render pipeline
  private renderPipeline: GPURenderPipeline;
  private renderBG_A: GPUBindGroup;  // renders phiA
  private renderBG_B: GPUBindGroup;  // renders phiB

  private parity = 0;  // 0 = phiA is current, 1 = phiB is current

  private _state: EngineState = {
    time: 0,
    diffusivity: 0,  // not meaningful for GG
    dx: 1,           // nondimensional
    dt: 0.005,
    r: 0,            // not meaningful for GG
    stepsRun: 0,
  };

  get state(): EngineState { return this._state; }
  get stepsRun(): number { return this._state.stepsRun; }

  get currentBuffer(): GPUBuffer {
    return this.parity === 0 ? this.phiA : this.phiB;
  }

  constructor(cfg: GGSolverConfig) {
    const { device, context, canvasFormat, width, height, numGrains } = cfg;
    this.device = device;
    this.context = context;
    this.width = width;
    this.height = height;
    this.numGrains = numGrains;

    const phiBytes = numGrains * width * height * BYTES_PER_CELL;

    // --- Buffers ---
    this.phiA = device.createBuffer({
      label: 'phi-A',
      size: phiBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.phiB = device.createBuffer({
      label: 'phi-B',
      size: phiBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.uniformBuffer = createUniformBuffer(device, UNIFORM_SIZE);
    this.renderUniformBuffer = createUniformBuffer(device, RENDER_UNIFORM_SIZE);
    this.colorTableBuffer = device.createBuffer({
      label: 'color-table',
      size: numGrains * 3 * BYTES_PER_CELL,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Initialize color table and render uniforms
    device.queue.writeBuffer(this.colorTableBuffer, 0, makeColorTable(numGrains) as Float32Array<ArrayBuffer>);
    const ru = new ArrayBuffer(RENDER_UNIFORM_SIZE);
    const ru32 = new Uint32Array(ru);
    ru32[0] = numGrains;
    ru32[1] = width;
    device.queue.writeBuffer(this.renderUniformBuffer, 0, ru);

    // --- Compute pipeline ---
    const computeModule = device.createShaderModule({ code: constrainedWGSL, label: 'gg-constrained' });
    const computeBGL = device.createBindGroupLayout({
      label: 'gg-compute-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.computePipeline = device.createComputePipeline({
      label: 'gg-constrained-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
      compute: { module: computeModule, entryPoint: 'main' },
    });

    this.bindGroupAtoB = device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiA } },
        { binding: 2, resource: { buffer: this.phiB } },
      ],
    });
    this.bindGroupBtoA = device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiB } },
        { binding: 2, resource: { buffer: this.phiA } },
      ],
    });

    // --- Render pipeline ---
    const renderModule = device.createShaderModule({ code: renderWGSL, label: 'gg-render' });
    const renderBGL = device.createBindGroupLayout({
      label: 'gg-render-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.renderPipeline = device.createRenderPipeline({
      label: 'gg-render-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
      vertex: { module: renderModule, entryPoint: 'vs_main' },
      fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format: canvasFormat }] },
    });
    this.renderBG_A = this.makeRenderBindGroup(this.phiA);
    this.renderBG_B = this.makeRenderBindGroup(this.phiB);
  }

  /** Apply physics parameters. Recomputes dt and writes the GPU uniform buffer. */
  updateGGConfig(config: GGConfig): void {
    const dt = computeGGDt(config.kappa, config.W, config.A, config.numGrains, config.L);
    this._state.dt = dt;

    const buf = new ArrayBuffer(UNIFORM_SIZE);
    const u32 = new Uint32Array(buf);
    const f32 = new Float32Array(buf);
    u32[0] = config.numGrains;
    u32[1] = config.gridWidth;
    f32[2] = dt;
    f32[3] = config.kappa;
    f32[4] = config.W;
    f32[5] = config.A;
    f32[6] = config.L;
    f32[7] = 0; // _pad
    this.device.queue.writeBuffer(this.uniformBuffer, 0, buf);
  }

  loadField(data: Float32Array<ArrayBuffer>): void {
    this.device.queue.writeBuffer(this.phiA, 0, data);
    this.device.queue.writeBuffer(this.phiB, 0, new Float32Array(data.length) as Float32Array<ArrayBuffer>);
    this.parity = 0;
    this._state.stepsRun = 0;
    this._state.time = 0;
  }

  step(n: number): void {
    const encoder = this.device.createCommandEncoder({ label: 'gg-step' });
    const [wx, wy] = dispatchSize(this.width, this.height);

    // Each step is its own compute pass. Separate beginComputePass/endComputePass
    // calls provide the implicit memory barrier: step i+1 is guaranteed to see
    // the phiOut values written by step i before it reads them as phiIn.
    // (Within a single pass, dispatches are not ordered — never combine steps.)
    for (let i = 0; i < n; i++) {
      const bg = this.parity === 0 ? this.bindGroupAtoB : this.bindGroupBtoA;
      const pass = encoder.beginComputePass({ label: 'gg-constrained' });
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wx, wy);
      pass.end();
      this.parity = 1 - this.parity;
    }

    this.device.queue.submit([encoder.finish()]);
    this._state.stepsRun += n;
    this._state.time += n * this._state.dt;
  }

  render(): void {
    const encoder = this.device.createCommandEncoder({ label: 'gg-render' });
    const bg = this.parity === 0 ? this.renderBG_A : this.renderBG_B;

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
        loadOp: 'clear' as const,
        storeOp: 'store' as const,
      }],
    });
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Not meaningful for GG (no scalar concentration). No-op for interface compatibility. */
  updateMaterialColors(_colorA: [number, number, number], _colorB: [number, number, number]): void {}

  /** Read the full phi field [numGrains × W × H] f32 from GPU. Slow — diagnostics only. */
  async readField(): Promise<Float32Array> {
    const size = this.numGrains * this.width * this.height * BYTES_PER_CELL;
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.currentBuffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const mapped = staging.getMappedRange();
    const copy = new ArrayBuffer(mapped.byteLength);
    new Uint8Array(copy).set(new Uint8Array(mapped));
    staging.unmap();
    staging.destroy();
    return new Float32Array(copy);
  }

  destroy(): void {
    this.phiA.destroy();
    this.phiB.destroy();
    this.uniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
    this.colorTableBuffer.destroy();
  }

  private makeRenderBindGroup(phiBuf: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBuffer } },
        { binding: 1, resource: { buffer: phiBuf } },
        { binding: 2, resource: { buffer: this.colorTableBuffer } },
      ],
    });
  }
}
