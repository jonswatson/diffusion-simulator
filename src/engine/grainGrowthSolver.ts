// ============================================================
// Grain growth solver — Allen-Cahn multi-order-parameter phase field.
//
// Two-pass explicit Euler on GPU:
//   Pass 1 (sum_sq):  sumSq[y,x] = Σᵢ ηᵢ²   (auxiliary scalar)
//   Pass 2 (update):  ηᵢ_new = ηᵢ + dt·L·(κ∇²ηᵢ − dF_bulk − dF_interact)
//
// Ping-pong: etaA ↔ etaB with bind group swap each step.
// Periodic boundary conditions (index wrapping in WGSL).
//
// Buffer layout: eta[g * W² + y * W + x]  (grain-major order, f32).
// ============================================================

import { createUniformBuffer, dispatchSize, BYTES_PER_CELL } from './buffers';
import { computeGGDt } from './physics';
import type { DiffusionEngine, SimMode, EngineState } from './types';
import type { GGConfig } from './materials';
import sumSqWGSL from './shaders/grainGrowthSumSq.wgsl?raw';
import updateWGSL from './shaders/grainGrowthUpdate.wgsl?raw';
import renderWGSL from './shaders/renderGrain.wgsl?raw';

export interface GGSolverConfig {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
  width: number;
  height: number;
  numGrains: number;
}

/** Uniform struct layout: numGrains(u32) gridWidth(u32) dt kappa Wbar A L _pad = 32 bytes */
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
  private etaA: GPUBuffer;
  private etaB: GPUBuffer;

  // Auxiliary: sum of squares Σᵢ ηᵢ² per pixel — [W × H] f32
  private sumSqBuf: GPUBuffer;

  private uniformBuffer: GPUBuffer;
  private renderUniformBuffer: GPUBuffer;
  private colorTableBuffer: GPUBuffer;

  // Compute pipelines
  private sumSqPipeline: GPUComputePipeline;
  private updatePipeline: GPUComputePipeline;

  // Bind groups: two directions × two passes
  private sumSqBG_A: GPUBindGroup;   // Pass 1 reads etaA
  private sumSqBG_B: GPUBindGroup;   // Pass 1 reads etaB
  private updateBG_AtoB: GPUBindGroup;  // Pass 2 reads etaA + sumSq → writes etaB
  private updateBG_BtoA: GPUBindGroup;  // Pass 2 reads etaB + sumSq → writes etaA

  // Render pipeline
  private renderPipeline: GPURenderPipeline;
  private renderBG_A: GPUBindGroup;  // renders etaA
  private renderBG_B: GPUBindGroup;  // renders etaB

  private parity = 0;  // 0 = etaA is current, 1 = etaB is current

  private _state: EngineState = {
    time: 0,
    diffusivity: 0,  // not meaningful for GG
    dx: 1,           // nondimensional
    dt: 0.01,
    r: 0,            // not meaningful for GG
    stepsRun: 0,
  };

  get state(): EngineState { return this._state; }
  get stepsRun(): number { return this._state.stepsRun; }

  get currentBuffer(): GPUBuffer {
    return this.parity === 0 ? this.etaA : this.etaB;
  }

  constructor(cfg: GGSolverConfig) {
    const { device, context, canvasFormat, width, height, numGrains } = cfg;
    this.device = device;
    this.context = context;
    this.width = width;
    this.height = height;
    this.numGrains = numGrains;

    const etaBytes = numGrains * width * height * BYTES_PER_CELL;
    const pixelBytes = width * height * BYTES_PER_CELL;

    // --- Buffers ---
    this.etaA = device.createBuffer({
      label: 'eta-A',
      size: etaBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.etaB = device.createBuffer({
      label: 'eta-B',
      size: etaBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.sumSqBuf = device.createBuffer({
      label: 'sumSq',
      size: pixelBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.uniformBuffer = createUniformBuffer(device, UNIFORM_SIZE);
    this.renderUniformBuffer = createUniformBuffer(device, RENDER_UNIFORM_SIZE);
    this.colorTableBuffer = device.createBuffer({
      label: 'color-table',
      size: numGrains * 3 * BYTES_PER_CELL,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Initialize color table
    device.queue.writeBuffer(this.colorTableBuffer, 0, makeColorTable(numGrains) as Float32Array<ArrayBuffer>);

    // Initialize render uniforms (numGrains, gridWidth, 0, 0)
    const ru = new ArrayBuffer(RENDER_UNIFORM_SIZE);
    const ru32 = new Uint32Array(ru);
    ru32[0] = numGrains;
    ru32[1] = width;
    device.queue.writeBuffer(this.renderUniformBuffer, 0, ru);

    // --- Sum-sq pipeline ---
    const sumSqModule = device.createShaderModule({ code: sumSqWGSL, label: 'gg-sumSq' });
    const sumSqBGL = device.createBindGroupLayout({
      label: 'gg-sumSq-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.sumSqPipeline = device.createComputePipeline({
      label: 'gg-sumSq-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [sumSqBGL] }),
      compute: { module: sumSqModule, entryPoint: 'main' },
    });

    // --- Update pipeline ---
    const updateModule = device.createShaderModule({ code: updateWGSL, label: 'gg-update' });
    const updateBGL = device.createBindGroupLayout({
      label: 'gg-update-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.updatePipeline = device.createComputePipeline({
      label: 'gg-update-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [updateBGL] }),
      compute: { module: updateModule, entryPoint: 'main' },
    });

    // --- Bind groups ---
    this.sumSqBG_A = device.createBindGroup({
      layout: sumSqBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.etaA } },
        { binding: 2, resource: { buffer: this.sumSqBuf } },
      ],
    });
    this.sumSqBG_B = device.createBindGroup({
      layout: sumSqBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.etaB } },
        { binding: 2, resource: { buffer: this.sumSqBuf } },
      ],
    });
    this.updateBG_AtoB = device.createBindGroup({
      layout: updateBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.etaA } },
        { binding: 2, resource: { buffer: this.sumSqBuf } },
        { binding: 3, resource: { buffer: this.etaB } },
      ],
    });
    this.updateBG_BtoA = device.createBindGroup({
      layout: updateBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.etaB } },
        { binding: 2, resource: { buffer: this.sumSqBuf } },
        { binding: 3, resource: { buffer: this.etaA } },
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

    this.renderBG_A = this.makeRenderBindGroup(this.etaA);
    this.renderBG_B = this.makeRenderBindGroup(this.etaB);
  }

  /** Apply physics parameters. Recomputes dt and updates the GPU uniform buffer. */
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
    this.device.queue.writeBuffer(this.etaA, 0, data);
    // Zero out etaB (not strictly required but keeps state clean)
    this.device.queue.writeBuffer(this.etaB, 0, new Float32Array(data.length) as Float32Array<ArrayBuffer>);
    this.parity = 0;
    this._state.stepsRun = 0;
    this._state.time = 0;
  }

  step(n: number): void {
    const encoder = this.device.createCommandEncoder({ label: 'gg-step' });
    const [wx, wy] = dispatchSize(this.width, this.height);
    const wz = this.numGrains; // one workgroup per grain (z dimension)

    for (let i = 0; i < n; i++) {
      const sumSqBG = this.parity === 0 ? this.sumSqBG_A : this.sumSqBG_B;
      const updateBG = this.parity === 0 ? this.updateBG_AtoB : this.updateBG_BtoA;

      // Pass 1: compute sumSq from current eta (separate pass = implicit barrier)
      const pass1 = encoder.beginComputePass({ label: 'gg-sumSq' });
      pass1.setPipeline(this.sumSqPipeline);
      pass1.setBindGroup(0, sumSqBG);
      pass1.dispatchWorkgroups(wx, wy);
      pass1.end();

      // Pass 2: update all grains (3D dispatch; z = grain index)
      const pass2 = encoder.beginComputePass({ label: 'gg-update' });
      pass2.setPipeline(this.updatePipeline);
      pass2.setBindGroup(0, updateBG);
      pass2.dispatchWorkgroups(wx, wy, wz);
      pass2.end();

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

  /** Not meaningful for GG (no scalar concentration). No-op. */
  updateMaterialColors(_colorA: [number, number, number], _colorB: [number, number, number]): void {
    // GG uses per-grain color table; no-op for interface compatibility.
  }

  /** Read the full eta field [numGrains × W × H] f32 from GPU. Slow — diagnostics only. */
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
    this.etaA.destroy();
    this.etaB.destroy();
    this.sumSqBuf.destroy();
    this.uniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
    this.colorTableBuffer.destroy();
  }

  private makeRenderBindGroup(etaBuf: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBuffer } },
        { binding: 1, resource: { buffer: etaBuf } },
        { binding: 2, resource: { buffer: this.colorTableBuffer } },
      ],
    });
  }
}

