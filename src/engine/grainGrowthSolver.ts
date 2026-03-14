// ============================================================
// Grain Growth solver — Allen-Cahn multi-order-parameter model.
//
// Each grain i has an order parameter η_i ∈ [0,1].
// Buffer layout: packed flat, η_i(x,y) at index i*W*H + y*W + x.
// Ping-pong between etaA and etaB buffers.
//
// Uses a custom renderGrain.wgsl for color display with a
// per-grain color table stored in a GPU storage buffer.
// ============================================================

import { createUniformBuffer, dispatchSize, BYTES_PER_CELL } from './buffers';
import { computeGGDt, computeDx } from './physics';
import type { DiffusionEngine, SimMode, EngineState } from './types';
import type { GrainGrowthConfig } from './materials';
import grainWGSL from './shaders/grainGrowth.wgsl?raw';
import renderGrainWGSL from './shaders/renderGrain.wgsl?raw';

export interface GGSolverConfig {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
  width: number;
  height: number;
  numGrains: number;
}

/**
 * Uniform struct layout (48 bytes):
 *   width(u32) height(u32) dt(f32) dx(f32)
 *   L(f32) kappa(f32) A(f32) B(f32)
 *   N(u32) _pad×3
 */
const UNIFORM_SIZE = 48;

export class GrainGrowthSolver implements DiffusionEngine {
  readonly mode: SimMode = 'grain-growth';
  readonly device: GPUDevice;
  private context: GPUCanvasContext;
  private width: number;
  private height: number;
  private numGrains: number;

  // Ping-pong packed eta buffers (N × W × H)
  private etaA: GPUBuffer;
  private etaB: GPUBuffer;
  private uniformBuffer: GPUBuffer;
  private colorTableBuffer: GPUBuffer;

  // Compute pipeline
  private computePipeline: GPUComputePipeline;
  private bindGroupAtoB: GPUBindGroup;
  private bindGroupBtoA: GPUBindGroup;
  private parity = 0;

  // Render pipeline (grain-colored display)
  private renderPipeline: GPURenderPipeline;
  private renderBindGroup: GPUBindGroup;

  private _state: EngineState = {
    time: 0,
    diffusivity: 0,
    dx: 1,
    dt: 0.001,
    r: 0,
    stepsRun: 0,
  };

  get state(): EngineState { return this._state; }
  get stepsRun(): number { return this._state.stepsRun; }

  get currentBuffer(): GPUBuffer {
    return this.parity === 0 ? this.etaA : this.etaB;
  }

  constructor(config: GGSolverConfig) {
    const { device, context, canvasFormat, width, height, numGrains } = config;
    this.device = device;
    this.context = context;
    this.width = width;
    this.height = height;
    this.numGrains = numGrains;

    const packedSize = numGrains * width * height * BYTES_PER_CELL;

    // --- Buffers ---
    this.etaA = device.createBuffer({
      label: 'eta-A',
      size: packedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.etaB = device.createBuffer({
      label: 'eta-B',
      size: packedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.uniformBuffer = createUniformBuffer(device, UNIFORM_SIZE);

    // Color table: N × vec4f (16 bytes each, padded for alignment)
    this.colorTableBuffer = device.createBuffer({
      label: 'grain-colors',
      size: numGrains * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Default colors: evenly spaced HSL hues
    this.setDefaultColors();

    // --- Compute pipeline ---
    const computeBGL = device.createBindGroupLayout({
      label: 'gg-compute-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    this.computePipeline = device.createComputePipeline({
      label: 'gg-compute-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
      compute: { module: device.createShaderModule({ code: grainWGSL, label: 'grain-growth' }), entryPoint: 'main' },
    });

    this.bindGroupAtoB = device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.etaA } },
        { binding: 2, resource: { buffer: this.etaB } },
      ],
    });
    this.bindGroupBtoA = device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.etaB } },
        { binding: 2, resource: { buffer: this.etaA } },
      ],
    });

    // --- Render pipeline (grain-colored) ---
    const renderShader = device.createShaderModule({ code: renderGrainWGSL, label: 'gg-render' });
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
      vertex: { module: renderShader, entryPoint: 'vs_main' },
      fragment: {
        module: renderShader,
        entryPoint: 'fs_main',
        targets: [{ format: canvasFormat }],
      },
    });

    this.renderBindGroup = this.makeRenderBindGroup();
  }

  /** Apply physics parameters from a grain growth config. */
  updateGGConfig(config: GrainGrowthConfig): void {
    const dx = computeDx(config.domainSize_m, config.gridWidth);
    const dt = computeGGDt(config.kinetic_L, config.kappa, config.barrierA, dx);

    this._state.dx = dx;
    this._state.dt = dt;
    this._state.diffusivity = config.kinetic_L; // informational
    this._state.r = 0;

    this.writeUniforms(dx, dt, config.kinetic_L, config.kappa, config.barrierA, config.crossB, config.numGrains);
  }

  private writeUniforms(
    dx: number, dt: number, L: number, kappa: number,
    A: number, B: number, N: number,
  ): void {
    const buf = new ArrayBuffer(UNIFORM_SIZE);
    const u32View = new Uint32Array(buf);
    const f32View = new Float32Array(buf);
    u32View[0] = this.width;
    u32View[1] = this.height;
    f32View[2] = dt;
    f32View[3] = dx;
    f32View[4] = L;
    f32View[5] = kappa;
    f32View[6] = A;
    f32View[7] = B;
    u32View[8] = N;
    u32View[9] = 0;  // pad
    u32View[10] = 0; // pad
    u32View[11] = 0; // pad
    this.device.queue.writeBuffer(this.uniformBuffer, 0, buf);
  }

  /** Load a packed field: Float32Array of size N × W × H. */
  loadField(data: Float32Array<ArrayBuffer>): void {
    this.device.queue.writeBuffer(this.etaA, 0, data);
    this.device.queue.writeBuffer(this.etaB, 0, new Float32Array(data.length) as Float32Array<ArrayBuffer>);
    this.parity = 0;
    this._state.stepsRun = 0;
    this._state.time = 0;
    this.renderBindGroup = this.makeRenderBindGroup();
  }

  step(n: number): void {
    const encoder = this.device.createCommandEncoder({ label: 'gg-step' });
    const pass = encoder.beginComputePass({ label: 'gg-compute' });
    pass.setPipeline(this.computePipeline);
    const [wx, wy] = dispatchSize(this.width, this.height);

    for (let i = 0; i < n; i++) {
      const bg = this.parity === 0 ? this.bindGroupAtoB : this.bindGroupBtoA;
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wx, wy);
      this.parity = 1 - this.parity;
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this._state.stepsRun += n;
    this._state.time += n * this._state.dt;
    this.renderBindGroup = this.makeRenderBindGroup();
  }

  render(): void {
    const encoder = this.device.createCommandEncoder({ label: 'gg-render' });
    const textureView = this.context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
        loadOp: 'clear' as const,
        storeOp: 'store' as const,
      }],
    });

    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Update material colors — for grain growth, this updates the color table. */
  updateMaterialColors(colorA: [number, number, number], colorB: [number, number, number]): void {
    // For grain growth, individual grain colors are set via setGrainColors
    // This is a no-op to satisfy the interface — use setGrainColors instead
    void colorA;
    void colorB;
  }

  /** Set the full grain color table (N × vec4f). */
  setGrainColors(colors: Float32Array<ArrayBuffer>): void {
    this.device.queue.writeBuffer(this.colorTableBuffer, 0, colors);
  }

  async readField(): Promise<Float32Array> {
    const size = this.numGrains * this.width * this.height * BYTES_PER_CELL;
    const stagingBuffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.currentBuffer, 0, stagingBuffer, 0, size);
    this.device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const mapped = stagingBuffer.getMappedRange();
    const copy = new ArrayBuffer(mapped.byteLength);
    new Uint8Array(copy).set(new Uint8Array(mapped));
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return new Float32Array(copy);
  }

  destroy(): void {
    this.etaA.destroy();
    this.etaB.destroy();
    this.uniformBuffer.destroy();
    this.colorTableBuffer.destroy();
  }

  private makeRenderBindGroup(): GPUBindGroup {
    const renderBGL = this.renderPipeline.getBindGroupLayout(0);
    return this.device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.currentBuffer } },
        { binding: 2, resource: { buffer: this.colorTableBuffer } },
      ],
    });
  }

  /** Generate evenly-spaced HSL hues as default grain colors. */
  private setDefaultColors(): void {
    const colors = new Float32Array(this.numGrains * 4);
    for (let i = 0; i < this.numGrains; i++) {
      const hue = (i / this.numGrains) * 360;
      const [r, g, b] = hslToRgb(hue, 0.7, 0.6);
      colors[i * 4 + 0] = r;
      colors[i * 4 + 1] = g;
      colors[i * 4 + 2] = b;
      colors[i * 4 + 3] = 1.0; // pad
    }
    this.device.queue.writeBuffer(this.colorTableBuffer, 0, colors);
  }
}

/** Convert HSL to linear RGB. H in degrees, S and L in [0,1]. */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;

  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }

  return [r + m, g + m, b + m];
}
