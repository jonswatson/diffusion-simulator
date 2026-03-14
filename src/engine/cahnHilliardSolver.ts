// ============================================================
// Cahn-Hilliard solver — spinodal decomposition on the GPU.
//
// Split-form explicit FTCS:
//   Pass 1: μ = f'(φ) − ε²·∇²φ         (chemical potential)
//   Pass 2: φ_new = φ_old + M·dt·∇²μ   (composition advance)
//
// Uses ping-pong buffers (phiA/phiB) for composition and a single
// mu buffer (rewritten each step). Reuses the existing render.wgsl
// for display — φ maps to color exactly like concentration.
// ============================================================

import { createFieldBuffer, createUniformBuffer, dispatchSize, BYTES_PER_CELL } from './buffers';
import { R_GAS, computeCHDt, computeDx, epsilonSqFromInterfaceWidth } from './physics';
import type { DiffusionEngine, SimMode, EngineState } from './types';
import type { CahnHilliardConfig } from './materials';
import muWGSL from './shaders/cahnHilliardMu.wgsl?raw';
import phiWGSL from './shaders/cahnHilliardPhi.wgsl?raw';
import renderWGSL from './shaders/render.wgsl?raw';

export interface CHSolverConfig {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
  width: number;
  height: number;
}

/** Uniform struct layout: width(u32) height(u32) dt(f32) dx(f32) M(f32) epsilon_sq(f32) A(f32) _pad(u32) = 32 bytes */
const UNIFORM_SIZE = 32;

export class CahnHilliardSolver implements DiffusionEngine {
  readonly mode: SimMode = 'cahn-hilliard';
  readonly device: GPUDevice;
  private context: GPUCanvasContext;
  private width: number;
  private height: number;

  // Ping-pong composition buffers
  private phiA: GPUBuffer;
  private phiB: GPUBuffer;
  private muBuf: GPUBuffer;
  private uniformBuffer: GPUBuffer;
  private materialColorBuffer: GPUBuffer;

  // Compute pipelines
  private muPipeline: GPUComputePipeline;
  private phiPipeline: GPUComputePipeline;

  // Bind groups for ping-pong: step reads phiA → writes phiB, then swap
  private muBindGroupAtoB: GPUBindGroup;   // mu pass: reads phiA
  private phiBindGroupAtoB: GPUBindGroup;  // phi pass: reads phiA + mu → writes phiB
  private muBindGroupBtoA: GPUBindGroup;   // mu pass: reads phiB
  private phiBindGroupBtoA: GPUBindGroup;  // phi pass: reads phiB + mu → writes phiA

  private parity = 0; // 0: current data in phiA, 1: in phiB

  // Render pipeline (reuses render.wgsl from Fick)
  private renderPipeline: GPURenderPipeline;
  private renderBindGroup: GPUBindGroup;

  private _state: EngineState = {
    time: 0,
    diffusivity: 0, // not used for CH, but part of the interface
    dx: 1,
    dt: 0.001,
    r: 0,
    stepsRun: 0,
  };

  get state(): EngineState { return this._state; }
  get stepsRun(): number { return this._state.stepsRun; }

  get currentBuffer(): GPUBuffer {
    return this.parity === 0 ? this.phiA : this.phiB;
  }

  constructor(config: CHSolverConfig) {
    const { device, context, canvasFormat, width, height } = config;
    this.device = device;
    this.context = context;
    this.width = width;
    this.height = height;

    // --- Buffers ---
    this.phiA = createFieldBuffer(device, width, height);
    this.phiB = createFieldBuffer(device, width, height);
    this.muBuf = createFieldBuffer(device, width, height);
    this.uniformBuffer = createUniformBuffer(device, UNIFORM_SIZE);
    this.materialColorBuffer = createUniformBuffer(device, 32);

    // Default colors: dark blue (φ=0) and bright yellow (φ=1)
    device.queue.writeBuffer(this.materialColorBuffer, 0, new Float32Array([
      0.05, 0.05, 0.3, 0.0,  // colorA: dark blue
      1.0, 0.9, 0.2, 0.0,    // colorB: bright yellow
    ]));

    // --- Mu pipeline (Pass 1): reads phi, writes mu ---
    const muBGL = device.createBindGroupLayout({
      label: 'ch-mu-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    this.muPipeline = device.createComputePipeline({
      label: 'ch-mu-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [muBGL] }),
      compute: { module: device.createShaderModule({ code: muWGSL, label: 'ch-mu' }), entryPoint: 'main' },
    });

    // --- Phi pipeline (Pass 2): reads phi + mu, writes phi_out ---
    const phiBGL = device.createBindGroupLayout({
      label: 'ch-phi-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    this.phiPipeline = device.createComputePipeline({
      label: 'ch-phi-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [phiBGL] }),
      compute: { module: device.createShaderModule({ code: phiWGSL, label: 'ch-phi' }), entryPoint: 'main' },
    });

    // --- Bind groups for both directions of the ping-pong ---
    // A → B direction
    this.muBindGroupAtoB = device.createBindGroup({
      layout: muBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiA } },
        { binding: 2, resource: { buffer: this.muBuf } },
      ],
    });
    this.phiBindGroupAtoB = device.createBindGroup({
      layout: phiBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiA } },
        { binding: 2, resource: { buffer: this.muBuf } },
        { binding: 3, resource: { buffer: this.phiB } },
      ],
    });

    // B → A direction
    this.muBindGroupBtoA = device.createBindGroup({
      layout: muBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiB } },
        { binding: 2, resource: { buffer: this.muBuf } },
      ],
    });
    this.phiBindGroupBtoA = device.createBindGroup({
      layout: phiBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiB } },
        { binding: 2, resource: { buffer: this.muBuf } },
        { binding: 3, resource: { buffer: this.phiA } },
      ],
    });

    // --- Render pipeline (reuses render.wgsl) ---
    const renderShader = device.createShaderModule({ code: renderWGSL, label: 'ch-render' });
    const renderBGL = device.createBindGroupLayout({
      label: 'ch-render-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.renderPipeline = device.createRenderPipeline({
      label: 'ch-render-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
      vertex: { module: renderShader, entryPoint: 'vs_main' },
      fragment: {
        module: renderShader,
        entryPoint: 'fs_main',
        targets: [{ format: canvasFormat }],
      },
    });

    // Initial render bind group pointing at phiA
    this.renderBindGroup = this.makeRenderBindGroup();
  }

  /**
   * Apply physics parameters from a CH config.
   * Computes dimensionless Ω/RT, Arrhenius mobility, and gradient energy
   * from the material database + temperature + interface width.
   */
  updateCHConfig(config: CahnHilliardConfig): void {
    const dx = computeDx(config.domainSize_m, config.gridWidth);
    const RT = R_GAS * config.temperature_K;

    // Dimensionless Ω/RT — spinodal instability when > 2
    const Omega_d = config.material.Omega_Jmol / RT;
    const RT_d = 1.0;

    // Effective barrier height for ε² calculation: (Ω_d − 2) / 4
    // This is the curvature of the regular solution free energy at φ = 0.5
    const A_eff = Math.max(0.01, (Omega_d - 2.0) / 4.0);

    // Gradient energy from interface width
    const epsilon_sq = epsilonSqFromInterfaceWidth(config.interfaceWidth_px, A_eff);
    const epsilon = Math.sqrt(epsilon_sq);

    // Mobility from Arrhenius, normalized to T_spinodal
    const T_ref = config.material.T_spinodal;
    const M = Math.exp(-config.material.Q / RT)
            / Math.exp(-config.material.Q / (R_GAS * T_ref));

    const dt = computeCHDt(M, epsilon, A_eff, dx);

    this._state.dx = dx;
    this._state.dt = dt;
    this._state.diffusivity = M; // informational
    this._state.r = 0; // not meaningful for CH

    this.writeUniforms(dx, dt, M, epsilon_sq, Omega_d, RT_d);
  }

  /** Write the GPU uniform struct: width(u32) height(u32) dt dx M epsilon_sq Omega RT — 32 bytes. */
  private writeUniforms(
    dx: number, dt: number, M: number, epsilon_sq: number,
    Omega: number, RT: number,
  ): void {
    const buf = new ArrayBuffer(UNIFORM_SIZE);
    const u32View = new Uint32Array(buf);
    const f32View = new Float32Array(buf);
    u32View[0] = this.width;
    u32View[1] = this.height;
    f32View[2] = dt;
    f32View[3] = dx;
    f32View[4] = M;
    f32View[5] = epsilon_sq;
    f32View[6] = Omega;
    f32View[7] = RT;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, buf);
  }

  loadField(data: Float32Array<ArrayBuffer>): void {
    this.device.queue.writeBuffer(this.phiA, 0, data);
    this.device.queue.writeBuffer(this.phiB, 0, new Float32Array(this.width * this.height) as Float32Array<ArrayBuffer>);
    this.parity = 0;
    this._state.stepsRun = 0;
    this._state.time = 0;
    this.renderBindGroup = this.makeRenderBindGroup();
  }

  step(n: number): void {
    const encoder = this.device.createCommandEncoder({ label: 'ch-step' });
    const [wx, wy] = dispatchSize(this.width, this.height);

    for (let i = 0; i < n; i++) {
      const muBG = this.parity === 0 ? this.muBindGroupAtoB : this.muBindGroupBtoA;
      const phiBG = this.parity === 0 ? this.phiBindGroupAtoB : this.phiBindGroupBtoA;

      // Pass 1: compute μ from φ (writes muBuf)
      const muPass = encoder.beginComputePass({ label: 'ch-mu' });
      muPass.setPipeline(this.muPipeline);
      muPass.setBindGroup(0, muBG);
      muPass.dispatchWorkgroups(wx, wy);
      muPass.end();

      // Pass 2: advance φ using ∇²μ (reads muBuf — needs separate pass for barrier)
      const phiPass = encoder.beginComputePass({ label: 'ch-phi' });
      phiPass.setPipeline(this.phiPipeline);
      phiPass.setBindGroup(0, phiBG);
      phiPass.dispatchWorkgroups(wx, wy);
      phiPass.end();

      this.parity = 1 - this.parity;
    }

    this.device.queue.submit([encoder.finish()]);
    this._state.stepsRun += n;
    this._state.time += n * this._state.dt;
    this.renderBindGroup = this.makeRenderBindGroup();
  }

  render(): void {
    const encoder = this.device.createCommandEncoder({ label: 'ch-render' });
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

  updateMaterialColors(colorA: [number, number, number], colorB: [number, number, number]): void {
    this.device.queue.writeBuffer(this.materialColorBuffer, 0, new Float32Array([
      ...colorA, 0.0,
      ...colorB, 0.0,
    ]));
  }

  async readField(): Promise<Float32Array> {
    const size = this.width * this.height * BYTES_PER_CELL;
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
    this.phiA.destroy();
    this.phiB.destroy();
    this.muBuf.destroy();
    this.uniformBuffer.destroy();
    this.materialColorBuffer.destroy();
  }

  private makeRenderBindGroup(): GPUBindGroup {
    const renderBGL = this.renderPipeline.getBindGroupLayout(0);
    // Render uses the same uniform struct as Fick's render.wgsl (width, height, r, _pad)
    // We need a separate 16-byte uniform for the render pipeline
    const renderUniformBuf = new ArrayBuffer(16);
    const u32v = new Uint32Array(renderUniformBuf);
    const f32v = new Float32Array(renderUniformBuf);
    u32v[0] = this.width;
    u32v[1] = this.height;
    f32v[2] = 0; // r (unused by render, but part of struct)
    u32v[3] = 0;
    // Write render uniforms into the same uniform buffer (first 16 bytes match)
    // Actually the uniform buffer is 32 bytes with CH-specific fields, but
    // render.wgsl only reads the first 16 bytes (width, height, r, _pad).
    // The GPU won't fault since the buffer is large enough.

    return this.device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.currentBuffer } },
        { binding: 2, resource: { buffer: this.materialColorBuffer } },
      ],
    });
  }
}
