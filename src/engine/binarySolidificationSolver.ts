// ============================================================
// Binary solidification solver — coupled Allen-Cahn + diffusion on the GPU.
//
// Three compute passes per step:
//   Pass 1 (aux):         φ, c → dF/dφ, μc, M(φ)         (thermodynamics)
//   Pass 2 (phase):       φ, dF/dφ → φ_new               (Allen-Cahn)
//   Pass 3 (composition): c, μc, M → c_new                (conserved diffusion)
//
// Each pass gets its own beginComputePass/endComputePass so that WebGPU
// memory barriers separate aux writes from phase/composition reads.
//
// State stored in ping-pong buffers (phiA/phiB, cA/cB).
// Auxiliary buffers (dfdPhiBuf, muCBuf, mobBuf) are rewritten every step.
//
// loadField / readField use a 2N layout: first N floats = φ, second N = c.
// ============================================================

import { createFieldBuffer, createUniformBuffer, dispatchSize, BYTES_PER_CELL } from './buffers';
import { computeSolidificationDt } from './physics';
import type { DiffusionEngine, SimMode, EngineState } from './types';
import type { SolidificationConfig } from './materials';
import auxWGSL   from './shaders/binarySolidificationAux.wgsl?raw';
import phaseWGSL from './shaders/binarySolidificationPhase.wgsl?raw';
import compWGSL  from './shaders/binarySolidificationComposition.wgsl?raw';
import renderWGSL from './shaders/renderSolidification.wgsl?raw';

export interface SolidSolverConfig {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
  width: number;
  height: number;
}

/** Uniform struct layout: 16 × f32 = 64 bytes, all fields 4-byte aligned. */
const UNIFORM_SIZE = 64; // bytes

export class BinarySolidificationSolver implements DiffusionEngine {
  readonly mode: SimMode = 'binary-solidification';
  readonly device: GPUDevice;
  private context: GPUCanvasContext;
  private width: number;
  private height: number;

  // Ping-pong phase field (φ)
  private phiA: GPUBuffer;
  private phiB: GPUBuffer;
  // Ping-pong composition (c)
  private cA: GPUBuffer;
  private cB: GPUBuffer;

  // Auxiliary buffers recomputed every step
  private dfdPhiBuf: GPUBuffer;
  private muCBuf: GPUBuffer;
  private mobBuf: GPUBuffer;

  private uniformBuffer: GPUBuffer;
  private materialColorBuffer: GPUBuffer;

  // Compute pipelines
  private auxPipeline: GPUComputePipeline;
  private phasePipeline: GPUComputePipeline;
  private compPipeline: GPUComputePipeline;

  // Bind groups (two directions of ping-pong × 3 passes = 6 total)
  private auxBG_AtoB: GPUBindGroup;
  private phaseBG_AtoB: GPUBindGroup;
  private compBG_AtoB: GPUBindGroup;
  private auxBG_BtoA: GPUBindGroup;
  private phaseBG_BtoA: GPUBindGroup;
  private compBG_BtoA: GPUBindGroup;

  // Render pipeline
  private renderPipeline: GPURenderPipeline;
  private renderBG_A: GPUBindGroup; // reads phiA, cA
  private renderBG_B: GPUBindGroup; // reads phiB, cB

  private parity = 0; // 0: data in A buffers; 1: data in B buffers

  private _state: EngineState = {
    time: 0,
    diffusivity: 0,
    dx: 1,      // nondimensional
    dt: 0.005,
    r: 0,
    stepsRun: 0,
  };

  get state(): EngineState { return this._state; }
  get stepsRun(): number { return this._state.stepsRun; }

  get currentBuffer(): GPUBuffer {
    // Returns current phi buffer (satisfies DiffusionEngine interface)
    return this.parity === 0 ? this.phiA : this.phiB;
  }

  get currentCBuffer(): GPUBuffer {
    return this.parity === 0 ? this.cA : this.cB;
  }

  constructor(config: SolidSolverConfig) {
    const { device, context, canvasFormat, width, height } = config;
    this.device = device;
    this.context = context;
    this.width = width;
    this.height = height;

    // --- Buffers ---
    this.phiA = createFieldBuffer(device, width, height);
    this.phiB = createFieldBuffer(device, width, height);
    this.cA   = createFieldBuffer(device, width, height);
    this.cB   = createFieldBuffer(device, width, height);
    this.dfdPhiBuf = createFieldBuffer(device, width, height);
    this.muCBuf    = createFieldBuffer(device, width, height);
    this.mobBuf    = createFieldBuffer(device, width, height);
    this.uniformBuffer = createUniformBuffer(device, UNIFORM_SIZE);
    this.materialColorBuffer = createUniformBuffer(device, 32);

    // Default colors: blue (liquid, φ=0) and white-gold (solid, φ=1)
    device.queue.writeBuffer(this.materialColorBuffer, 0, new Float32Array([
      0.05, 0.15, 0.55, 0.0,  // colorA: deep blue (liquid)
      0.92, 0.85, 0.60, 0.0,  // colorB: warm gold (solid)
    ]));

    // --- Bind group layouts ---
    const auxBGL = device.createBindGroupLayout({
      label: 'solid-aux-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const phaseBGL = device.createBindGroupLayout({
      label: 'solid-phase-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const compBGL = device.createBindGroupLayout({
      label: 'solid-comp-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const renderBGL = device.createBindGroupLayout({
      label: 'solid-render-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    // --- Pipelines ---
    this.auxPipeline = device.createComputePipeline({
      label: 'solid-aux',
      layout: device.createPipelineLayout({ bindGroupLayouts: [auxBGL] }),
      compute: { module: device.createShaderModule({ code: auxWGSL, label: 'solid-aux' }), entryPoint: 'main' },
    });

    this.phasePipeline = device.createComputePipeline({
      label: 'solid-phase',
      layout: device.createPipelineLayout({ bindGroupLayouts: [phaseBGL] }),
      compute: { module: device.createShaderModule({ code: phaseWGSL, label: 'solid-phase' }), entryPoint: 'main' },
    });

    this.compPipeline = device.createComputePipeline({
      label: 'solid-comp',
      layout: device.createPipelineLayout({ bindGroupLayouts: [compBGL] }),
      compute: { module: device.createShaderModule({ code: compWGSL, label: 'solid-comp' }), entryPoint: 'main' },
    });

    const renderShader = device.createShaderModule({ code: renderWGSL, label: 'solid-render' });
    this.renderPipeline = device.createRenderPipeline({
      label: 'solid-render',
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
      vertex: { module: renderShader, entryPoint: 'vs_main' },
      fragment: { module: renderShader, entryPoint: 'fs_main', targets: [{ format: canvasFormat }] },
    });

    // --- Bind groups: A→B direction ---
    this.auxBG_AtoB = device.createBindGroup({
      layout: auxBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiA } },
        { binding: 2, resource: { buffer: this.cA } },
        { binding: 3, resource: { buffer: this.dfdPhiBuf } },
        { binding: 4, resource: { buffer: this.muCBuf } },
        { binding: 5, resource: { buffer: this.mobBuf } },
      ],
    });

    this.phaseBG_AtoB = device.createBindGroup({
      layout: phaseBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiA } },
        { binding: 2, resource: { buffer: this.dfdPhiBuf } },
        { binding: 3, resource: { buffer: this.phiB } },
      ],
    });

    this.compBG_AtoB = device.createBindGroup({
      layout: compBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.cA } },
        { binding: 2, resource: { buffer: this.muCBuf } },
        { binding: 3, resource: { buffer: this.mobBuf } },
        { binding: 4, resource: { buffer: this.cB } },
      ],
    });

    // --- Bind groups: B→A direction ---
    this.auxBG_BtoA = device.createBindGroup({
      layout: auxBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiB } },
        { binding: 2, resource: { buffer: this.cB } },
        { binding: 3, resource: { buffer: this.dfdPhiBuf } },
        { binding: 4, resource: { buffer: this.muCBuf } },
        { binding: 5, resource: { buffer: this.mobBuf } },
      ],
    });

    this.phaseBG_BtoA = device.createBindGroup({
      layout: phaseBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiB } },
        { binding: 2, resource: { buffer: this.dfdPhiBuf } },
        { binding: 3, resource: { buffer: this.phiA } },
      ],
    });

    this.compBG_BtoA = device.createBindGroup({
      layout: compBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.cB } },
        { binding: 2, resource: { buffer: this.muCBuf } },
        { binding: 3, resource: { buffer: this.mobBuf } },
        { binding: 4, resource: { buffer: this.cA } },
      ],
    });

    // --- Render bind groups ---
    this.renderBG_A = device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiA } },
        { binding: 2, resource: { buffer: this.cA } },
        { binding: 3, resource: { buffer: this.materialColorBuffer } },
      ],
    });

    this.renderBG_B = device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.phiB } },
        { binding: 2, resource: { buffer: this.cB } },
        { binding: 3, resource: { buffer: this.materialColorBuffer } },
      ],
    });
  }

  /**
   * Apply physics parameters from a SolidificationConfig.
   * Computes stable dt from stability analysis and writes the GPU uniform buffer.
   */
  updateSolidConfig(config: SolidificationConfig): void {
    const dt = computeSolidificationDt(
      config.kappa_phi, config.w, config.Lphi,
      config.As, config.Al, config.Ml,
    );
    this._state.dt = dt;
    this._state.dx = 1.0; // nondimensional
    this._state.r = 0;
    this._state.diffusivity = 0;

    this.writeUniforms(config, dt);
  }

  /** Write the 64-byte GPU uniform struct (16 × f32). */
  private writeUniforms(config: SolidificationConfig, dt: number): void {
    const buf = new ArrayBuffer(UNIFORM_SIZE);
    const u32v = new Uint32Array(buf);
    const f32v = new Float32Array(buf);
    u32v[0] = this.width;
    u32v[1] = this.height;
    f32v[2] = dt;
    f32v[3] = 0; // _pad0 (dx placeholder, nondimensional)
    f32v[4] = config.kappa_phi;
    f32v[5] = config.w;
    f32v[6] = config.Lphi;
    f32v[7] = config.As;
    f32v[8] = config.Al;
    f32v[9] = config.cs_eq;
    f32v[10] = config.cl_eq;
    f32v[11] = config.Ms;
    f32v[12] = config.Ml;
    f32v[13] = 0; // _pad1
    f32v[14] = 0; // _pad2
    f32v[15] = 0; // _pad3
    this.device.queue.writeBuffer(this.uniformBuffer, 0, buf);
  }

  /**
   * Load initial conditions into the GPU.
   * data layout: first N = φ field, second N = c field (N = width × height)
   */
  loadField(data: Float32Array<ArrayBuffer>): void {
    const N = this.width * this.height;
    const phiData = data.subarray(0, N) as Float32Array<ArrayBuffer>;
    const cData   = data.subarray(N, 2 * N) as Float32Array<ArrayBuffer>;
    this.device.queue.writeBuffer(this.phiA, 0, phiData);
    this.device.queue.writeBuffer(this.cA,   0, cData);
    // Zero out the B buffers so they don't contain garbage on the first render
    this.device.queue.writeBuffer(this.phiB, 0, new Float32Array(N) as Float32Array<ArrayBuffer>);
    this.device.queue.writeBuffer(this.cB,   0, new Float32Array(N) as Float32Array<ArrayBuffer>);
    this.parity = 0;
    this._state.stepsRun = 0;
    this._state.time = 0;
  }

  step(n: number): void {
    const encoder = this.device.createCommandEncoder({ label: 'solid-step' });
    const [wx, wy] = dispatchSize(this.width, this.height);

    for (let i = 0; i < n; i++) {
      const isAtoB = this.parity === 0;
      const auxBG   = isAtoB ? this.auxBG_AtoB   : this.auxBG_BtoA;
      const phaseBG = isAtoB ? this.phaseBG_AtoB : this.phaseBG_BtoA;
      const compBG  = isAtoB ? this.compBG_AtoB  : this.compBG_BtoA;

      // Pass 1: compute thermodynamic quantities
      // Must end before Pass 2/3 read the aux buffers
      const auxPass = encoder.beginComputePass({ label: 'solid-aux' });
      auxPass.setPipeline(this.auxPipeline);
      auxPass.setBindGroup(0, auxBG);
      auxPass.dispatchWorkgroups(wx, wy);
      auxPass.end();

      // Pass 2: advance φ using dF/dφ from aux
      const phasePass = encoder.beginComputePass({ label: 'solid-phase' });
      phasePass.setPipeline(this.phasePipeline);
      phasePass.setBindGroup(0, phaseBG);
      phasePass.dispatchWorkgroups(wx, wy);
      phasePass.end();

      // Pass 3: advance c using μc and M from aux
      const compPass = encoder.beginComputePass({ label: 'solid-comp' });
      compPass.setPipeline(this.compPipeline);
      compPass.setBindGroup(0, compBG);
      compPass.dispatchWorkgroups(wx, wy);
      compPass.end();

      this.parity = 1 - this.parity;
    }

    this.device.queue.submit([encoder.finish()]);
    this._state.stepsRun += n;
    this._state.time += n * this._state.dt;
  }

  render(): void {
    const encoder = this.device.createCommandEncoder({ label: 'solid-render' });
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
    pass.setBindGroup(0, this.parity === 0 ? this.renderBG_A : this.renderBG_B);
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

  /**
   * Read both φ and c from GPU. Returns 2N floats: [φ₀…φₙ, c₀…cₙ].
   */
  async readField(): Promise<Float32Array> {
    const N = this.width * this.height;
    const singleSize = N * BYTES_PER_CELL;
    const phiBuf = this.parity === 0 ? this.phiA : this.phiB;
    const cBuf   = this.parity === 0 ? this.cA   : this.cB;

    const stagePhi = this.device.createBuffer({
      size: singleSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const stageC = this.device.createBuffer({
      size: singleSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(phiBuf, 0, stagePhi, 0, singleSize);
    enc.copyBufferToBuffer(cBuf,   0, stageC,   0, singleSize);
    this.device.queue.submit([enc.finish()]);

    await stagePhi.mapAsync(GPUMapMode.READ);
    await stageC.mapAsync(GPUMapMode.READ);

    const result = new Float32Array(2 * N);
    result.set(new Float32Array(stagePhi.getMappedRange()));
    result.set(new Float32Array(stageC.getMappedRange()), N);

    stagePhi.unmap();
    stageC.unmap();
    stagePhi.destroy();
    stageC.destroy();

    return result;
  }

  destroy(): void {
    this.phiA.destroy();
    this.phiB.destroy();
    this.cA.destroy();
    this.cB.destroy();
    this.dfdPhiBuf.destroy();
    this.muCBuf.destroy();
    this.mobBuf.destroy();
    this.uniformBuffer.destroy();
    this.materialColorBuffer.destroy();
  }
}
