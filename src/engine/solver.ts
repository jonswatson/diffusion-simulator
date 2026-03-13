// ============================================================
// GPU pipeline orchestration for the diffusion engine.
//
// Manages compute + render pipelines, ping-pong buffers, and
// the per-frame command encoding. The solver never touches the DOM.
// ============================================================

import { createFieldBuffer, createUniformBuffer, dispatchSize, BYTES_PER_CELL } from './buffers';
import diffusionWGSL from './shaders/diffusion.wgsl?raw';
import renderWGSL from './shaders/render.wgsl?raw';

export interface SolverConfig {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
  width: number;
  height: number;
}

export class Solver {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private width: number;
  private height: number;

  // Ping-pong concentration buffers
  private bufA: GPUBuffer;
  private bufB: GPUBuffer;
  private uniformBuffer: GPUBuffer;
  private materialColorBuffer: GPUBuffer;

  // Compute pipeline (fill.wgsl in Stage 1, diffusion.wgsl from Stage 2)
  private computePipeline: GPUComputePipeline;
  private computeBindGroupLayout: GPUBindGroupLayout;
  private bindGroupAtoB: GPUBindGroup;
  private bindGroupBtoA: GPUBindGroup;
  private currentBindGroup: GPUBindGroup;

  // Render pipeline
  private renderPipeline: GPURenderPipeline;
  private renderBindGroup: GPUBindGroup;

  stepsRun = 0;

  constructor(config: SolverConfig) {
    const { device, context, canvasFormat, width, height } = config;
    this.device = device;
    this.context = context;
    this.width = width;
    this.height = height;

    // --- Buffers ---
    this.bufA = createFieldBuffer(device, width, height);
    this.bufB = createFieldBuffer(device, width, height);
    this.uniformBuffer = createUniformBuffer(device, 16); // Uniforms: width, height, r, _pad
    this.materialColorBuffer = createUniformBuffer(device, 32); // 2 × (vec3f + f32 pad)

    // Write initial uniform values
    const uniformData = new ArrayBuffer(16);
    new Uint32Array(uniformData, 0, 2).set([width, height]);
    new Float32Array(uniformData, 8, 1).set([0.1]); // r placeholder
    device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // Default material colors: black (A) and white (B) for Stage 1 checkerboard
    device.queue.writeBuffer(this.materialColorBuffer, 0, new Float32Array([
      0.0, 0.0, 0.0, 0.0, // colorA: black
      1.0, 1.0, 1.0, 0.0, // colorB: white
    ]));

    // --- Compute pipeline (FTCS diffusion stencil) ---
    this.computeBindGroupLayout = device.createBindGroupLayout({
      label: 'compute-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    this.computePipeline = device.createComputePipeline({
      label: 'diffusion-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.computeBindGroupLayout] }),
      compute: { module: device.createShaderModule({ code: diffusionWGSL, label: 'diffusion' }), entryPoint: 'main' },
    });

    this.bindGroupAtoB = this.makeComputeBindGroup(this.bufA, this.bufB);
    this.bindGroupBtoA = this.makeComputeBindGroup(this.bufB, this.bufA);
    this.currentBindGroup = this.bindGroupAtoB;

    // --- Render pipeline ---
    const renderShader = device.createShaderModule({ code: renderWGSL, label: 'render' });
    const renderBindGroupLayout = device.createBindGroupLayout({
      label: 'render-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.renderPipeline = device.createRenderPipeline({
      label: 'render-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
      vertex: { module: renderShader, entryPoint: 'vs_main' },
      fragment: {
        module: renderShader,
        entryPoint: 'fs_main',
        targets: [{ format: canvasFormat }],
      },
    });

    // Render always reads from bufA initially (currentBuffer)
    this.renderBindGroup = device.createBindGroup({
      layout: renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bufA } },
        { binding: 2, resource: { buffer: this.materialColorBuffer } },
      ],
    });
  }

  private makeComputeBindGroup(readBuf: GPUBuffer, writeBuf: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: readBuf } },
        { binding: 2, resource: { buffer: writeBuf } },
      ],
    });
  }

  /** Load a concentration field into the GPU. Resets simulation state. */
  loadField(data: Float32Array<ArrayBuffer>): void {
    this.device.queue.writeBuffer(this.bufA, 0, data);
    this.device.queue.writeBuffer(this.bufB, 0, new Float32Array(this.width * this.height) as Float32Array<ArrayBuffer>);
    this.currentBindGroup = this.bindGroupAtoB;
    this.stepsRun = 0;
    this.updateRenderBindGroup();
  }

  /** Advance the simulation by n timesteps. All dispatches in one command buffer. */
  step(n: number): void {
    const encoder = this.device.createCommandEncoder({ label: 'diffusion-step' });
    const pass = encoder.beginComputePass({ label: 'compute' });
    pass.setPipeline(this.computePipeline);

    const [wx, wy] = dispatchSize(this.width, this.height);
    for (let i = 0; i < n; i++) {
      pass.setBindGroup(0, this.currentBindGroup);
      pass.dispatchWorkgroups(wx, wy);
      this.currentBindGroup = (this.currentBindGroup === this.bindGroupAtoB)
        ? this.bindGroupBtoA
        : this.bindGroupAtoB;
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.stepsRun += n;
    this.updateRenderBindGroup();
  }

  /** Render the current concentration field to the canvas. */
  render(): void {
    const encoder = this.device.createCommandEncoder({ label: 'render' });
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
    pass.draw(3); // full-screen triangle
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  /** Update material display colors. Does not reset the concentration field. */
  updateMaterialColors(colorA: [number, number, number], colorB: [number, number, number]): void {
    this.device.queue.writeBuffer(this.materialColorBuffer, 0, new Float32Array([
      ...colorA, 0.0,
      ...colorB, 0.0,
    ]));
  }

  /** The buffer currently holding the latest concentration values. */
  get currentBuffer(): GPUBuffer {
    return this.currentBindGroup === this.bindGroupAtoB ? this.bufA : this.bufB;
  }

  /** Read the current concentration field from GPU to CPU. Slow — debug/export only. */
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

  /** Rebuild the render bind group to point at the current concentration buffer. */
  private updateRenderBindGroup(): void {
    const renderBGL = this.renderPipeline.getBindGroupLayout(0);
    this.renderBindGroup = this.device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.currentBuffer } },
        { binding: 2, resource: { buffer: this.materialColorBuffer } },
      ],
    });
  }
}
