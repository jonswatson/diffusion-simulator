// ============================================================
// Shared interfaces for all simulation engines.
//
// Every solver (Fick, Cahn-Hilliard, Grain Growth) implements
// DiffusionEngine so the app layer can swap them freely.
// ============================================================

export type SimMode = 'fick' | 'cahn-hilliard' | 'grain-growth' | 'binary-solidification';

/** CPU-side simulation state. No GPU readback needed. */
export interface EngineState {
  time: number;         // seconds simulated (f64 precision)
  diffusivity: number;  // m²/s — D(T)
  dx: number;           // m/pixel
  dt: number;           // seconds/step
  r: number;            // Fourier number ≈ 0.1
  stepsRun: number;
}

/** Common interface for all GPU-backed simulation engines. */
export interface DiffusionEngine {
  readonly device: GPUDevice;
  readonly state: EngineState;
  readonly stepsRun: number;
  readonly currentBuffer: GPUBuffer;
  readonly mode: SimMode;

  /** Load a field into the GPU. Resets simulation state. */
  loadField(data: Float32Array<ArrayBuffer>): void;

  /** Advance the simulation by n timesteps. */
  step(n: number): void;

  /** Advance and render in one GPU submission when the solver can do so efficiently. */
  stepAndRender?(n: number): void;

  /** Render the current field to the canvas. */
  render(): void;

  /** Read the current field from GPU to CPU. Slow — debug/export only. */
  readField(): Promise<Float32Array>;

  /** Update the display colormap endpoints. */
  updateMaterialColors(colorA: [number, number, number], colorB: [number, number, number]): void;

  /** Release all GPU resources held by this engine. */
  destroy(): void;
}
