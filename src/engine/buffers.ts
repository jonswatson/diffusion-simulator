// ============================================================
// Buffer allocation and layout constants for the diffusion engine.
// All GPU buffers use a flat f32 array layout: buffer[y * width + x].
// ============================================================

export const WORKGROUP_SIZE = 16;
export const BYTES_PER_CELL = 4; // f32

/** Allocate a storage buffer for one scalar field (concentration, chemical potential, etc.) */
export function createFieldBuffer(device: GPUDevice, width: number, height: number): GPUBuffer {
  return device.createBuffer({
    label: `field-${width}x${height}`,
    size: width * height * BYTES_PER_CELL,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

/** Allocate uniform buffer. Size must be a multiple of 16 bytes (WGSL alignment). */
export function createUniformBuffer(device: GPUDevice, sizeBytes: number): GPUBuffer {
  return device.createBuffer({
    label: 'uniforms',
    size: sizeBytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

/** Dispatch dimensions for a 2D grid with 16x16 workgroups. */
export function dispatchSize(width: number, height: number): [number, number] {
  return [
    Math.ceil(width / WORKGROUP_SIZE),
    Math.ceil(height / WORKGROUP_SIZE),
  ];
}
