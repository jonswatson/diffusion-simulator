/**
 * Convert an uploaded image file to a Float32Array concentration field.
 * Grid size is the physics grid (not the image resolution).
 *
 * Encoding: luminance → concentration
 *   Black (lum=0) → C=0 → pure material A
 *   White (lum=1) → C=1 → pure material B
 */
export async function imageFileToField(
  file: File,
  gridSize: number,
): Promise<Float32Array> {
  const bitmap = await createImageBitmap(file);

  const canvas = new OffscreenCanvas(gridSize, gridSize);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0, gridSize, gridSize);
  bitmap.close();

  const { data } = ctx.getImageData(0, 0, gridSize, gridSize);
  const field = new Float32Array(gridSize * gridSize);

  for (let i = 0; i < gridSize * gridSize; i++) {
    const r = data[i * 4 + 0] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    // sRGB luminance — perceptually correct grayscale conversion
    field[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  return field;
}

/**
 * Generate a default two-phase initial condition when no image is uploaded.
 * Left half: C = 1.0 (pure material B)
 * Right half: C = 0.0 (pure material A)
 * Transition: sine-smoothed over 5% of domain width to avoid sharp-step ringing.
 *
 * This IC matches the analytical error-function solution used in Stage 8 validation.
 */
export function generateDefaultField(gridSize: number): Float32Array {
  const field = new Float32Array(gridSize * gridSize);
  const cx = (gridSize - 1) / 2.0;
  const blendWidth = Math.max(3, Math.floor(gridSize * 0.05));

  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      const d = x - cx;
      let c: number;

      if (d < -blendWidth) {
        c = 1.0;
      } else if (d > blendWidth) {
        c = 0.0;
      } else {
        // Smooth S-curve from 1 to 0 across the interface
        c = 0.5 * (1 - Math.sin((Math.PI * d) / (2 * blendWidth)));
      }

      field[y * gridSize + x] = c;
    }
  }

  return field;
}
