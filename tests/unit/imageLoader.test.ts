import { describe, it, expect } from 'vitest';
import { generateDefaultField } from '../../src/app/imageLoader';

describe('generateDefaultField', () => {
  it('returns correct size', () => {
    const field = generateDefaultField(128);
    expect(field.length).toBe(128 * 128);
  });

  it('left edge is 1.0 (pure material B)', () => {
    const field = generateDefaultField(256);
    for (let y = 0; y < 256; y++) {
      expect(field[y * 256 + 0]).toBeCloseTo(1.0, 5);
    }
  });

  it('right edge is 0.0 (pure material A)', () => {
    const field = generateDefaultField(256);
    for (let y = 0; y < 256; y++) {
      expect(field[y * 256 + 255]).toBeCloseTo(0.0, 5);
    }
  });

  it('center pair averages to 0.5 by symmetry', () => {
    const N = 256;
    const field = generateDefaultField(N);
    const row = Math.floor(N / 2);
    // For even N, center is between cells N/2-1 and N/2
    const left  = field[row * N + (N / 2 - 1)];
    const right = field[row * N + (N / 2)];
    expect((left + right) / 2).toBeCloseTo(0.5, 3);
  });

  it('all values are in [0, 1]', () => {
    const field = generateDefaultField(128);
    for (const v of field) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('is horizontally symmetric: C(x) + C(mirror(x)) ≈ 1', () => {
    const N = 128;
    const field = generateDefaultField(N);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const left  = field[y * N + x];
        const right = field[y * N + (N - 1 - x)];
        expect(left + right).toBeCloseTo(1.0, 3);
      }
    }
  });

  it('is vertically uniform (same pattern on every row)', () => {
    const N = 128;
    const field = generateDefaultField(N);
    for (let x = 0; x < N; x++) {
      expect(field[0 * N + x]).toBe(field[64 * N + x]);
    }
  });

  it('mass ≈ 50% of cells (symmetric IC)', () => {
    const N = 256;
    const field = generateDefaultField(N);
    const totalMass = field.reduce((a, b) => a + b, 0);
    const expectedMass = N * N * 0.5;
    expect(totalMass / expectedMass).toBeCloseTo(1.0, 3);
  });

  it('works for minimum grid size (64)', () => {
    expect(() => generateDefaultField(64)).not.toThrow();
    const field = generateDefaultField(64);
    expect(field.length).toBe(64 * 64);
  });
});
