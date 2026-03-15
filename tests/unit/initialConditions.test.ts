import { describe, it, expect } from 'vitest';
import {
  generateSpinodalField,
  generateVoronoiField,
  generateSmoothVoronoiField,
  generateGrainColors,
} from '../../src/app/initialConditions';

describe('generateSpinodalField', () => {
  it('returns correct length (gridSize²)', () => {
    const field = generateSpinodalField(64);
    expect(field.length).toBe(64 * 64);
  });

  it('mean is approximately meanPhi', () => {
    const field = generateSpinodalField(128, 0.5, 0.05);
    let sum = 0;
    for (let i = 0; i < field.length; i++) sum += field[i];
    const mean = sum / field.length;
    expect(mean).toBeCloseTo(0.5, 1); // within ±0.05
  });

  it('all values are in [0, 1]', () => {
    const field = generateSpinodalField(64, 0.5, 0.3);
    for (let i = 0; i < field.length; i++) {
      expect(field[i]).toBeGreaterThanOrEqual(0);
      expect(field[i]).toBeLessThanOrEqual(1);
    }
  });

  it('different calls produce different noise', () => {
    const a = generateSpinodalField(32, 0.5, 0.1);
    const b = generateSpinodalField(32, 0.5, 0.1);
    // Not all values should be identical (astronomically unlikely)
    let anyDifferent = false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) { anyDifferent = true; break; }
    }
    expect(anyDifferent).toBe(true);
  });

  it('respects custom meanPhi', () => {
    const field = generateSpinodalField(128, 0.3, 0.02);
    let sum = 0;
    for (let i = 0; i < field.length; i++) sum += field[i];
    const mean = sum / field.length;
    expect(mean).toBeCloseTo(0.3, 1);
  });
});

describe('generateVoronoiField', () => {
  const gridSize = 64;
  const numGrains = 4;

  it('returns correct packed length (N × gridSize²)', () => {
    const packed = generateVoronoiField(gridSize, numGrains);
    expect(packed.length).toBe(numGrains * gridSize * gridSize);
  });

  it('each cell is assigned to exactly one grain', () => {
    const packed = generateVoronoiField(gridSize, numGrains);
    const WH = gridSize * gridSize;

    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        let ownerCount = 0;
        for (let i = 0; i < numGrains; i++) {
          if (packed[i * WH + y * gridSize + x] === 1.0) ownerCount++;
        }
        expect(ownerCount).toBe(1);
      }
    }
  });

  it('all grains have at least one cell', () => {
    const packed = generateVoronoiField(gridSize, numGrains);
    const WH = gridSize * gridSize;

    for (let i = 0; i < numGrains; i++) {
      let count = 0;
      for (let j = 0; j < WH; j++) {
        if (packed[i * WH + j] === 1.0) count++;
      }
      expect(count).toBeGreaterThan(0);
    }
  });

  it('values are only 0 or 1', () => {
    const packed = generateVoronoiField(gridSize, numGrains);
    for (let i = 0; i < packed.length; i++) {
      expect(packed[i] === 0 || packed[i] === 1).toBe(true);
    }
  });
});

describe('generateSmoothVoronoiField', () => {
  const gridSize = 64;
  const numGrains = 4;
  const xi_px = 6;

  it('returns correct packed length (N × gridSize²)', () => {
    const packed = generateSmoothVoronoiField(gridSize, numGrains, xi_px);
    expect(packed.length).toBe(numGrains * gridSize * gridSize);
  });

  it('Ση_i = 1 at every cell (partition of unity)', () => {
    const packed = generateSmoothVoronoiField(gridSize, numGrains, xi_px);
    const WH = gridSize * gridSize;

    for (let j = 0; j < WH; j++) {
      let sumEta = 0;
      for (let i = 0; i < numGrains; i++) {
        sumEta += packed[i * WH + j];
      }
      expect(sumEta).toBeCloseTo(1.0, 5);
    }
  });

  it('all η values are in [0, 1]', () => {
    const packed = generateSmoothVoronoiField(gridSize, numGrains, xi_px);
    for (let i = 0; i < packed.length; i++) {
      expect(packed[i]).toBeGreaterThanOrEqual(-1e-6);
      expect(packed[i]).toBeLessThanOrEqual(1.0 + 1e-6);
    }
  });

  it('interior cells have η_owner close to 1.0', () => {
    const packed = generateSmoothVoronoiField(gridSize, numGrains, xi_px);
    const WH = gridSize * gridSize;

    // Count cells where maxEta > 0.9 (well inside a grain)
    // With ξ=6 on 64×64, cells must be ~6px from any boundary for η>0.9.
    // On a small grid with 4 grains, boundaries are close together,
    // so only ~30-40% of cells are deep interior.
    let nearOneCount = 0;
    for (let j = 0; j < WH; j++) {
      let maxEta = 0;
      for (let i = 0; i < numGrains; i++) {
        maxEta = Math.max(maxEta, packed[i * WH + j]);
      }
      if (maxEta > 0.9) nearOneCount++;
    }

    // At least some cells should be deep inside grains
    expect(nearOneCount / WH).toBeGreaterThan(0.1);
  });

  it('boundary cells have smooth (non-binary) η values', () => {
    const packed = generateSmoothVoronoiField(gridSize, numGrains, xi_px);
    const WH = gridSize * gridSize;

    // Count cells where 0.1 < maxEta < 0.9 (near boundaries)
    let boundaryCount = 0;
    for (let j = 0; j < WH; j++) {
      let maxEta = 0;
      for (let i = 0; i < numGrains; i++) {
        maxEta = Math.max(maxEta, packed[i * WH + j]);
      }
      if (maxEta > 0.1 && maxEta < 0.9) boundaryCount++;
    }

    // Should have some boundary cells (smooth transitions)
    expect(boundaryCount).toBeGreaterThan(0);
  });
});

describe('generateGrainColors', () => {
  it('returns correct length (N × 4)', () => {
    const colors = generateGrainColors(8);
    expect(colors.length).toBe(8 * 4);
  });

  it('RGB values are in [0, 1]', () => {
    const colors = generateGrainColors(16);
    for (let i = 0; i < colors.length; i++) {
      expect(colors[i]).toBeGreaterThanOrEqual(0);
      expect(colors[i]).toBeLessThanOrEqual(1);
    }
  });

  it('different grains have distinct hues', () => {
    const colors = generateGrainColors(4);
    // Compare first two grains — they should have different RGB
    const r0 = colors[0], g0 = colors[1], b0 = colors[2];
    const r1 = colors[4], g1 = colors[5], b1 = colors[6];
    const diff = Math.abs(r0 - r1) + Math.abs(g0 - g1) + Math.abs(b0 - b1);
    expect(diff).toBeGreaterThan(0.1);
  });

  it('alpha (pad) channel is 1.0', () => {
    const colors = generateGrainColors(8);
    for (let i = 0; i < 8; i++) {
      expect(colors[i * 4 + 3]).toBe(1.0);
    }
  });
});
