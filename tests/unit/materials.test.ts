import { describe, it, expect } from 'vitest';
import { MATERIALS } from '../../src/engine/materials';

describe('MATERIALS database integrity', () => {
  const keys = Object.keys(MATERIALS);

  it('database has at least 4 entries', () => {
    expect(keys.length).toBeGreaterThanOrEqual(4);
  });

  for (const key of keys) {
    describe(key, () => {
      const mat = MATERIALS[key];

      it('has positive D0', () => expect(mat.D0).toBeGreaterThan(0));
      it('has positive Q',  () => expect(mat.Q).toBeGreaterThan(0));

      it('D0 is in physically realistic range for metals [1e-7, 1e-3 m²/s]', () => {
        expect(mat.D0).toBeGreaterThan(1e-7);
        expect(mat.D0).toBeLessThan(1e-3);
      });

      it('Q is in realistic range for solid-state diffusion [50, 400 kJ/mol]', () => {
        expect(mat.Q).toBeGreaterThan(50_000);
        expect(mat.Q).toBeLessThan(400_000);
      });

      it('T_min < T_max', () => expect(mat.T_min).toBeLessThan(mat.T_max));
      it('T_min > 0 K',   () => expect(mat.T_min).toBeGreaterThan(0));

      it('color has 3 components in [0, 1]', () => {
        expect(mat.color).toHaveLength(3);
        for (const c of mat.color) {
          expect(c).toBeGreaterThanOrEqual(0);
          expect(c).toBeLessThanOrEqual(1);
        }
      });

      it('has non-empty name and symbol', () => {
        expect(mat.name.length).toBeGreaterThan(0);
        expect(mat.symbol.length).toBeGreaterThan(0);
      });
    });
  }
});
