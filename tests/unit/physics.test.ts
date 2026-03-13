import { describe, it, expect } from 'vitest';
import {
  arrheniusDiffusivity,
  computeDx,
  computeDt,
  fourierNumber,
  erfcApprox,
  analyticalConcentration,
  R_GAS,
} from '../../src/engine/physics';

describe('arrheniusDiffusivity', () => {
  const D0_CuAl = 6.47e-5;
  const Q_CuAl  = 136_800;

  it('matches published value for Cu in Al at 600°C (873 K)', () => {
    const D = arrheniusDiffusivity(D0_CuAl, Q_CuAl, 873);
    expect(D).toBeGreaterThan(1e-13);
    expect(D).toBeLessThan(5e-13);
  });

  it('increases with temperature (Arrhenius)', () => {
    const D_low  = arrheniusDiffusivity(D0_CuAl, Q_CuAl, 700);
    const D_high = arrheniusDiffusivity(D0_CuAl, Q_CuAl, 900);
    expect(D_high).toBeGreaterThan(D_low);
  });

  it('approaches D0 as T → ∞', () => {
    const D = arrheniusDiffusivity(1e-5, 100_000, 1e10);
    expect(D).toBeCloseTo(1e-5, 10);
  });

  it('returns exactly D0 when Q = 0', () => {
    const D = arrheniusDiffusivity(3e-5, 0, 1000);
    expect(D).toBe(3e-5);
  });

  it('matches manual Arrhenius calculation', () => {
    const D0 = 2e-4, Q = 150_000, T = 800;
    const expected = D0 * Math.exp(-Q / (R_GAS * T));
    expect(arrheniusDiffusivity(D0, Q, T)).toBeCloseTo(expected, 20);
  });

  it('is sensitive to temperature in the expected physical range', () => {
    const D1 = arrheniusDiffusivity(D0_CuAl, Q_CuAl, 800);
    const D2 = arrheniusDiffusivity(D0_CuAl, Q_CuAl, 900);
    const ratio = D2 / D1;
    expect(ratio).toBeGreaterThan(2);
    expect(ratio).toBeLessThan(20);
  });
});

describe('computeDx', () => {
  it('100 µm domain / 512 pixels = 195 nm/pixel', () => {
    const dx = computeDx(100e-6, 512);
    expect(dx).toBeCloseTo(1.953e-7, 10);
  });

  it('1 mm domain / 256 pixels ≈ 3.9 µm/pixel', () => {
    const dx = computeDx(1e-3, 256);
    expect(dx).toBeCloseTo(1e-3 / 256, 15);
  });

  it('scales linearly with domain size', () => {
    const dx1 = computeDx(1e-3, 256);
    const dx2 = computeDx(2e-3, 256);
    expect(dx2 / dx1).toBeCloseTo(2, 10);
  });
});

describe('computeDt and fourierNumber', () => {
  const D  = 1e-12;
  const dx = 1e-7;

  it('default safety factor gives r ≈ 0.1', () => {
    const dt = computeDt(D, dx);
    const r  = fourierNumber(D, dt, dx);
    expect(r).toBeCloseTo(0.1, 5);
  });

  it('r ≤ 0.25 for any valid D and dx', () => {
    const cases: [number, number][] = [
      [1e-12, 1e-7],
      [1e-15, 1e-8],
      [1e-10, 5e-7],
    ];
    for (const [Di, dxi] of cases) {
      const dt = computeDt(Di, dxi);
      const r  = fourierNumber(Di, dt, dxi);
      expect(r).toBeLessThanOrEqual(0.25);
    }
  });

  it('throws when r > 0.25', () => {
    // D=1, dt=1, dx=1 → r = 1.0 >> 0.25
    expect(() => fourierNumber(1, 1, 1)).toThrow('Unstable');
  });

  it('manual dt gives r = 0.1 with safety factor 0.4', () => {
    const dt = computeDt(D, dx, 0.4);
    expect(fourierNumber(D, dt, dx)).toBeCloseTo(0.1, 8);
  });
});

describe('erfcApprox', () => {
  it('erfc(0) = 1', () => {
    expect(erfcApprox(0)).toBeCloseTo(1.0, 6);
  });

  it('erfc(∞) ≈ 0', () => {
    expect(erfcApprox(10)).toBeCloseTo(0.0, 6);
  });

  it('erfc(-∞) ≈ 2', () => {
    expect(erfcApprox(-10)).toBeCloseTo(2.0, 6);
  });

  it('erfc(x) + erfc(-x) = 2 (symmetry)', () => {
    for (const x of [0.1, 0.5, 1.0, 2.0]) {
      expect(erfcApprox(x) + erfcApprox(-x)).toBeCloseTo(2.0, 6);
    }
  });

  it('erfc(1) ≈ 0.1573 (known value)', () => {
    expect(erfcApprox(1)).toBeCloseTo(0.157299, 4);
  });

  it('maximum error < 2e-4 vs reference values', () => {
    const references: [number, number][] = [
      [0.0,  1.000000],
      [0.5,  0.479500],
      [1.0,  0.157299],
      [1.5,  0.033895],
      [2.0,  0.004678],
    ];
    for (const [x, expected] of references) {
      expect(Math.abs(erfcApprox(x) - expected)).toBeLessThan(2e-4);
    }
  });
});

describe('analyticalConcentration', () => {
  it('returns 0.5 at the midpoint for all t > 0', () => {
    expect(analyticalConcentration(0, 0, 1e-12, 1)).toBeCloseTo(0.5, 6);
    expect(analyticalConcentration(0, 0, 1e-12, 1000)).toBeCloseTo(0.5, 6);
  });

  it('returns step function at t = 0', () => {
    expect(analyticalConcentration(-1e-6, 0, 1e-12, 0)).toBe(1.0);
    expect(analyticalConcentration( 1e-6, 0, 1e-12, 0)).toBe(0.0);
  });

  it('interface spreads over time (√t scaling)', () => {
    const D = 1e-12, x0 = 0;
    const t1 = 1e6, t2 = 4e6;
    const spread1 = 2 * Math.sqrt(D * t1);
    const spread2 = 2 * Math.sqrt(D * t2);
    expect(spread2 / spread1).toBeCloseTo(2.0, 5);

    const C1 = analyticalConcentration(spread1, x0, D, t1);
    const C2 = analyticalConcentration(spread2, x0, D, t2);
    expect(C1).toBeCloseTo(C2, 3);
  });
});
