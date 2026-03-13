// ============================================================
// Physics utility functions for the diffusion engine.
// All functions are pure (no side effects, no GPU calls).
// Units are SI throughout: meters, seconds, Joules, Kelvin.
// ============================================================

export const R_GAS = 8.314; // J/(mol·K) — universal gas constant

/**
 * Arrhenius equation: D(T) = D₀ · exp(-Q / RT)
 * D₀ [m²/s], Q [J/mol], T [K] → D [m²/s]
 * Describes how diffusivity increases exponentially with temperature.
 */
export function arrheniusDiffusivity(D0: number, Q: number, T_K: number): number {
  return D0 * Math.exp(-Q / (R_GAS * T_K));
}

/**
 * Grid spacing: physical size of one pixel.
 * dx = domainSize / gridWidth  [m/pixel]
 */
export function computeDx(domainSize_m: number, gridWidth_px: number): number {
  return domainSize_m / gridWidth_px;
}

/**
 * Maximum stable timestep for 2D explicit FTCS.
 * Stability requires r = D·dt/dx² ≤ 0.25.
 * The '4' in denominator is the number of neighbors in 2D (not 2 as in 1D).
 * Safety factor 0.4 gives r ≈ 0.1 — well inside the stability region.
 */
export function computeDt(D: number, dx: number, safetyFactor = 0.4): number {
  return safetyFactor * (dx * dx) / (4.0 * D);
}

/**
 * Fourier number: r = D·dt/dx²
 * This single dimensionless parameter encodes all physics for the FTCS shader.
 * Must be ≤ 0.25 for stability. Throws if violated.
 */
export function fourierNumber(D: number, dt: number, dx: number): number {
  const r = D * dt / (dx * dx);
  if (r > 0.25) {
    throw new Error(
      `Unstable: Fourier number r=${r.toFixed(4)} > 0.25. ` +
      `Reduce domain size or increase grid resolution.`
    );
  }
  return r;
}

/**
 * erfc approximation — Abramowitz & Stegun 7.1.26, max error 1.5×10⁻⁷
 * Used for analytical validation against the 1D error-function solution.
 */
export function erfcApprox(x: number): number {
  const t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const result = poly * Math.exp(-x * x);
  return x >= 0 ? result : 2.0 - result;
}

/**
 * Analytical solution for a step-function IC on a semi-infinite domain.
 * C(x, t) = 0.5 · erfc((x - x₀) / (2·√(D·t)))
 * Used in Stage 8 to validate the numerical solver against known physics.
 */
export function analyticalConcentration(x_m: number, x0_m: number, D: number, t: number): number {
  if (t <= 0) return x_m < x0_m ? 1.0 : 0.0;
  return 0.5 * erfcApprox((x_m - x0_m) / (2.0 * Math.sqrt(D * t)));
}
