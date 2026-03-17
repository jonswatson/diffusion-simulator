// ============================================================
// Material databases for solid-state diffusion simulation modes.
//
// Fick mode:
//   D₀ and Q are for the interdiffusion coefficient D̃ in binary systems.
//   Sources: Smithells Metals Reference Book, 8th ed., Table 13.3
//            Shewmon, Diffusion in Solids, 2nd ed.
//
// Cahn-Hilliard mode:
//   Ω (interaction parameter) from regular solution model fits.
//   D₀, Q for Arrhenius mobility.
//   Sources: de Fontaine, Solid State Physics 34 (1979)
//            Hillert, Phase Equilibria, Phase Diagrams and Phase Transformations
//
// Grain Growth mode:
//   Q_gb (grain boundary migration activation energy).
//   Sources: Humphreys & Hatherly, Recrystallization and Related Phenomena
//
// Q is stored in J/mol (not kJ/mol). The display layer converts for readability.
// ============================================================

export interface Material {
  name: string;
  symbol: string;
  D0: number;   // m²/s — pre-exponential factor
  Q: number;    // J/mol — activation energy
  color: [number, number, number]; // linear RGB [0, 1] for display
  T_min: number; // K — lower bound of validated temperature range
  T_max: number; // K — upper bound of validated temperature range
}

export const MATERIALS: Record<string, Material> = {
  'Cu-in-Al': {
    name: 'Copper in Aluminium', symbol: 'Cu↔Al',
    D0: 6.47e-5, Q: 136_800,
    color: [0.72, 0.45, 0.20], // copper brown
    T_min: 400, T_max: 900,
  },
  'Al-self': {
    name: 'Aluminium (self-diffusion)', symbol: 'Al↔Al',
    D0: 1.71e-4, Q: 142_200,
    color: [0.71, 0.71, 0.82], // silver-blue
    T_min: 700, T_max: 933,
  },
  'C-in-gFe': {
    name: 'Carbon in γ-Iron (austenite)', symbol: 'C↔Fe',
    D0: 2.30e-5, Q: 148_000,
    color: [0.16, 0.16, 0.16], // near-black (carbon)
    T_min: 1000, T_max: 1400,
  },
  'Ni-in-Cu': {
    name: 'Nickel in Copper', symbol: 'Ni↔Cu',
    D0: 1.93e-4, Q: 230_000,
    color: [0.47, 0.63, 0.47], // muted green (nickel)
    T_min: 900, T_max: 1300,
  },
};

// ============================================================
// Cahn-Hilliard material database (binary regular solution systems)
// ============================================================

export interface CHMaterial {
  name: string;
  symbol: string;
  Omega_Jmol: number;    // J/mol — regular solution interaction parameter
  D0: number;            // m²/s — pre-exponential for Arrhenius mobility
  Q: number;             // J/mol — activation energy for mobility
  T_spinodal: number;    // K — Ω/(2R), max T for spinodal decomposition
  T_min: number;         // K — lower bound of validated range
  T_max: number;         // K — upper bound of validated range
  colorA: [number, number, number]; // linear RGB for phase A
  colorB: [number, number, number]; // linear RGB for phase B
}

export const CH_MATERIALS: Record<string, CHMaterial> = {
  'Al-Zn': {
    name: 'Aluminium-Zinc', symbol: 'Al-Zn',
    Omega_Jmol: 10_000, D0: 2.59e-5, Q: 120_500,
    T_spinodal: 601,  // 10000 / (2 × 8.314) ≈ 601 K
    T_min: 300, T_max: 600,
    colorA: [0.75, 0.75, 0.82], // aluminium silver
    colorB: [0.65, 0.65, 0.70], // zinc grey
  },
  'Fe-Cr': {
    name: 'Iron-Chromium', symbol: 'Fe-Cr',
    Omega_Jmol: 20_500, D0: 2.33e-4, Q: 241_000,
    T_spinodal: 1233, // 20500 / (2 × 8.314) ≈ 1233 K — 475°C embrittlement regime
    T_min: 600, T_max: 1200,
    colorA: [0.65, 0.65, 0.70], // iron grey
    colorB: [0.47, 0.63, 0.47], // chromium green
  },
  'Cu-Ni': {
    name: 'Copper-Nickel', symbol: 'Cu-Ni',
    Omega_Jmol: 8_000, D0: 1.93e-4, Q: 230_000,
    T_spinodal: 481,  // 8000 / (2 × 8.314) ≈ 481 K
    T_min: 300, T_max: 480,
    colorA: [0.72, 0.45, 0.20], // copper brown
    colorB: [0.47, 0.63, 0.47], // nickel green
  },
};

// ============================================================
// Mode-specific config interfaces
// ============================================================

/** Fick's 2nd law: requires a real material + temperature for Arrhenius D(T). */
export interface FickConfig {
  mode: 'fick';
  material: Material;
  temperature_K: number; // Kelvin
  domainSize_m: number;  // meters — physical width of the simulation domain
  gridWidth: number;     // pixels (square grid)
}

/**
 * Cahn-Hilliard spinodal decomposition.
 * Uses regular solution thermodynamics: f(φ) = Ω·φ·(1−φ) + RT·[φ·ln(φ) + (1−φ)·ln(1−φ)]
 * Mobility and gradient energy derived from material properties + temperature.
 */
export interface CahnHilliardConfig {
  mode: 'cahn-hilliard';
  material: CHMaterial;
  temperature_K: number;     // K — simulation temperature
  interfaceWidth_m: number;  // meters — desired diffuse interface width
  domainSize_m: number;      // meters — physical domain size
  gridWidth: number;         // pixels (square grid)
}

/**
 * Nondimensional grain growth config (Allen-Cahn multi-order-parameter model).
 * All parameters are dimensionless; dx = 1 is implicit in the shader Laplacian.
 */
export interface GGConfig {
  mode: 'grain-growth';
  numGrains: number; // number of order parameters (one per grain)
  kappa: number;     // gradient energy coefficient
  W: number;         // double-well barrier height
  A: number;         // grain-grain interaction coefficient
  L: number;         // Allen-Cahn kinetic coefficient
  gridWidth: number; // pixels (square grid)
}

/**
 * Binary alloy solidification config.
 * Coupled Allen-Cahn (φ, non-conserved) + diffusion (c, conserved).
 * All parameters are nondimensional (dx = 1 implicit).
 *
 * Free energy: f = h(φ)·As·(c−cs_eq)² + (1−h(φ))·Al·(c−cl_eq)² + w·g(φ)
 * where h(φ) = φ²(3−2φ), g(φ) = φ²(1−φ)²
 */
export interface SolidificationConfig {
  mode: 'binary-solidification';
  kappa_phi: number; // gradient energy coefficient for phase field
  w: number;         // double-well barrier height
  Lphi: number;      // Allen-Cahn kinetic coefficient
  As: number;        // curvature of solid free energy [nondimensional]
  Al: number;        // curvature of liquid free energy [nondimensional]
  cs_eq: number;     // equilibrium solid composition (mole fraction)
  cl_eq: number;     // equilibrium liquid composition (mole fraction)
  Ms: number;        // composition mobility in solid phase
  Ml: number;        // composition mobility in liquid phase
  gridWidth: number; // pixels (square grid)
}

/** Default solidification parameters (nondimensional). */
export const DEFAULT_SOLID_CONFIG: Omit<SolidificationConfig, 'mode' | 'gridWidth'> = {
  kappa_phi: 1.0,
  w: 1.0,
  Lphi: 1.0,
  As: 2.0,
  Al: 2.0,
  cs_eq: 0.3,
  cl_eq: 0.7,
  Ms: 0.01,
  Ml: 1.0,
};

/** Legacy SimConfig for backward compatibility with Fick-only code. */
export interface SimConfig {
  material: Material;
  temperature_K: number; // Kelvin
  domainSize_m: number;  // meters — physical width of the simulation domain
  gridWidth: number;     // pixels (square grid for MVP)
}

/** Union of all mode-specific configs. */
export type ModeConfig = FickConfig | CahnHilliardConfig | GGConfig | SolidificationConfig;
