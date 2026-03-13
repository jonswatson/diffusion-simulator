// ============================================================
// Material database for solid-state diffusion.
//
// D₀ and Q are for the interdiffusion coefficient D̃ in binary systems.
// For the dilute limit, the solute-in-solvent value is used.
//
// Sources:
//   Smithells Metals Reference Book, 8th ed., Table 13.3
//   Shewmon, Diffusion in Solids, 2nd ed.
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

export interface SimConfig {
  material: Material;
  temperature_K: number; // Kelvin
  domainSize_m: number;  // meters — physical width of the simulation domain
  gridWidth: number;     // pixels (square grid for MVP)
}
