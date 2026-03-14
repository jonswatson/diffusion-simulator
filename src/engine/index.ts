// Public API for the diffusion engine.
// No barrel re-exports — each symbol listed explicitly.

export { FickSolver, Solver } from './solver';
export type { SolverConfig } from './solver';
export { CahnHilliardSolver } from './cahnHilliardSolver';
export type { CHSolverConfig } from './cahnHilliardSolver';
export { GrainGrowthSolver } from './grainGrowthSolver';
export type { GGSolverConfig } from './grainGrowthSolver';
export type { DiffusionEngine, SimMode, EngineState } from './types';
export type { Material, SimConfig, FickConfig, CahnHilliardConfig, GrainGrowthConfig, ModeConfig } from './materials';
export { MATERIALS } from './materials';
export {
  R_GAS,
  arrheniusDiffusivity,
  computeDx,
  computeDt,
  computeCHDt,
  computeGGDt,
  fourierNumber,
  erfcApprox,
  analyticalConcentration,
} from './physics';
