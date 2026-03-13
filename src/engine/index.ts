// Public API for the diffusion engine.
// No barrel re-exports — each symbol listed explicitly.

export { Solver } from './solver';
export type { SolverConfig, EngineState } from './solver';
export type { Material, SimConfig } from './materials';
export { MATERIALS } from './materials';
export {
  R_GAS,
  arrheniusDiffusivity,
  computeDx,
  computeDt,
  fourierNumber,
  erfcApprox,
  analyticalConcentration,
} from './physics';
