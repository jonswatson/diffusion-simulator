// ============================================================
// UI wiring — connects DOM controls to the engine and loop.
//
// Mode-aware: handles Fick diffusion, Cahn-Hilliard spinodal
// decomposition, Allen-Cahn grain growth, and binary solidification.
//
// Temperature cascade (Fick): T -> D(T) -> dt -> r -> writeBuffer
// CH: material + temperature + interface width → physical params
// GG: numGrains (nondimensional, kappa=W=A=L=1 defaults)
// Solidification: Allen-Cahn φ + conserved composition c
// Grid/domain/mode change: full engine recreation (new buffers).
// ============================================================

import {
  FickSolver, CahnHilliardSolver, GrainGrowthSolver, BinarySolidificationSolver,
  MATERIALS, CH_MATERIALS, DEFAULT_SOLID_CONFIG, SOLID_MATERIALS,
} from '../engine';
import type { SimConfig, CahnHilliardConfig, GGConfig, SolidificationConfig } from '../engine';
import type { SimMode, DiffusionEngine } from '../engine/types';
import type { Loop } from './loop';
import { generateDefaultField, imageFileToField } from './imageLoader';
import { generateSpinodalField, generateVoronoiField } from './initialConditions';
import { generateSolidSeed, generatePlanarSolidInterface, generateMultiSeed } from './binarySolidificationInit';
import { createValidation, fieldSum, runValidationChecks } from './validation';
import type { GGValidationParams } from './validation';
import { runGGValidationSuite } from '../engine/ggPhysicsTests';

export interface UIContext {
  solver: DiffusionEngine;
  loop: Loop;
  canvas: HTMLCanvasElement;
  device: GPUDevice;
  context: GPUCanvasContext;
  canvasFormat: GPUTextureFormat;
}

export type FrameCallback = (info: { time: number; stepsRun: number; fps: number }) => void;

/** Format a number in scientific notation for display. */
function sci(value: number, digits = 2): string {
  if (value === 0) return '0';
  return value.toExponential(digits);
}

/** Format seconds into a readable string. */
function formatTime(seconds: number): string {
  if (seconds < 1e-6) return `${(seconds * 1e9).toFixed(1)} ns`;
  if (seconds < 1e-3) return `${(seconds * 1e6).toFixed(1)} µs`;
  if (seconds < 1) return `${(seconds * 1e3).toFixed(1)} ms`;
  if (seconds < 60) return `${seconds.toFixed(2)} s`;
  return `${(seconds / 60).toFixed(1)} min`;
}

/** Wire all DOM controls to the simulation. Returns the per-frame info callback. */
export function initUI(ctx: UIContext): FrameCallback {
  const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

  // DOM elements — transport
  const btnPlay = $<HTMLButtonElement>('btn-play');
  const btnPause = $<HTMLButtonElement>('btn-pause');
  const btnReset = $<HTMLButtonElement>('btn-reset');

  // Mode selector
  const selMode = $<HTMLSelectElement>('sel-mode');
  const fickControls = $<HTMLDivElement>('fick-controls');
  const chControls = $<HTMLDivElement>('ch-controls');
  const ggControls = $<HTMLDivElement>('gg-controls');
  const solidControls = $<HTMLDivElement>('solid-controls');
  const lblImageUpload = $<HTMLLabelElement>('lbl-image-upload');

  // Fick-specific
  const selMaterial = $<HTMLSelectElement>('sel-material');
  const slTemperature = $<HTMLInputElement>('sl-temperature');
  const valTemperature = $<HTMLSpanElement>('val-temperature');

  // Cahn-Hilliard-specific
  const selCHMaterial = $<HTMLSelectElement>('sel-ch-material');
  const slCHTemp = $<HTMLInputElement>('sl-ch-temp');
  const valCHTemp = $<HTMLSpanElement>('val-ch-temp');
  const slCHXi = $<HTMLInputElement>('sl-ch-xi');
  const valCHXi = $<HTMLSpanElement>('val-ch-xi');
  const slCHMean = $<HTMLInputElement>('sl-ch-mean');
  const valCHMean = $<HTMLSpanElement>('val-ch-mean');
  const chSpinodalWarning = $<HTMLDivElement>('ch-spinodal-warning');

  // Grain Growth-specific
  const selGGGrains = $<HTMLSelectElement>('sel-gg-grains');
  const btnGGValidate = $<HTMLButtonElement>('btn-gg-validate');

  // Solidification-specific
  const selSolidIC = $<HTMLSelectElement>('sel-solid-ic');
  const selSolidMaterial = $<HTMLSelectElement>('sel-solid-material');
  const slSolidUndercooling = $<HTMLInputElement>('sl-solid-undercooling');
  const valSolidUndercooling = $<HTMLSpanElement>('val-solid-undercooling');
  const slSolidSeedRadius = $<HTMLInputElement>('sl-solid-seed-radius');
  const valSolidSeedRadius = $<HTMLSpanElement>('val-solid-seed-radius');

  // Domain / grid
  const selDomain = $<HTMLSelectElement>('sel-domain');
  const selGrid = $<HTMLSelectElement>('sel-grid');
  const slSpeed = $<HTMLInputElement>('sl-speed');
  const valSpeed = $<HTMLSpanElement>('val-speed');
  const inpImage = $<HTMLInputElement>('inp-image');

  // Info readouts
  const infoMode = $('info-mode');
  const infoD = $('info-D');
  const infoDt = $('info-dt');
  const infoDx = $('info-dx');
  const infoR = $('info-r');
  const infoTime = $('info-time');
  const infoSteps = $('info-steps');
  const infoFps = $('info-fps');
  const lblMassError = $('lbl-mass-error');
  const infoMassError = $('info-mass-error');
  const infoRmsError = $('info-rms-error');
  const infoSimplexResidual = $('info-simplex-residual');
  const infoMinMaxPhi = $('info-min-max-phi');
  const rowSimplexResidual = $('row-simplex-residual');
  const rowMinMaxPhi = $('row-min-max-phi');
  const warningsDiv = $('warnings');

  // Validation state
  let validation = createValidation();

  // Current config state
  let currentMode: SimMode = 'fick';
  let currentMaterialKey = selMaterial.value;
  let currentTemperature = parseInt(slTemperature.value, 10);
  let currentDomainSize = parseFloat(selDomain.value);
  let currentGridWidth = parseInt(selGrid.value, 10);

  // GG nondimensional defaults
  const GG_KAPPA = 1.0;
  const GG_W = 1.0;
  const GG_A = 1.0;
  const GG_L = 1.0;

  // ---- Engine factory ----

  /** Create a new solver for the given mode and grid width. */
  function createEngine(mode: SimMode, gridWidth: number): DiffusionEngine {
    const base = { device: ctx.device, context: ctx.context, canvasFormat: ctx.canvasFormat, width: gridWidth, height: gridWidth };
    switch (mode) {
      case 'fick':
        return new FickSolver(base);
      case 'cahn-hilliard':
        return new CahnHilliardSolver(base);
      case 'grain-growth': {
        const numGrains = parseInt(selGGGrains.value, 10);
        return new GrainGrowthSolver({ ...base, numGrains });
      }
      case 'binary-solidification':
        return new BinarySolidificationSolver(base);
    }
  }

  /** Destroy the old solver, create a new one, and wire it into the loop. */
  function recreateEngine(mode: SimMode, gridWidth: number): void {
    ctx.solver.destroy();
    ctx.canvas.width = gridWidth;
    ctx.canvas.height = gridWidth;
    ctx.context.configure({ device: ctx.device, format: ctx.canvasFormat });
    const newSolver = createEngine(mode, gridWidth);
    ctx.solver = newSolver;
    ctx.loop.setEngine(newSolver);
  }

  // ---- Mode-aware config & field ----

  /** Build a Fick SimConfig from current UI state. */
  function makeFickConfig(): SimConfig {
    return {
      material: MATERIALS[currentMaterialKey],
      temperature_K: currentTemperature,
      domainSize_m: currentDomainSize,
      gridWidth: currentGridWidth,
    };
  }

  /** Build a CH config from current UI state. */
  function makeCHConfig(): CahnHilliardConfig {
    return {
      mode: 'cahn-hilliard',
      material: CH_MATERIALS[selCHMaterial.value],
      temperature_K: parseInt(slCHTemp.value, 10),
      interfaceWidth_m: parseFloat(slCHXi.value) * 1e-6, // µm → m
      domainSize_m: currentDomainSize,
      gridWidth: currentGridWidth,
    };
  }

  /** Build a GG config from current UI state. */
  function makeGGConfig(): GGConfig {
    return {
      mode: 'grain-growth',
      numGrains: parseInt(selGGGrains.value, 10),
      kappa: GG_KAPPA,
      W: GG_W,
      A: GG_A,
      L: GG_L,
      gridWidth: currentGridWidth,
    };
  }

  /** Build a solidification config from current UI state. */
  function makeSolidConfig(): SolidificationConfig {
    const mat = SOLID_MATERIALS[selSolidMaterial.value];
    return {
      mode: 'binary-solidification',
      kappa_phi: DEFAULT_SOLID_CONFIG.kappa_phi,
      w: DEFAULT_SOLID_CONFIG.w,
      Lphi: DEFAULT_SOLID_CONFIG.Lphi,
      As: mat.As,
      Al: mat.Al,
      cs_eq: mat.cs_eq,
      cl_eq: mat.cl_eq,
      Ms: DEFAULT_SOLID_CONFIG.Ms,
      Ml: DEFAULT_SOLID_CONFIG.Ml,
      deltaF: parseFloat(slSolidUndercooling.value),
      gridWidth: currentGridWidth,
    };
  }

  /** Get the "base" color for material A (solvent). */
  function getColorA(materialKey: string): [number, number, number] {
    const map: Record<string, [number, number, number]> = {
      'Cu-in-Al': [0.75, 0.75, 0.82],
      'Al-self':  [0.75, 0.75, 0.82],
      'C-in-gFe': [0.65, 0.65, 0.70],
      'Ni-in-Cu': [0.72, 0.45, 0.20],
    };
    return map[materialKey] ?? [0.8, 0.8, 0.8];
  }

  /** Update spinodal warning for CH mode. */
  function updateCHSpinodalWarning(): void {
    const mat = CH_MATERIALS[selCHMaterial.value];
    const T = parseInt(slCHTemp.value, 10);
    if (T >= mat.T_spinodal) {
      chSpinodalWarning.hidden = false;
      chSpinodalWarning.textContent =
        `T = ${T} K \u2265 T\u209B = ${Math.round(mat.T_spinodal)} K \u2014 ` +
        `above spinodal temperature, no decomposition will occur.`;
    } else {
      chSpinodalWarning.hidden = true;
      chSpinodalWarning.textContent = '';
    }
  }

  /** Apply physics config to the current solver. Mode-aware. */
  function applyConfig(): void {
    switch (currentMode) {
      case 'fick': {
        const config = makeFickConfig();
        (ctx.solver as FickSolver).updateConfig(config);
        const mat = MATERIALS[currentMaterialKey];
        ctx.solver.updateMaterialColors(getColorA(currentMaterialKey), mat.color);
        updateWarnings(config);
        break;
      }
      case 'cahn-hilliard': {
        const config = makeCHConfig();
        (ctx.solver as CahnHilliardSolver).updateCHConfig(config);
        const mat = CH_MATERIALS[selCHMaterial.value];
        ctx.solver.updateMaterialColors(mat.colorA, mat.colorB);
        updateCHSpinodalWarning();
        warningsDiv.innerHTML = '';
        break;
      }
      case 'grain-growth': {
        const config = makeGGConfig();
        (ctx.solver as GrainGrowthSolver).updateGGConfig(config);
        warningsDiv.innerHTML = '';
        break;
      }
      case 'binary-solidification': {
        const config = makeSolidConfig();
        (ctx.solver as BinarySolidificationSolver).updateSolidConfig(config);
        warningsDiv.innerHTML = '';
        break;
      }
    }
  }

  /** Reload the default field for the current mode and reset simulation. */
  function resetField(): void {
    const wasPlaying = ctx.loop.playing;
    if (wasPlaying) ctx.loop.pause();

    let field: Float32Array;
    switch (currentMode) {
      case 'fick':
        field = generateDefaultField(currentGridWidth);
        break;
      case 'cahn-hilliard': {
        const meanPhi = parseFloat(slCHMean.value);
        field = generateSpinodalField(currentGridWidth, meanPhi, 0.05);
        break;
      }
      case 'grain-growth': {
        const numGrains = parseInt(selGGGrains.value, 10);
        field = generateVoronoiField(currentGridWidth, numGrains);
        break;
      }
      case 'binary-solidification': {
        const cfg = makeSolidConfig();
        const mat = SOLID_MATERIALS[selSolidMaterial.value];
        const ic = selSolidIC.value;
        const seedRadiusPx = (parseInt(slSolidSeedRadius.value, 10) / 100) * currentGridWidth;
        if (ic === 'planar') {
          field = generatePlanarSolidInterface(currentGridWidth, cfg.cs_eq, cfg.cl_eq);
        } else if (ic === 'multi') {
          field = generateMultiSeed(currentGridWidth, 4, seedRadiusPx, mat.c0, cfg.cs_eq);
        } else {
          // 'seed' — default
          field = generateSolidSeed(currentGridWidth, seedRadiusPx, mat.c0, cfg.cs_eq, cfg.cl_eq);
        }
        break;
      }
    }

    ctx.solver.loadField(field as Float32Array<ArrayBuffer>);
    validation = createValidation();
    validation.initialMass = fieldSum(field);

    if (wasPlaying) ctx.loop.play();
  }

  /** Check for warning conditions (Fick only). */
  function updateWarnings(config: SimConfig): void {
    const warnings: string[] = [];
    const mat = config.material;
    const T = config.temperature_K;

    if (T < mat.T_min || T > mat.T_max) {
      warnings.push(
        `Temperature ${T} K is outside the validated range ` +
        `(${mat.T_min}\u2013${mat.T_max} K) for ${mat.name}.`
      );
    }

    const state = ctx.solver.state;
    if (state.r > 0.2) {
      warnings.push(`Fourier number r = ${state.r.toFixed(3)} is close to stability limit (0.25).`);
    }

    const dx_px = config.domainSize_m / config.gridWidth;
    const interfaceWidth_px = 2 * Math.sqrt(state.diffusivity * Math.max(state.time, 1)) / dx_px;
    if (interfaceWidth_px < 2 && state.time > 0) {
      warnings.push('Interface width < 2 pixels \u2014 increase grid resolution or domain size.');
    }

    warningsDiv.innerHTML = warnings
      .map(w => `<div class="warning">${w}</div>`)
      .join('');
  }

  /** Show/hide mode-specific control sections and diagnostic rows. */
  function updateModeVisibility(): void {
    fickControls.hidden = currentMode !== 'fick';
    chControls.hidden = currentMode !== 'cahn-hilliard';
    ggControls.hidden = currentMode !== 'grain-growth';
    solidControls.hidden = currentMode !== 'binary-solidification';
    // Image upload is only useful for Fick
    lblImageUpload.hidden = currentMode !== 'fick';
    // Simplex diagnostics only meaningful for GG
    rowSimplexResidual.hidden = currentMode !== 'grain-growth';
    rowMinMaxPhi.hidden = currentMode !== 'grain-growth';
  }

  /** Mode display name for the info panel. */
  function modeName(mode: SimMode): string {
    switch (mode) {
      case 'fick': return 'Fick';
      case 'cahn-hilliard': return 'Spinodal';
      case 'grain-growth': return 'Grain Growth';
      case 'binary-solidification': return 'Solidification';
    }
  }

  // ---- Play / Pause / Reset ----
  btnPlay.addEventListener('click', () => {
    ctx.loop.play();
    btnPlay.disabled = true;
    btnPause.disabled = false;
  });

  btnPause.addEventListener('click', () => {
    ctx.loop.pause();
    btnPlay.disabled = false;
    btnPause.disabled = true;
  });

  btnReset.addEventListener('click', () => {
    ctx.loop.pause();
    btnPlay.disabled = false;
    btnPause.disabled = true;
    resetField();
  });

  // ---- Mode selector ----
  selMode.addEventListener('change', () => {
    const wasPlaying = ctx.loop.playing;
    if (wasPlaying) ctx.loop.pause();

    currentMode = selMode.value as SimMode;
    updateModeVisibility();
    recreateEngine(currentMode, currentGridWidth);
    applyConfig();
    resetField();

    if (wasPlaying) {
      ctx.loop.play();
      btnPlay.disabled = true;
      btnPause.disabled = false;
    }
  });

  // ---- Material (Fick) ----
  selMaterial.addEventListener('change', () => {
    currentMaterialKey = selMaterial.value;
    applyConfig();
  });

  // ---- Temperature (Fick) ----
  slTemperature.addEventListener('input', () => {
    currentTemperature = parseInt(slTemperature.value, 10);
    valTemperature.textContent = String(currentTemperature);
    applyConfig();
  });

  // ---- CH controls ----
  selCHMaterial.addEventListener('change', () => {
    applyConfig();
  });
  slCHTemp.addEventListener('input', () => {
    valCHTemp.textContent = slCHTemp.value;
    applyConfig();
  });
  slCHXi.addEventListener('input', () => {
    valCHXi.textContent = slCHXi.value;
    applyConfig();
  });
  slCHMean.addEventListener('input', () => {
    valCHMean.textContent = parseFloat(slCHMean.value).toFixed(2);
    // Mean phi only affects the IC, not the running sim config
  });

  // ---- GG validation suite ----
  btnGGValidate.addEventListener('click', async () => {
    const wasPlaying = ctx.loop.playing;
    if (wasPlaying) ctx.loop.pause();
    btnGGValidate.disabled = true;
    btnGGValidate.textContent = 'Running…';
    warningsDiv.innerHTML = '<div class="warning">Running physics validation suite (may take ~30 s)…</div>';

    try {
      const results = await runGGValidationSuite(ctx.device, ctx.canvasFormat);
      const html = results.map(r => {
        const icon = r.passed ? '✓' : '✗';
        const cls = r.passed ? 'info' : 'warning';
        return `<div class="${cls}">${icon} <strong>${r.name}</strong>: ${r.reason}</div>`;
      }).join('');
      warningsDiv.innerHTML = html;
    } catch (e) {
      warningsDiv.innerHTML = `<div class="warning">Validation error: ${String(e)}</div>`;
    } finally {
      btnGGValidate.disabled = false;
      btnGGValidate.textContent = 'Run validation suite';
      if (wasPlaying) ctx.loop.play();
    }
  });

  // ---- Solidification controls ----
  selSolidIC.addEventListener('change', () => {
    resetField();
  });

  selSolidMaterial.addEventListener('change', () => {
    applyConfig();
    resetField();
  });

  slSolidUndercooling.addEventListener('input', () => {
    valSolidUndercooling.textContent = parseFloat(slSolidUndercooling.value).toFixed(2);
    applyConfig();
  });

  slSolidSeedRadius.addEventListener('input', () => {
    valSolidSeedRadius.textContent = slSolidSeedRadius.value;
    // Seed size only takes effect on next reset — no applyConfig needed
  });

  // ---- GG controls ----
  selGGGrains.addEventListener('change', () => {
    // Changing grain count requires engine recreation (buffer size changes)
    const wasPlaying = ctx.loop.playing;
    if (wasPlaying) ctx.loop.pause();
    recreateEngine('grain-growth', currentGridWidth);
    applyConfig();
    resetField();
    if (wasPlaying) {
      ctx.loop.play();
      btnPlay.disabled = true;
      btnPause.disabled = false;
    }
  });

  // ---- Domain size ----
  selDomain.addEventListener('change', () => {
    currentDomainSize = parseFloat(selDomain.value);
    applyConfig();
    resetField();
  });

  // ---- Grid resolution (requires engine recreation) ----
  selGrid.addEventListener('change', () => {
    const wasPlaying = ctx.loop.playing;
    if (wasPlaying) ctx.loop.pause();

    currentGridWidth = parseInt(selGrid.value, 10);
    recreateEngine(currentMode, currentGridWidth);
    applyConfig();
    resetField();

    if (wasPlaying) {
      ctx.loop.play();
      btnPlay.disabled = true;
      btnPause.disabled = false;
    }
  });

  // ---- Speed (logarithmic: 10^0 to 10^5) ----
  function updateSpeed(): void {
    const logVal = parseFloat(slSpeed.value);
    const sps = Math.round(Math.pow(10, logVal));
    valSpeed.textContent = sps.toLocaleString();
    ctx.loop.setStepsPerSecond(sps);
  }
  slSpeed.addEventListener('input', updateSpeed);
  updateSpeed(); // set initial

  // ---- Image upload (Fick only) ----
  inpImage.addEventListener('change', async () => {
    if (currentMode !== 'fick') return;
    const file = inpImage.files?.[0];
    if (!file) return;

    const wasPlaying = ctx.loop.playing;
    if (wasPlaying) ctx.loop.pause();

    const field = await imageFileToField(file, currentGridWidth) as Float32Array<ArrayBuffer>;
    ctx.solver.loadField(field);
    validation = createValidation();
    validation.initialMass = fieldSum(field);

    if (wasPlaying) ctx.loop.play();
  });

  // Initialize validation with current default field
  {
    const initField = generateDefaultField(currentGridWidth);
    validation.initialMass = fieldSum(initField);
  }

  // Set initial mode visibility
  updateModeVisibility();

  // Return the per-frame callback for the loop to call
  return (info) => {
    const state = ctx.solver.state;

    // Mode name
    infoMode.textContent = modeName(currentMode);

    // D(T) and r are only meaningful for Fick
    if (currentMode === 'fick') {
      infoD.textContent = `${sci(state.diffusivity)} m\u00B2/s`;
      infoR.textContent = state.r.toFixed(4);
    } else {
      infoD.textContent = '\u2014';
      infoR.textContent = '\u2014';
    }

    // dt display: solidification is nondimensional — show raw value
    infoDt.textContent = currentMode === 'binary-solidification'
      ? sci(state.dt)
      : `${sci(state.dt)} s`;
    infoDx.textContent = currentMode === 'binary-solidification'
      ? '1 (nondim)'
      : `${sci(state.dx)} m`;
    infoTime.textContent = currentMode === 'binary-solidification'
      ? sci(state.time)
      : formatTime(state.time);
    infoSteps.textContent = info.stepsRun.toLocaleString();
    infoFps.textContent = `${info.fps.toFixed(0)}`;

    // Validation readouts — mode-dependent
    if (currentMode === 'grain-growth') {
      lblMassError.textContent = 'Grains';
      infoMassError.textContent = validation.lastGrainCount > 0
        ? String(validation.lastGrainCount)
        : '\u2014';
      // Repurpose RMS slot for full free energy
      infoRmsError.textContent = validation.lastFreeEnergy > 0
        ? sci(validation.lastFreeEnergy, 3)
        : '\u2014';
      // GG-specific rows
      infoSimplexResidual.textContent = validation.lastCheckStep > 0
        ? sci(validation.lastSimplexResidual, 2)
        : '\u2014';
      infoMinMaxPhi.textContent = validation.lastCheckStep > 0
        ? `${validation.lastMinPhi.toFixed(4)} / ${validation.lastMaxPhi.toFixed(4)}`
        : '\u2014';
    } else {
      lblMassError.textContent = 'Mass drift';
      infoMassError.textContent = validation.lastMassError > 0
        ? `${(validation.lastMassError * 100).toFixed(4)}%`
        : '\u2014';
      infoRmsError.textContent = currentMode === 'fick' && validation.lastAnalyticalRMS > 0
        ? `${(validation.lastAnalyticalRMS * 100).toFixed(2)}%`
        : '\u2014';
    }

    // Run async validation checks (throttled internally)
    if (state.stepsRun > 0) {
      const ggParams: GGValidationParams | undefined = currentMode === 'grain-growth'
        ? { numGrains: parseInt(selGGGrains.value, 10), kappa: GG_KAPPA, W: GG_W, A: GG_A }
        : undefined;
      runValidationChecks(validation, ctx.solver, currentGridWidth, currentMode, ggParams);
    }
  };
}
