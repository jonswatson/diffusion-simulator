// ============================================================
// UI wiring — connects DOM controls to the engine and loop.
//
// Temperature cascade: T → D(T) → dt → r → writeBuffer
// Grid/domain change: full engine recreation (new buffers).
// Material change: update colors + recompute physics.
// ============================================================

import { Solver, MATERIALS } from '../engine';
import type { SimConfig } from '../engine';
import type { Loop } from './loop';
import { generateDefaultField, imageFileToField } from './imageLoader';

export interface UIContext {
  solver: Solver;
  loop: Loop;
  canvas: HTMLCanvasElement;
  recreateEngine: (gridWidth: number) => Promise<void>;
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

  // DOM elements
  const btnPlay = $<HTMLButtonElement>('btn-play');
  const btnPause = $<HTMLButtonElement>('btn-pause');
  const btnReset = $<HTMLButtonElement>('btn-reset');
  const selMaterial = $<HTMLSelectElement>('sel-material');
  const slTemperature = $<HTMLInputElement>('sl-temperature');
  const valTemperature = $<HTMLSpanElement>('val-temperature');
  const selDomain = $<HTMLSelectElement>('sel-domain');
  const selGrid = $<HTMLSelectElement>('sel-grid');
  const slSpeed = $<HTMLInputElement>('sl-speed');
  const valSpeed = $<HTMLSpanElement>('val-speed');
  const inpImage = $<HTMLInputElement>('inp-image');

  // Info readouts
  const infoD = $('info-D');
  const infoDt = $('info-dt');
  const infoDx = $('info-dx');
  const infoR = $('info-r');
  const infoTime = $('info-time');
  const infoSteps = $('info-steps');
  const infoFps = $('info-fps');
  const warningsDiv = $('warnings');

  // Current config state
  let currentMaterialKey = selMaterial.value;
  let currentTemperature = parseInt(slTemperature.value, 10);
  let currentDomainSize = parseFloat(selDomain.value);
  let currentGridWidth = parseInt(selGrid.value, 10);

  /** Build a SimConfig from current UI state. */
  function makeConfig(): SimConfig {
    return {
      material: MATERIALS[currentMaterialKey],
      temperature_K: currentTemperature,
      domainSize_m: currentDomainSize,
      gridWidth: currentGridWidth,
    };
  }

  /** Get the "base" color for material A (solvent). */
  function getColorA(materialKey: string): [number, number, number] {
    // Use a neutral/light color for the matrix/solvent
    const map: Record<string, [number, number, number]> = {
      'Cu-in-Al': [0.75, 0.75, 0.82],  // aluminium silver
      'Al-self':  [0.75, 0.75, 0.82],  // aluminium silver
      'C-in-gFe': [0.65, 0.65, 0.70],  // iron grey
      'Ni-in-Cu': [0.72, 0.45, 0.20],  // copper
    };
    return map[materialKey] ?? [0.8, 0.8, 0.8];
  }

  /** Update physics + GPU from current config. Does NOT reload field. */
  function applyConfig(): void {
    const config = makeConfig();
    ctx.solver.updateConfig(config);
    const mat = MATERIALS[currentMaterialKey];
    ctx.solver.updateMaterialColors(getColorA(currentMaterialKey), mat.color);
    updateWarnings(config);
  }

  /** Reload the default field and reset simulation. */
  function resetField(): void {
    const wasPlaying = ctx.loop.playing;
    if (wasPlaying) ctx.loop.pause();
    const field = generateDefaultField(currentGridWidth) as Float32Array<ArrayBuffer>;
    ctx.solver.loadField(field);
    if (wasPlaying) ctx.loop.play();
  }

  /** Check for warning conditions. */
  function updateWarnings(config: SimConfig): void {
    const warnings: string[] = [];
    const mat = config.material;
    const T = config.temperature_K;

    if (T < mat.T_min || T > mat.T_max) {
      warnings.push(
        `Temperature ${T} K is outside the validated range ` +
        `(${mat.T_min}–${mat.T_max} K) for ${mat.name}.`
      );
    }

    const state = ctx.solver.state;
    if (state.r > 0.2) {
      warnings.push(`Fourier number r = ${state.r.toFixed(3)} is close to stability limit (0.25).`);
    }

    const dx_px = config.domainSize_m / config.gridWidth;
    const interfaceWidth_px = 2 * Math.sqrt(state.diffusivity * Math.max(state.time, 1)) / dx_px;
    if (interfaceWidth_px < 2 && state.time > 0) {
      warnings.push('Interface width < 2 pixels — increase grid resolution or domain size.');
    }

    warningsDiv.innerHTML = warnings
      .map(w => `<div class="warning">${w}</div>`)
      .join('');
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

  // ---- Material ----
  selMaterial.addEventListener('change', () => {
    currentMaterialKey = selMaterial.value;
    applyConfig();
  });

  // ---- Temperature ----
  slTemperature.addEventListener('input', () => {
    currentTemperature = parseInt(slTemperature.value, 10);
    valTemperature.textContent = String(currentTemperature);
    applyConfig();
  });

  // ---- Domain size ----
  selDomain.addEventListener('change', () => {
    currentDomainSize = parseFloat(selDomain.value);
    applyConfig();
    resetField();
  });

  // ---- Grid resolution (requires engine recreation) ----
  selGrid.addEventListener('change', async () => {
    const wasPlaying = ctx.loop.playing;
    if (wasPlaying) ctx.loop.pause();

    currentGridWidth = parseInt(selGrid.value, 10);
    await ctx.recreateEngine(currentGridWidth);
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

  // ---- Image upload ----
  inpImage.addEventListener('change', async () => {
    const file = inpImage.files?.[0];
    if (!file) return;

    const wasPlaying = ctx.loop.playing;
    if (wasPlaying) ctx.loop.pause();

    const field = await imageFileToField(file, currentGridWidth) as Float32Array<ArrayBuffer>;
    ctx.solver.loadField(field);

    if (wasPlaying) ctx.loop.play();
  });

  // Return the per-frame callback for the loop to call
  return (info) => {
    const state = ctx.solver.state;
    infoD.textContent = `${sci(state.diffusivity)} m²/s`;
    infoDt.textContent = `${sci(state.dt)} s`;
    infoDx.textContent = `${sci(state.dx)} m`;
    infoR.textContent = state.r.toFixed(4);
    infoTime.textContent = formatTime(state.time);
    infoSteps.textContent = info.stepsRun.toLocaleString();
    infoFps.textContent = `${info.fps.toFixed(0)}`;
  };
}
