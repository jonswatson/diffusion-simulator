// ============================================================
// requestAnimationFrame simulation loop with play/pause.
//
// Time accumulation in TypeScript (f64) — no GPU readback for time.
// Steps-per-frame derived from wall-clock delta × stepsPerSecond.
// 100ms frame cap prevents burst after tab backgrounding.
// ============================================================

import type { Solver } from '../engine/solver';

export interface Loop {
  play(): void;
  pause(): void;
  setStepsPerSecond(sps: number): void;
  setEngine(newEngine: Solver): void;
  readonly playing: boolean;
}

/** Create the RAF loop. Renders every frame (even when paused). */
export function createLoop(
  initialEngine: Solver,
  onFrame: (info: { time: number; stepsRun: number; fps: number }) => void,
): Loop {
  let engine = initialEngine;
  let playing = false;
  let lastTimestamp = 0;
  let stepsPerSecond = 100;

  // Rolling FPS over last 30 frames
  const frameTimes: number[] = [];
  function recordFrameTime(deltaMs: number): number {
    frameTimes.push(deltaMs);
    if (frameTimes.length > 30) frameTimes.shift();
    const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
    return avg > 0 ? 1000 / avg : 0;
  }

  function tick(timestamp: DOMHighResTimeStamp): void {
    const deltaMs = timestamp - lastTimestamp;
    lastTimestamp = timestamp;
    const fps = recordFrameTime(deltaMs);

    if (playing) {
      const wallDeltaS = Math.min(deltaMs / 1000, 0.1);
      const n = Math.round(stepsPerSecond * wallDeltaS);
      if (n > 0) {
        engine.step(n);
      }
    }

    engine.render();
    onFrame({ time: engine.state.time, stepsRun: engine.state.stepsRun, fps });
    requestAnimationFrame(tick);
  }

  requestAnimationFrame((t) => {
    lastTimestamp = t;
    tick(t);
  });

  return {
    play() {
      if (!playing) {
        lastTimestamp = performance.now();
        playing = true;
      }
    },
    pause() {
      playing = false;
    },
    setStepsPerSecond(sps: number) {
      stepsPerSecond = Math.max(1, Math.round(sps));
    },
    setEngine(newEngine: Solver) {
      engine = newEngine;
    },
    get playing() { return playing; },
  };
}
