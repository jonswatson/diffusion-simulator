# Diffusion Simulator

A 2D solid-state diffusion simulator for materials science education, built with TypeScript and WebGPU compute shaders.

Solves Fick's Second Law via explicit FTCS finite difference on the GPU in real time, with Arrhenius temperature-dependent diffusivity from published literature values.

## Requirements

- **Node.js 22+** (see `.nvmrc`)
- **Chrome 113+** or **Edge 113+** with WebGPU support

## Getting Started

```bash
npm install
npm run dev
```

Open the local URL printed by Vite (typically `http://localhost:5173`).

## Things to Try

1. **Watch diffusion spread** — Click the ▶ play button. The default step-function initial condition will smooth out over time as copper diffuses into aluminium at 600 °C.

2. **Crank up the temperature** — Drag the temperature slider to 900 K and watch the diffusion speed up dramatically (Arrhenius exponential dependence).

3. **Switch materials** — Select "C in γ-Fe" from the material dropdown. Carbon interstitial diffusion in austenite has a very different diffusivity than substitutional Cu-in-Al.

4. **Change domain size** — Pick "1 mm" from the domain size select. A larger physical domain means more real time is needed for the concentration front to traverse the grid.

5. **Adjust simulation speed** — The speed slider is logarithmic (1 to 100,000 steps/s). Crank it up to see long-time evolution, or slow it down to watch individual timesteps.

6. **Upload a custom image** — Use the "Upload grayscale image" input to load any image. It will be converted to a concentration field (dark = 0, bright = 1) and used as the initial condition.

7. **Check the validation panel** — After running for a while, the "Mass drift" readout shows how well mass is conserved (should stay well under 0.1%). "RMS vs erfc" compares the numerical midline to the analytical error-function solution.

8. **Trigger a warning** — Set the temperature below 400 K with Cu-in-Al selected. A warning appears because the Arrhenius parameters are only validated above 400 K for that system.

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server |
| `npm run build` | Type-check and build for production |
| `npm run test:unit` | Run Vitest unit tests |
| `npm run test:e2e` | Run Playwright E2E tests (requires Chrome) |
| `npm test` | Run all tests |

## Project Structure

```
src/
  engine/           Physics engine (zero DOM dependencies)
    shaders/
      diffusion.wgsl   FTCS compute shader
      render.wgsl      Concentration-to-color fragment shader
    solver.ts          GPU pipeline orchestration, ping-pong buffers
    physics.ts         Arrhenius, erfc, analytical solutions
    materials.ts       Material database (D₀, Q from literature)
    buffers.ts         GPU buffer helpers
    index.ts           Public API
  app/               Application layer (DOM, UI, loop)
    ui.ts              Control panel wiring
    loop.ts            requestAnimationFrame simulation loop
    imageLoader.ts     Image-to-concentration conversion
    validation.ts      Mass conservation and analytical checks
  main.ts            Entry point
  style.css          Styles
```
