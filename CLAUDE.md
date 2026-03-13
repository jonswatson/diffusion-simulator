# Diffusion Simulator — Project Guidelines

## Project Purpose
2D solid-state diffusion simulator for materials science education.
Stack: TypeScript + Vite + WebGPU. MVP implements Fick's 2nd law via FTCS finite difference on the GPU.

---

## Requirements

1. **Self-contained physics engine** — `src/engine/` has zero imports from `src/app/` or any DOM API. It speaks only `GPUDevice`, `Float32Array`, and plain TypeScript interfaces.
2. **Frontend only** — no server, no backend, no build-time data fetching.
3. **WebGPU compute shaders** — diffusion loop runs on the GPU. No CPU fallback needed for MVP.
4. **Physically accurate** — Arrhenius D(T) with real D₀/Q values from published literature.
5. **Readable code** — a materials scientist (not a JS expert) should open any file and understand it.
6. **Simplicity over robustness** — no overengineering, no premature abstraction, no defensive code for cases that can't happen.

---

## Code Guidelines

### Naming
- GPU resources: prefixed `gpu` or suffixed with resource type — `uniformBuffer`, `computePipeline`, `diffusionPass`. Never ambiguous whether a variable is CPU or GPU memory.
- Physical quantities: unit in variable name or adjacent comment — `domainSize_m`, `temperature_K`, `dt // seconds/step`, `D0 // m²/s`.
- `Q` (activation energy) stored internally in **J/mol** always. kJ/mol conversion happens only at the UI display boundary.

### Comments
- Every function in `engine/` has a one-line JSDoc explaining the **physics rationale**, not just what the code does.
- WGSL shaders open with a comment block containing: the PDE being solved, its discretization, units of key variables, and the stability condition.

### TypeScript
- `"strict": true` — no `any`, no unchecked index access.
- No `!` non-null assertions except on canvas context (`.getContext('2d')!` where guaranteed).
- `engine/index.ts` lists all exports explicitly — no `export * from`.

### Architecture
- `engine/` public surface: `createDiffusionEngine(device, config)` returns a `DiffusionEngine` instance. The `app/` layer calls only methods on that interface.
- Grid size is a **physics parameter** set by the user — never driven by uploaded image resolution.
- Ping-pong via **bind group swap** — never `copyBuffer` between concentration fields.
- Compute + render passes in **one `GPUCommandEncoder`**, one `device.queue.submit()` per frame.
- Physical time accumulated in **TypeScript `number` (f64)** — `state.time += n * state.dt`. Never read back from GPU to track time.
- `step(n)` loops all N dispatch calls inside one compute pass, submitted once. No per-step submits.

---

## Physics Reference

### Fick's Second Law (2D)
```
∂C/∂t = D∇²C = D(∂²C/∂x² + ∂²C/∂y²)
```
C = mole fraction [0, 1]. D = interdiffusion coefficient [m²/s].

### FTCS Discretization
```
C[i,j]^(n+1) = C[i,j]^n + r * (C[i+1,j] + C[i-1,j] + C[i,j+1] + C[i,j-1] - 4*C[i,j])
```
r = D·dt/dx² (Fourier number). Stability requires r ≤ 0.25 in 2D.
Safety factor 0.4 is applied: `dt = 0.4 * dx² / (4D)` → r ≈ 0.1.

### Arrhenius Temperature Dependence
```
D(T) = D₀ · exp(-Q / RT)
```
R = 8.314 J/(mol·K). T in Kelvin. D₀ in m²/s. Q in J/mol.

### Boundary Conditions
Zero-flux Neumann: `∂C/∂n = 0` at all walls.
Implemented by clamping neighbor indices at grid edges (mirror ghost cell).

---

## Materials Database (literature values)

| System         | D₀ (m²/s)  | Q (kJ/mol) | Valid range    | Source              |
|----------------|------------|------------|----------------|---------------------|
| Cu in Al       | 6.47×10⁻⁵  | 136.8      | 400–900 K      | Smithells, Table 13 |
| Al self-diff   | 1.71×10⁻⁴  | 142.2      | 700–933 K      | Smithells, Table 13 |
| C in γ-Fe      | 2.30×10⁻⁵  | 148.0      | 1000–1400 K    | Shewmon, p. 64      |
| Ni in Cu       | 1.93×10⁻⁴  | 230.0      | 900–1300 K     | Smithells, Table 13 |

---

## Extension Path (document, don't implement in MVP)

A `CahnHilliardEngine` implementing the same `DiffusionEngine` interface is the phase field extension.
Swap it in `main.ts` — the `app/` layer is unchanged.
Requires: semi-implicit solver (split-Laplacian or spectral FFT), thermodynamic free energy module,
and much finer grids (dx must resolve the interface width ξ, typically nm-scale).

---

## Key Pitfalls

| Pitfall | Impact | Fix |
|---|---|---|
| WGSL struct misalignment | Silent wrong values | `vec3f` always needs trailing `f32` pad; check 16-byte alignment |
| Ping-pong bind group slot mismatch | Pipeline creation error | Both `AtoB` and `BtoA` must bind uniforms at the same slot index |
| Reusing `GPUCommandEncoder` | Runtime error | Create a new encoder every frame; never reuse after `finish()` |
| `writeBuffer` async race | Corrupted simulation | Always `writeBuffer` before `submit` in same synchronous call stack |
| Upload race condition | Reads partially-written buffer | `loop.pause()` before `engine.loadField()`, then resume |
| f32 time accumulation | Precision loss over long runs | Use TypeScript f64 for `state.time`; f32 only in GPU uniforms |
| iGPU on dual-GPU laptop | Too slow at 512×512 | `requestAdapter({ powerPreference: 'high-performance' })` |

---

## Stage Implementation Order

```
0 (scaffold) → 1 (GPU bootstrap) → 2 (diffusion shader) → 3 (physics/materials)
  → 4 (image input) ─┬─ 5 (rendering) → 6 (loop) → 7 (UI) → 8 (validation)
                      └─ 8 physics helpers (erfc, mass sum) can start here
```

See `.claude/plans/stages/` for per-stage spec files.
