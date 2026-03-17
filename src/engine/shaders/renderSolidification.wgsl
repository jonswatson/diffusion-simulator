// ============================================================
// Binary Solidification render shader.
//
// Displays the solid fraction φ and composition c.
//
// Left half of color blend: φ drives liquid→solid interpolation.
// Tint by composition: c above midpoint cl_eq shifts toward solute color.
//
// Simple visualization strategy:
//   base_color = mix(colorLiquid, colorSolid, smoothstep(0.1, 0.9, phi))
//   composition tint applied as a subtle hue shift (c relative to mean)
//
// For clarity in educational use, just blend by φ — solidification front
// is the most important feature to see.
// ============================================================

struct Uniforms {
  width    : u32,
  height   : u32,
  dt       : f32,
  _pad0    : f32,
  kappa_phi: f32,
  w        : f32,
  Lphi     : f32,
  As       : f32,
  Al       : f32,
  cs_eq    : f32,
  cl_eq    : f32,
  Ms       : f32,
  Ml       : f32,
  _pad1    : f32,
  _pad2    : f32,
  _pad3    : f32,
}

struct MaterialColors {
  colorA : vec3f,  // liquid phase color (φ=0)
  _padA  : f32,
  colorB : vec3f,  // solid phase color  (φ=1)
  _padB  : f32,
}

@group(0) @binding(0) var<uniform>        uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read>  phi_buf   : array<f32>;
@group(0) @binding(2) var<storage, read>  c_buf     : array<f32>;
@group(0) @binding(3) var<uniform>        matColors : MaterialColors;

struct VertexOut {
  @builtin(position) pos : vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vIdx: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  var out: VertexOut;
  out.pos = vec4f(positions[vIdx], 0.0, 1.0);
  return out;
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = u32(pos.x);
  let y = u32(pos.y);

  if (x >= uniforms.width || y >= uniforms.height) {
    return vec4f(0.05, 0.05, 0.05, 1.0);
  }

  let pix = y * uniforms.width + x;
  let phi = clamp(phi_buf[pix], 0.0, 1.0);
  let c   = clamp(c_buf[pix],   0.0, 1.0);

  // Phase field → liquid/solid blend
  let t_phase = smoothstep(0.1, 0.9, phi);
  var color = mix(matColors.colorA, matColors.colorB, t_phase);

  // Composition overlay: shift toward a slightly warmer tone inside the solid.
  // c_mid = midpoint between cs_eq and cl_eq; deviation from c_mid tints toward
  // a yellow hue to show solute redistribution.
  let c_mid = 0.5 * (uniforms.cs_eq + uniforms.cl_eq);
  let c_dev = clamp((c - c_mid) / (uniforms.cl_eq - uniforms.cs_eq + 1e-6), -0.5, 0.5);
  // Positive c_dev (solute-rich, liquid side) → orange-yellow tint
  let solute_tint = vec3f(0.4, 0.15, -0.2) * c_dev;
  color = clamp(color + 0.3 * t_phase * solute_tint + 0.3 * (1.0 - t_phase) * solute_tint,
                vec3f(0.0), vec3f(1.0));

  return vec4f(color, 1.0);
}
