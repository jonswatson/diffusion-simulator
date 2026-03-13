// ============================================================
// Render shader: maps concentration field → RGBA pixels on canvas.
//
// The vertex shader draws one full-screen triangle (3 vertices, no buffer).
// The fragment shader looks up concentration C in [0,1] for the current pixel,
// then linearly interpolates between material A color (C=0) and B color (C=1).
//
// Pixel → grid: each canvas pixel maps 1:1 to one grid cell.
// Pixels outside the grid → dark grey background.
// ============================================================

struct Uniforms {
  width  : u32,
  height : u32,
  r      : f32,
  _pad   : u32,
}

struct MaterialColors {
  colorA : vec3f,  // linear RGB [0,1] for C=0 (pure material A)
  _padA  : f32,    // 16-byte alignment padding
  colorB : vec3f,  // linear RGB [0,1] for C=1 (pure material B)
  _padB  : f32,
}

@group(0) @binding(0) var<uniform>        uniforms   : Uniforms;
@group(0) @binding(1) var<storage, read>  concBuffer : array<f32>;
@group(0) @binding(2) var<uniform>        matColors  : MaterialColors;

struct VertexOut {
  @builtin(position) pos : vec4f,
}

// Full-screen triangle: covers [-1,1]x[-1,1] NDC with 3 vertices.
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

  // Pixels outside the grid → dark grey background
  if (x >= uniforms.width || y >= uniforms.height) {
    return vec4f(0.05, 0.05, 0.05, 1.0);
  }

  let c = concBuffer[y * uniforms.width + x]; // concentration in [0, 1]
  let color = mix(matColors.colorA, matColors.colorB, c);
  return vec4f(color, 1.0);

  // NOTE: Colors are linear RGB. For gamma-correct display, apply:
  //   let gamma_correct = pow(color, vec3f(1.0 / 2.2));
}
