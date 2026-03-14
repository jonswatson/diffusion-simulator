// ============================================================
// Render shader for multi-order-parameter grain growth.
//
// Reads packed eta buffer [N × W × H], finds the dominant grain
// (argmax η_i) at each pixel, and looks up its color from a
// storage buffer color table.
//
// Grain boundaries (where max η < threshold) are rendered as
// dark lines using smoothstep for anti-aliasing.
// ============================================================

struct Uniforms {
  width   : u32,
  height  : u32,
  dt      : f32,
  dx      : f32,
  L       : f32,
  kappa   : f32,
  A       : f32,
  B       : f32,
  N       : u32,
  _pad1   : u32,
  _pad2   : u32,
  _pad3   : u32,
}

@group(0) @binding(0) var<uniform>        uniforms   : Uniforms;
@group(0) @binding(1) var<storage, read>  etaBuffer  : array<f32>;
@group(0) @binding(2) var<storage, read>  colorTable : array<vec4f>;

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

  if (x >= uniforms.width || y >= uniforms.height) {
    return vec4f(0.05, 0.05, 0.05, 1.0);
  }

  let WH = uniforms.width * uniforms.height;

  // Find dominant grain (argmax η_i)
  var maxEta = -1.0;
  var maxIdx = 0u;
  for (var i = 0u; i < uniforms.N; i = i + 1u) {
    let eta = etaBuffer[i * WH + y * uniforms.width + x];
    if (eta > maxEta) {
      maxEta = eta;
      maxIdx = i;
    }
  }

  // Look up grain color from table
  let grainColor = colorTable[maxIdx].rgb;

  // Darken at grain boundaries where the dominant η is weak
  let boundary = smoothstep(0.3, 0.7, maxEta);
  let color = grainColor * boundary;

  return vec4f(color, 1.0);
}
