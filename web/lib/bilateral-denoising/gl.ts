// webgl2 pipeline: source -> clean fbo -> noisy fbo -> bilateral (1-3 passes) -> screen.
// every texture stores the top image row at texel y = 0, only the screen pass flips.

const vs = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const head = `#version 300 es
precision highp float;
precision highp int;
out vec4 o;
`;

// copy the source canvas into the clean fbo (unflipped upload, so the top row lands at y = 0)
const fsCopy = `${head}
uniform sampler2D src;
uniform vec2 size;
void main() {
  vec2 uv = gl_FragCoord.xy / size;
  o = vec4(texture(src, uv).rgb, 1.0);
}`;

// gaussian noise from a hash, so the same seed gives the same noise image
const fsNoise = `${head}
uniform sampler2D src;
uniform float sig;
uniform uint seed;
uint hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
float u01(uint h) { return (float(h >> 8) + 0.5) / 16777216.0; }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 c = texelFetch(src, p, 0).rgb;
  uint k = hash(uint(p.x) * 73856093u ^ uint(p.y) * 19349663u ^ seed * 83492791u);
  vec3 n;
  for (int i = 0; i < 3; i++) {
    uint a = hash(k + uint(2 * i)), b = hash(k + uint(2 * i + 1));
    n[i] = sqrt(-2.0 * log(u01(a))) * cos(6.2831853 * u01(b));
  }
  o = vec4(clamp(c + sig * n, 0.0, 1.0), 1.0);
}`;

// brute force bilateral, circular window, reflect-101 borders, per channel range weights (like cv2 on gray)
const fsBilat = `${head}
uniform sampler2D src;
uniform int r;
uniform float ss;
uniform float sr;
int refl(int i, int n) { i = i < 0 ? -i : i; return i >= n ? 2 * n - 2 - i : i; }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 sz = textureSize(src, 0);
  vec3 c = texelFetch(src, p, 0).rgb;
  vec3 acc = vec3(0.0), wsum = vec3(0.0);
  float is = -0.5 / (ss * ss), ir = -0.5 / (sr * sr);
  for (int dy = -r; dy <= r; dy++) {
    for (int dx = -r; dx <= r; dx++) {
      int d2 = dx * dx + dy * dy;
      if (d2 > r * r) continue;
      vec3 v = texelFetch(src, ivec2(refl(p.x + dx, sz.x), refl(p.y + dy, sz.y)), 0).rgb;
      vec3 dv = v - c;
      vec3 w = exp(float(d2) * is + dv * dv * ir);
      acc += w * v;
      wsum += w;
    }
  }
  o = vec4(acc / wsum, 1.0);
}`;

// split view + optional heatmap overlay of a predicted parameter map
const fsShow = `${head}
uniform sampler2D left;
uniform sampler2D right;
uniform sampler2D heat;
uniform float split;
uniform float alpha;
uniform vec2 size;
uniform float px;
vec3 cmap(float t) {
  // viridis, polynomial fit
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.2777, 0.0054, 0.3341), c1 = vec3(0.1050, 1.4046, 1.3846), c2 = vec3(-0.3309, 0.2148, 0.0951);
  vec3 c3 = vec3(-4.6342, -5.7991, -19.3324), c4 = vec3(6.2283, 14.1799, 56.6906), c5 = vec3(4.7764, -13.7451, -65.3530);
  vec3 c6 = vec3(-5.4355, 4.6459, 26.3124);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
void main() {
  vec2 uv = vec2(gl_FragCoord.x / size.x, 1.0 - gl_FragCoord.y / size.y);
  bool l = uv.x < split;
  vec3 c = l ? texture(left, uv).rgb : texture(right, uv).rgb;
  if (!l && alpha > 0.0) c = mix(c, cmap(texture(heat, uv).r), alpha);
  if (abs(gl_FragCoord.x - split * size.x) < px) c = vec3(1.0);
  o = vec4(c, 1.0);
}`;

type tgt = { tex: WebGLTexture; fb: WebGLFramebuffer };

export type prm = { sig: number; r: number; ss: number; sr: number; passes: number };

export class bgl {
  gl: WebGL2RenderingContext;
  w = 0;
  h = 0;
  private progs: Record<string, WebGLProgram> = {};
  private src: WebGLTexture;
  private nn: WebGLTexture;
  private heat: WebGLTexture;
  private t: Record<"clean" | "noisy" | "a" | "b", tgt> | null = null;
  out: "a" | "b" = "a";
  useNN = false;

  constructor(cv: HTMLCanvasElement) {
    const gl = cv.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false });
    if (!gl) throw new Error("webgl2 not available");
    this.gl = gl;
    const srcs = { copy: fsCopy, noise: fsNoise, bilat: fsBilat, show: fsShow };
    for (const [k, fs] of Object.entries(srcs)) this.progs[k] = this.prog(vs, fs);
    this.src = this.mkTex();
    this.nn = this.mkTex();
    this.heat = this.mkTex();
    gl.bindVertexArray(gl.createVertexArray());
  }

  private prog(a: string, b: string) {
    const gl = this.gl;
    const p = gl.createProgram()!;
    for (const [type, s] of [
      [gl.VERTEX_SHADER, a],
      [gl.FRAGMENT_SHADER, b],
    ] as const) {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, s);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) ?? "shader");
      gl.attachShader(p, sh);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
    return p;
  }

  private mkTex() {
    const gl = this.gl;
    const t = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  private mkTgt(): tgt {
    const gl = this.gl;
    const tex = this.mkTex();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.w, this.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fb };
  }

  resize(w: number, h: number) {
    if (w === this.w && h === this.h && this.t) return;
    const gl = this.gl;
    for (const v of Object.values(this.t ?? {})) {
      gl.deleteTexture(v.tex);
      gl.deleteFramebuffer(v.fb);
    }
    this.w = w;
    this.h = h;
    this.t = { clean: this.mkTgt(), noisy: this.mkTgt(), a: this.mkTgt(), b: this.mkTgt() };
    this.useNN = false;
  }

  // source is drawn into a canvas of the working size first, so this is always top row first
  upload(el: TexImageSource) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, el);
  }

  private pass(name: string, to: tgt | null, tex: Record<string, WebGLTexture>, u: Record<string, number>) {
    const gl = this.gl;
    const p = this.progs[name];
    gl.useProgram(p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, to ? to.fb : null);
    const [vw, vh] = to ? [this.w, this.h] : [gl.drawingBufferWidth, gl.drawingBufferHeight];
    gl.viewport(0, 0, vw, vh);
    let i = 0;
    for (const [k, t] of Object.entries(tex)) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.uniform1i(gl.getUniformLocation(p, k), i++);
    }
    for (const [k, v] of Object.entries(u)) {
      const loc = gl.getUniformLocation(p, k);
      if (k === "r") gl.uniform1i(loc, v);
      else if (k === "seed") gl.uniform1ui(loc, v);
      else gl.uniform1f(loc, v);
    }
    const sz = gl.getUniformLocation(p, "size");
    if (sz) gl.uniform2f(sz, vw, vh);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  prep(sig: number, seed: number) {
    const t = this.t!;
    this.pass("copy", t.clean, { src: this.src }, {});
    this.pass("noise", t.noisy, { src: t.clean.tex }, { sig, seed });
  }

  filter(q: prm) {
    const t = this.t!;
    let from = t.noisy;
    let to: "a" | "b" = "a";
    for (let i = 0; i < q.passes; i++) {
      this.pass("bilat", t[to], { src: from.tex }, { r: q.r, ss: q.ss, sr: q.sr });
      from = t[to];
      this.out = to;
      to = to === "a" ? "b" : "a";
    }
  }

  // neural result (rgba8, top row first) replaces the filter output on screen
  setNN(px: Uint8Array | null, heat: Uint8Array | null) {
    const gl = this.gl;
    this.useNN = !!px;
    if (px) {
      gl.bindTexture(gl.TEXTURE_2D, this.nn);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.w, this.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
    }
    if (heat) {
      gl.bindTexture(gl.TEXTURE_2D, this.heat);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, this.w, this.h, 0, gl.RED, gl.UNSIGNED_BYTE, heat);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    }
  }

  // raw: show the noisy image on the right too (neural result not there yet)
  show(split: number, alpha: number, showClean: boolean, raw = false) {
    const t = this.t!;
    const right = raw ? t.noisy.tex : this.useNN ? this.nn : t[this.out].tex;
    const px = Math.max(1, window.devicePixelRatio || 1);
    this.pass("show", null, { left: showClean ? t.clean.tex : t.noisy.tex, right, heat: this.heat }, { split, alpha, px });
  }

  read(which: "clean" | "noisy" | "out") {
    const gl = this.gl;
    const t = this.t!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, (which === "out" ? t[this.out] : t[which]).fb);
    const px = new Uint8Array(this.w * this.h * 4);
    gl.readPixels(0, 0, this.w, this.h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return px;
  }
}
