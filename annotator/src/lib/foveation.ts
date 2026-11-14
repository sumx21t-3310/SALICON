import type { FoveationParams } from "./types";

/**
 * SALICON 論文の多重解像度表示を WebGL2 で描画する。
 *
 * - 6 段のガウシアンピラミッド（各段で 1/2 に縮小）を GPU で作る。
 * - 画素ごとに注視点（マウス位置）からの視角 θ を求め、解像度
 *   R = α / (α + θ) に対応する段 L = log2(1 / R) = log2(1 + θ / α) を選ぶ。
 * - 隣り合う段を線形に混ぜ、元の大きさに補間して表示する。
 *
 * 論文は Geisler & Perry の伝達関数で混合行列を作っている。ここでは同じ
 * 解像度関数 R を使い、段の間を線形補間する近似にしている。
 */

const VERT = `#version 300 es
in vec2 a_pos;
uniform float u_flipY;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, mix(a_pos.y * 0.5 + 0.5, 0.5 - a_pos.y * 0.5, u_flipY));
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// 5x5 の二項フィルター（Burt & Adelson）で平滑化しながら 1/2 に縮小する
const DOWN_FRAG = `#version 300 es
precision highp float;
uniform sampler2D u_src;
uniform vec2 u_srcTexel;
in vec2 v_uv;
out vec4 o;
void main() {
  float k[5] = float[5](1.0, 4.0, 6.0, 4.0, 1.0);
  vec4 acc = vec4(0.0);
  for (int j = 0; j < 5; j++) {
    for (int i = 0; i < 5; i++) {
      vec2 off = vec2(float(i - 2), float(j - 2)) * u_srcTexel;
      acc += texture(u_src, v_uv + off) * k[i] * k[j];
    }
  }
  o = acc / 256.0;
}`;

const MAX_LEVELS = 6;

const COMPOSE_FRAG = `#version 300 es
precision highp float;
uniform sampler2D u_l0;
uniform sampler2D u_l1;
uniform sampler2D u_l2;
uniform sampler2D u_l3;
uniform sampler2D u_l4;
uniform sampler2D u_l5;
uniform vec2 u_imgSize;
uniform vec2 u_gaze;
uniform float u_ppd;
uniform float u_alpha;
uniform float u_maxLevel;
uniform float u_cursorR;
uniform float u_ringHalf;
uniform float u_aa;
in vec2 v_uv;
out vec4 o;
float w(float L, float i) { return max(0.0, 1.0 - abs(L - i)); }
void main() {
  vec2 p = v_uv * u_imgSize;
  float dist = distance(p, u_gaze);
  float theta = dist / u_ppd;
  float L = clamp(log2(1.0 + theta / u_alpha), 0.0, u_maxLevel);
  vec4 c = texture(u_l0, v_uv) * w(L, 0.0)
         + texture(u_l1, v_uv) * w(L, 1.0)
         + texture(u_l2, v_uv) * w(L, 2.0)
         + texture(u_l3, v_uv) * w(L, 3.0)
         + texture(u_l4, v_uv) * w(L, 4.0)
         + texture(u_l5, v_uv) * w(L, 5.0);
  if (u_cursorR > 0.0) {
    float d = abs(dist - u_cursorR);
    float ring = 1.0 - smoothstep(u_ringHalf, u_ringHalf + u_aa, d);
    c.rgb = mix(c.rgb, vec3(1.0, 0.0, 0.0), ring);
  }
  o = vec4(c.rgb, 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(`シェーダーのコンパイルに失敗しました: ${gl.getShaderInfoLog(s)}`);
  }
  return s;
}

function program(gl: WebGL2RenderingContext, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, "a_pos");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`シェーダーのリンクに失敗しました: ${gl.getProgramInfoLog(p)}`);
  }
  return p;
}

function makeTexture(gl: WebGL2RenderingContext, w: number, h: number, source?: TexImageSource) {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  if (source) {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, source);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  }
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  return t;
}

/** ミップマップを作り、表示が元画像より小さいときのエイリアシングを防ぐ */
function enableMipmaps(gl: WebGL2RenderingContext, t: WebGLTexture) {
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
}

export class FoveatedRenderer {
  private gl: WebGL2RenderingContext;
  private compose: WebGLProgram;
  private levels: WebGLTexture[] = [];
  private vao: WebGLVertexArrayObject;
  private uniforms: Record<string, WebGLUniformLocation | null> = {};
  readonly width: number;
  readonly height: number;

  constructor(
    private canvas: HTMLCanvasElement,
    image: HTMLImageElement,
    private params: FoveationParams,
  ) {
    const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error("このブラウザーは WebGL2 に対応していません。");
    this.gl = gl;
    this.width = image.naturalWidth;
    this.height = image.naturalHeight;

    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    this.buildPyramid(image);

    this.compose = program(gl, COMPOSE_FRAG);
    gl.useProgram(this.compose);
    for (const name of [
      "u_imgSize", "u_gaze", "u_ppd", "u_alpha", "u_maxLevel", "u_cursorR", "u_ringHalf", "u_aa", "u_flipY",
    ]) {
      this.uniforms[name] = gl.getUniformLocation(this.compose, name);
    }
    for (let i = 0; i < MAX_LEVELS; i++) {
      gl.uniform1i(gl.getUniformLocation(this.compose, `u_l${i}`), i);
    }
  }

  private buildPyramid(image: HTMLImageElement) {
    const gl = this.gl;
    const n = Math.max(1, Math.min(MAX_LEVELS, Math.round(this.params.levels)));
    const down = program(gl, DOWN_FRAG);
    const fbo = gl.createFramebuffer();
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);

    let w = this.width;
    let h = this.height;
    const base = makeTexture(gl, w, h, image);
    this.levels.push(base);

    gl.useProgram(down);
    gl.uniform1f(gl.getUniformLocation(down, "u_flipY"), 0);
    gl.uniform1i(gl.getUniformLocation(down, "u_src"), 0);
    const texelLoc = gl.getUniformLocation(down, "u_srcTexel");
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    for (let i = 1; i < n; i++) {
      const src = this.levels[i - 1];
      const nw = Math.max(1, Math.floor(w / 2));
      const nh = Math.max(1, Math.floor(h / 2));
      const dst = makeTexture(gl, nw, nh);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, dst, 0);
      gl.viewport(0, 0, nw, nh);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, src);
      gl.uniform2f(texelLoc, 1 / w, 1 / h);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      this.levels.push(dst);
      w = nw;
      h = nh;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteProgram(down);
    // 段数が 6 未満のときは最後の段で埋める（重みは 0 になる）
    while (this.levels.length < MAX_LEVELS) this.levels.push(this.levels[this.levels.length - 1]);
    for (const t of new Set(this.levels)) enableMipmaps(gl, t);
    this.maxLevel = n - 1;
  }

  private maxLevel = MAX_LEVELS - 1;

  /** 注視点（画像の元解像度の画素）を指定して描画する */
  render(gazeX: number, gazeY: number) {
    const gl = this.gl;
    const { canvas, params, uniforms: u } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(this.compose);
    gl.bindVertexArray(this.vao);
    for (let i = 0; i < MAX_LEVELS; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, this.levels[i]);
    }
    // 表示 1 画素が元画像の何画素に当たるか（線の太さとアンチエイリアス用）
    const imgPxPerScreenPx = this.width / Math.max(1, canvas.width);
    gl.uniform1f(u.u_flipY, 1);
    gl.uniform2f(u.u_imgSize, this.width, this.height);
    gl.uniform2f(u.u_gaze, gazeX, gazeY);
    gl.uniform1f(u.u_ppd, params.pixelsPerDegree);
    gl.uniform1f(u.u_alpha, params.halfResolutionDeg);
    gl.uniform1f(u.u_maxLevel, this.maxLevel);
    gl.uniform1f(u.u_cursorR, params.cursorRadiusDeg * params.pixelsPerDegree);
    gl.uniform1f(u.u_ringHalf, 1.0 * imgPxPerScreenPx * (window.devicePixelRatio || 1));
    gl.uniform1f(u.u_aa, imgPxPerScreenPx);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  dispose() {
    const gl = this.gl;
    for (const t of new Set(this.levels)) gl.deleteTexture(t);
    gl.deleteProgram(this.compose);
    gl.deleteVertexArray(this.vao);
  }
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`画像を読み込めませんでした: ${src}`));
    img.src = src;
  });
}
