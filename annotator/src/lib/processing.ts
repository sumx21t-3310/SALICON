/**
 * 記録したマウス軌跡から注視点と顕著性マップを作る処理。
 * SALICON 論文の手順に合わせている。
 *
 * 1. 12 Hz 未満の試行を破棄する
 * 2. 形状保存区分 3 次補間（PCHIP）で 100 Hz に再標本化する
 * 3. 被験者ごとに速度の大きい半分のサンプル（サッカード）を除く
 * 4. 残ったサンプルを集計し、ガウシアンフィルターでぼかす
 *
 * 論文はマップ生成時のガウシアンの σ を明記していないため、既定値は 1 度
 * （pixelsPerDegree 画素）とした。必要に応じて変更すること。
 */
import type { GazeSample, TrialResult } from "./types";

export interface ProcessOptions {
  /** 再標本化の周波数（Hz）。論文は 100。 */
  resampleHz?: number;
  /** 残すサンプルの割合（速度の小さい順）。論文は 0.5。 */
  keepRatio?: number;
  /** この周波数未満の試行を破棄する。論文は 12。 */
  minSamplingHz?: number;
}

/** PCHIP（Fritsch–Carlson）の傾きを求める */
function pchipSlopes(x: number[], y: number[]): number[] {
  const n = x.length;
  const d = new Array<number>(n).fill(0);
  if (n < 2) return d;
  const h: number[] = [];
  const delta: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    h.push(x[i + 1] - x[i]);
    delta.push((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
  }
  if (n === 2) return [delta[0], delta[0]];
  for (let i = 1; i < n - 1; i++) {
    if (delta[i - 1] * delta[i] <= 0) {
      d[i] = 0;
    } else {
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
    }
  }
  const end = (h0: number, h1: number, d0: number, d1: number) => {
    let v = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (Math.sign(v) !== Math.sign(d0)) v = 0;
    else if (Math.sign(d0) !== Math.sign(d1) && Math.abs(v) > Math.abs(3 * d0)) v = 3 * d0;
    return v;
  };
  d[0] = end(h[0], h[1], delta[0], delta[1]);
  d[n - 1] = end(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
  return d;
}

function pchipEval(x: number[], y: number[], d: number[], xi: number[]): number[] {
  const out: number[] = [];
  let k = 0;
  for (const t of xi) {
    while (k < x.length - 2 && t > x[k + 1]) k++;
    const h = x[k + 1] - x[k];
    const s = (t - x[k]) / h;
    const h00 = (1 + 2 * s) * (1 - s) * (1 - s);
    const h10 = s * (1 - s) * (1 - s);
    const h01 = s * s * (3 - 2 * s);
    const h11 = s * s * (s - 1);
    out.push(h00 * y[k] + h10 * h * d[k] + h01 * y[k + 1] + h11 * h * d[k + 1]);
  }
  return out;
}

/** 時刻の重複を除き、時刻順に並べる */
function dedupe(samples: GazeSample[]): GazeSample[] {
  const sorted = samples.slice().sort((a, b) => a.t - b.t);
  const out: GazeSample[] = [];
  for (const s of sorted) {
    if (out.length && s.t - out[out.length - 1].t < 1e-6) out[out.length - 1] = s;
    else out.push(s);
  }
  return out;
}

/** PCHIP で一定周波数に再標本化する */
export function resample(samples: GazeSample[], hz = 100): GazeSample[] {
  const s = dedupe(samples);
  if (s.length < 2) return s.slice();
  const t = s.map((p) => p.t);
  const xs = s.map((p) => p.x);
  const ys = s.map((p) => p.y);
  const step = 1000 / hz;
  const ti: number[] = [];
  const n = Math.floor((t[t.length - 1] - t[0]) / step + 1e-9) + 1;
  for (let i = 0; i < n; i++) ti.push(t[0] + i * step);
  const xi = pchipEval(t, xs, pchipSlopes(t, xs), ti);
  const yi = pchipEval(t, ys, pchipSlopes(t, ys), ti);
  return ti.map((v, i) => ({ t: v, x: xi[i], y: yi[i] }));
}

/** 速度の大きいサンプルを除き、速度の小さい keepRatio の割合だけ残す */
export function removeSaccades(samples: GazeSample[], keepRatio = 0.5): GazeSample[] {
  if (samples.length < 3) return samples.slice();
  const speed = samples.map((_p, i) => {
    const a = samples[Math.max(0, i - 1)];
    const b = samples[Math.min(samples.length - 1, i + 1)];
    const dt = b.t - a.t || 1;
    return Math.hypot(b.x - a.x, b.y - a.y) / dt;
  });
  const idx = speed.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const keep = new Set(idx.slice(0, Math.round(samples.length * keepRatio)).map(([, i]) => i));
  return samples.filter((_, i) => keep.has(i));
}

/** 論文の基準で試行が有効か判定する */
export function isValidTrial(trial: TrialResult, minSamplingHz = 12): boolean {
  return trial.completed && trial.samples.length > 0 && trial.quality.samplingRateHz >= minSamplingHz;
}

/** 1 試行から注視点（サッカードを除いたサンプル）を得る */
export function extractFixations(trial: TrialResult, opts: ProcessOptions = {}): GazeSample[] {
  const { resampleHz = 100, keepRatio = 0.5 } = opts;
  const pts = removeSaccades(resample(trial.samples, resampleHz), keepRatio);
  return pts.filter((p) => p.x >= 0 && p.y >= 0 && p.x < trial.imageWidth && p.y < trial.imageHeight);
}

export interface MapOptions extends ProcessOptions {
  width: number;
  height: number;
  /** ガウシアンの σ（画素）。省略時は 1 度相当。 */
  sigmaPx?: number;
}

/** 注視点の回数マップ（行優先、width × height） */
export function buildFixationCounts(trials: TrialResult[], opts: MapOptions): Float32Array {
  const { width, height, minSamplingHz = 12 } = opts;
  const map = new Float32Array(width * height);
  for (const tr of trials) {
    if (!isValidTrial(tr, minSamplingHz)) continue;
    const sx = width / tr.imageWidth;
    const sy = height / tr.imageHeight;
    for (const p of extractFixations(tr, opts)) {
      const x = Math.min(width - 1, Math.floor(p.x * sx));
      const y = Math.min(height - 1, Math.floor(p.y * sy));
      map[y * width + x] += 1;
    }
  }
  return map;
}

function gaussianKernel(sigma: number): Float32Array {
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + r] = v;
    sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}

/** 分離可能なガウシアンフィルター（境界は 0 埋め） */
export function gaussianBlur(src: Float32Array, width: number, height: number, sigma: number): Float32Array {
  if (sigma <= 0) return src.slice();
  const k = gaussianKernel(sigma);
  const r = (k.length - 1) / 2;
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const xx = x + i;
        if (xx >= 0 && xx < width) acc += src[row + xx] * k[i + r];
      }
      tmp[row + x] = acc;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        const yy = y + i;
        if (yy >= 0 && yy < height) acc += tmp[yy * width + x] * k[i + r];
      }
      out[y * width + x] = acc;
    }
  }
  return out;
}

/** 顕著性マップ（最大値 1 に正規化、行優先） */
export function buildSaliencyMap(trials: TrialResult[], opts: MapOptions): Float32Array {
  const counts = buildFixationCounts(trials, opts);
  const ref = trials[0];
  const ppd = ref ? (ref.foveation.pixelsPerDegree * opts.width) / ref.imageWidth : 7.5;
  const sigma = opts.sigmaPx ?? ppd;
  const map = gaussianBlur(counts, opts.width, opts.height, sigma);
  let max = 0;
  for (const v of map) if (v > max) max = v;
  if (max > 0) for (let i = 0; i < map.length; i++) map[i] /= max;
  return map;
}
