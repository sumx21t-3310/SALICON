import { describe, expect, it } from "vitest";
import { buildSaliencyMap, extractFixations, gazeTrack, removeSaccades, resample } from "./processing";
import { DEFAULT_FOVEATION, type TrialResult } from "./types";

const trial = (samples: { t: number; x: number; y: number }[]): TrialResult => ({
  imageId: "a", imageSrc: "a.png", imageWidth: 64, imageHeight: 48, displayWidth: 64, displayHeight: 48,
  durationMs: 5000, startedAt: "", completed: true, startGaze: null, samples,
  quality: { samplingRateHz: (samples.length / 5000) * 1000, samplingRateOk: true, outsideMs: 0, lostFocus: false, noMovement: false },
  foveation: DEFAULT_FOVEATION, pointerType: "mouse", devicePixelRatio: 1, userAgent: "",
});

describe("resample", () => {
  it("一定間隔にそろえ、端点を通る", () => {
    const r = resample([{ t: 0, x: 0, y: 0 }, { t: 33, x: 10, y: 5 }, { t: 70, x: 20, y: 5 }], 100);
    expect(r.length).toBe(8);
    expect(r[1].t - r[0].t).toBeCloseTo(10);
    expect(r[0].x).toBeCloseTo(0);
    // 単調なデータで行き過ぎない（形状保存）
    for (const p of r) expect(p.y).toBeLessThanOrEqual(5 + 1e-9);
  });
});

describe("removeSaccades", () => {
  it("速度の大きい半分を除く", () => {
    const s = [0, 1, 2, 3, 50, 100, 101, 102].map((x, i) => ({ t: i * 10, x, y: 0 }));
    const kept = removeSaccades(s, 0.5);
    expect(kept.length).toBe(4);
    expect(kept.some((p) => p.x === 50)).toBe(false);
  });
});

describe("buildSaliencyMap", () => {
  it("注視が集まった位置が最大になる", () => {
    const samples = Array.from({ length: 300 }, (_, i) => ({ t: i * 16, x: 40 + Math.sin(i) * 0.5, y: 10 }));
    const map = buildSaliencyMap([trial(samples)], { width: 64, height: 48 });
    let arg = 0;
    map.forEach((v, i) => { if (v > map[arg]) arg = i; });
    expect(arg % 64).toBeGreaterThanOrEqual(39);
    expect(arg % 64).toBeLessThanOrEqual(41);
    expect(Math.floor(arg / 64)).toBe(10);
  });
  it("12 Hz 未満の試行は無視する", () => {
    const t = trial([{ t: 0, x: 1, y: 1 }, { t: 4000, x: 2, y: 2 }]);
    const map = buildSaliencyMap([t], { width: 64, height: 48 });
    expect(Math.max(...map)).toBe(0);
  });
});

describe("gazeTrack", () => {
  it("最後の移動から提示終了まで静止位置を保持する", () => {
    // 最初の 990 ms だけ動き、その後は静止
    const samples = Array.from({ length: 100 }, (_, i) => ({ t: i * 10, x: 10 + i * 0.2, y: 20 }));
    const t = trial(samples);
    const track = gazeTrack(t);
    expect(track[track.length - 1]).toEqual({ t: 5000, x: samples[99].x, y: 20 });
    const fix = extractFixations(t);
    // 静止区間（約 4 秒）の注視が残る
    expect(fix.filter((p) => p.t > 1000).length).toBeGreaterThan(240);
  });
  it("開始位置を最初の移動まで保持する", () => {
    const t = { ...trial([{ t: 1500, x: 33, y: 24 }, { t: 1510, x: 34, y: 24 }]), startGaze: { x: 32, y: 24 } };
    const track = gazeTrack(t);
    expect(track[0]).toEqual({ t: 0, x: 32, y: 24 });
    expect(track[1]).toEqual({ t: 1490, x: 32, y: 24 });
  });
  it("中断した試行は終了時刻まで延ばさない", () => {
    const t = { ...trial([{ t: 0, x: 1, y: 1 }, { t: 100, x: 2, y: 2 }]), completed: false };
    const track = gazeTrack(t);
    expect(track[track.length - 1].t).toBe(100);
  });
});
