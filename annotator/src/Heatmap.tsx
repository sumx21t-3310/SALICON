import { useEffect, useRef } from "react";
import { buildSaliencyMap, extractFixations, type TrialResult } from "./lib";

/** 1 試行の顕著性マップを画像に重ねて表示する（結果の確認用） */
export function Heatmap({ trial, src, width = 300 }: { trial: TrialResult; src: string; width?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    // 大きな画像でも軽く描けるよう、確認用は幅 640 px までに縮める
    const scale = Math.min(1, 640 / trial.imageWidth);
    const w = Math.max(1, Math.round(trial.imageWidth * scale));
    const h = Math.max(1, Math.round(trial.imageHeight * scale));
    c.width = w;
    c.height = h;
    const g = c.getContext("2d")!;
    const img = new Image();
    img.onload = () => {
      g.drawImage(img, 0, 0, w, h);
      const map = buildSaliencyMap([trial], { width: w, height: h });
      const px = g.getImageData(0, 0, w, h);
      for (let i = 0; i < map.length; i++) {
        const v = map[i];
        const a = Math.min(1, v * 1.2) * 0.65;
        px.data[i * 4] = px.data[i * 4] * (1 - a) + 255 * a;
        px.data[i * 4 + 1] = px.data[i * 4 + 1] * (1 - a) + 80 * v * a;
        px.data[i * 4 + 2] = px.data[i * 4 + 2] * (1 - a);
      }
      g.putImageData(px, 0, 0);
      g.fillStyle = "rgba(0,120,255,.8)";
      for (const p of extractFixations(trial)) g.fillRect(p.x * scale - 1, p.y * scale - 1, 2, 2);
    };
    img.src = src;
  }, [trial, src]);
  return <canvas ref={ref} style={{ width, display: "block", borderRadius: 4 }} />;
}
