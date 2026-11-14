import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { FoveatedRenderer, loadImage } from "./foveation";
import {
  DEFAULT_FOVEATION,
  type FoveationParams,
  type GazeSample,
  type TrialResult,
} from "./types";

export interface SaliconAnnotatorProps {
  /** 画像の URL。別オリジンの場合は CORS ヘッダーが必要。 */
  src: string;
  /** 結果に記録する画像 ID。省略時は src。 */
  imageId?: string;
  /** 提示時間（ms）。論文は 5000。 */
  durationMs?: number;
  /** フォビエーションのパラメーター。省略した項目は論文の値。 */
  foveation?: Partial<FoveationParams>;
  /**
   * 表示幅（CSS px）。省略時は親要素の幅に合わせ、縦横比は画像に合わせる。
   * 実験条件をそろえるため、固定値の指定を推奨する。
   */
  width?: number;
  /**
   * true の場合、中央の点をクリックしてから提示を始める。
   * 論文と同じく、注視点が画像中央から始まるようにするための手順。
   */
  requireCenterStart?: boolean;
  /** true の間だけ提示を始める。false にすると待機状態に戻る。 */
  active?: boolean;
  /** 提示が終わったとき（または中断されたとき）に呼ばれる */
  onComplete?: (result: TrialResult) => void;
  /** 画像の読み込みや WebGL の初期化に失敗したとき */
  onError?: (error: Error) => void;
  className?: string;
  style?: CSSProperties;
  /** 開始前に表示する文言 */
  startLabel?: string;
}

type Phase = "loading" | "waiting" | "viewing" | "done" | "error";

const MIN_SAMPLING_HZ = 12;

/**
 * SALICON 方式のマウス追跡アノテーションを 1 枚の画像で行うコンポーネント。
 *
 * 画像全体をぼかして表示し、マウス位置の周辺だけを鮮明に描画する。
 * 提示中のマウス位置を画像の元解像度の座標で記録し、終了時に onComplete で返す。
 */
export function SaliconAnnotator({
  src,
  imageId,
  durationMs = 5000,
  foveation,
  width,
  requireCenterStart = true,
  active = true,
  onComplete,
  onError,
  className,
  style,
  startLabel = "中央の点をクリックすると開始します",
}: SaliconAnnotatorProps) {
  const params: FoveationParams = { ...DEFAULT_FOVEATION, ...foveation };
  const paramsKey = JSON.stringify(params);

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FoveatedRenderer | null>(null);
  const [phase, setPhase] = useState<Phase>("loading");
  const [error, setError] = useState<string | null>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [displayW, setDisplayW] = useState<number>(width ?? 0);

  // 計測中の状態は再描画を起こさないよう ref に持つ
  const rec = useRef({
    samples: [] as GazeSample[],
    gaze: { x: 0, y: 0 },
    onset: 0,
    startedAt: "",
    outsideSince: null as number | null,
    outsideMs: 0,
    lostFocus: false,
    pointerType: "mouse",
    timer: 0,
    raf: 0,
    finished: false,
  });

  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  // 表示幅
  useEffect(() => {
    if (width) {
      setDisplayW(width);
      return;
    }
    const el = wrapRef.current?.parentElement;
    if (!el) return;
    const ro = new ResizeObserver(() => setDisplayW(el.clientWidth));
    ro.observe(el);
    setDisplayW(el.clientWidth);
    return () => ro.disconnect();
  }, [width]);

  // 画像の読み込みとレンダラーの初期化
  useEffect(() => {
    let cancelled = false;
    setPhase("loading");
    setError(null);
    loadImage(src)
      .then((img) => {
        if (cancelled || !canvasRef.current) return;
        setNatural({ w: img.naturalWidth, h: img.naturalHeight });
        rendererRef.current?.dispose();
        rendererRef.current = new FoveatedRenderer(canvasRef.current, img, params);
        rec.current.gaze = { x: img.naturalWidth / 2, y: img.naturalHeight / 2 };
        setPhase("waiting");
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setError(e.message);
        setPhase("error");
        onErrorRef.current?.(e);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, paramsKey]);

  useEffect(() => () => rendererRef.current?.dispose(), []);

  const displayH = natural && displayW ? Math.round((displayW * natural.h) / natural.w) : 0;

  // キャンバスの解像度を表示サイズ × devicePixelRatio に合わせる
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !displayW || !displayH) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(displayW * dpr);
    c.height = Math.round(displayH * dpr);
    if (phase === "viewing") {
      const g = rec.current.gaze;
      rendererRef.current?.render(g.x, g.y);
    }
  }, [displayW, displayH, phase]);

  const finish = useCallback(
    (completed: boolean) => {
      const r = rec.current;
      if (r.finished) return;
      r.finished = true;
      window.clearTimeout(r.timer);
      cancelAnimationFrame(r.raf);
      const now = performance.now();
      if (r.outsideSince !== null) r.outsideMs += now - r.outsideSince;
      const elapsed = Math.min(now - r.onset, durationMs);
      const samples = r.samples.filter((s) => s.t <= durationMs);
      const hz = elapsed > 0 ? (samples.length / elapsed) * 1000 : 0;
      setPhase("done");
      const result: TrialResult = {
        imageId: imageId ?? src,
        imageSrc: src,
        imageWidth: natural?.w ?? 0,
        imageHeight: natural?.h ?? 0,
        displayWidth: displayW,
        displayHeight: displayH,
        durationMs,
        startedAt: r.startedAt,
        completed,
        startGaze: requireCenterStart && natural ? { x: natural.w / 2, y: natural.h / 2 } : null,
        samples,
        quality: {
          samplingRateHz: Math.round(hz * 100) / 100,
          samplingRateOk: hz >= MIN_SAMPLING_HZ,
          outsideMs: Math.round(r.outsideMs),
          lostFocus: r.lostFocus,
          noMovement: samples.length === 0,
        },
        foveation: params,
        pointerType: r.pointerType,
        devicePixelRatio: window.devicePixelRatio || 1,
        userAgent: navigator.userAgent,
      };
      onCompleteRef.current?.(result);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [durationMs, imageId, src, natural, displayW, displayH, paramsKey, requireCenterStart],
  );

  const start = useCallback(() => {
    const r = rec.current;
    if (!natural) return;
    r.samples = [];
    r.gaze = { x: natural.w / 2, y: natural.h / 2 };
    r.outsideSince = null;
    r.outsideMs = 0;
    r.lostFocus = false;
    r.finished = false;
    r.onset = performance.now();
    r.startedAt = new Date().toISOString();
    setPhase("viewing");
    r.timer = window.setTimeout(() => finish(true), durationMs);
  }, [natural, durationMs, finish]);

  // active が false になったら待機に戻す
  useEffect(() => {
    if (!active && phase === "viewing") {
      rec.current.finished = true;
      window.clearTimeout(rec.current.timer);
      setPhase("waiting");
    }
  }, [active, phase]);

  // 中央クリック不要の場合は自動で開始
  useEffect(() => {
    if (active && phase === "waiting" && !requireCenterStart && displayH) start();
  }, [active, phase, requireCenterStart, displayH, start]);

  // 提示中の描画ループ（マウスイベントの頻度に関係なく毎フレーム描く）
  useEffect(() => {
    if (phase !== "viewing") return;
    const r = rec.current;
    const loop = () => {
      rendererRef.current?.render(r.gaze.x, r.gaze.y);
      r.raf = requestAnimationFrame(loop);
    };
    r.raf = requestAnimationFrame(loop);
    const onBlur = () => {
      r.lostFocus = true;
    };
    const onVis = () => {
      if (document.hidden) r.lostFocus = true;
    };
    window.addEventListener("blur", onBlur);
    document.addEventListener("visibilitychange", onVis);
    return () => {
      cancelAnimationFrame(r.raf);
      window.removeEventListener("blur", onBlur);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [phase]);

  // 待機中・終了後はぼかした状態（注視点なし）を表示する
  useEffect(() => {
    if (phase === "waiting" || phase === "done") {
      rendererRef.current?.render(-1e6, -1e6);
    }
  }, [phase, displayW, displayH]);

  // アンマウント時に計測を止める
  useEffect(
    () => () => {
      window.clearTimeout(rec.current.timer);
      cancelAnimationFrame(rec.current.raf);
    },
    [],
  );

  const toImage = (clientX: number, clientY: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return {
      x: ((clientX - rect.left) / rect.width) * (natural?.w ?? 0),
      y: ((clientY - rect.top) / rect.height) * (natural?.h ?? 0),
    };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (phase !== "viewing") return;
    const r = rec.current;
    r.pointerType = e.pointerType;
    // 合体されたイベントも取り出して、ブラウザーが出せる最高頻度で記録する
    const events =
      typeof e.nativeEvent.getCoalescedEvents === "function" ? e.nativeEvent.getCoalescedEvents() : [];
    const list = events.length ? events : [e.nativeEvent];
    for (const ev of list) {
      const t = ev.timeStamp - r.onset;
      if (t < 0 || t > durationMs) continue;
      const p = toImage(ev.clientX, ev.clientY);
      r.samples.push({ t: Math.round(t * 100) / 100, x: Math.round(p.x * 10) / 10, y: Math.round(p.y * 10) / 10 });
      r.gaze = p;
    }
  };

  const onPointerLeave = () => {
    if (phase === "viewing" && rec.current.outsideSince === null) rec.current.outsideSince = performance.now();
  };
  const onPointerEnter = () => {
    const r = rec.current;
    if (phase === "viewing" && r.outsideSince !== null) {
      r.outsideMs += performance.now() - r.outsideSince;
      r.outsideSince = null;
    }
  };

  const showStart = active && phase === "waiting" && requireCenterStart;

  return (
    <div
      ref={wrapRef}
      className={className}
      style={{
        position: "relative",
        width: displayW || "100%",
        height: displayH || undefined,
        userSelect: "none",
        touchAction: "none",
        background: "#000",
        ...style,
      }}
      data-phase={phase}
    >
      <canvas
        ref={canvasRef}
        onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
        onPointerEnter={onPointerEnter}
        onContextMenu={(e) => e.preventDefault()}
        style={{
          display: "block",
          width: displayW || "100%",
          height: displayH || 0,
          cursor: phase === "viewing" ? "none" : "default",
        }}
      />
      {showStart && (
        <div style={overlay}>
          <button
            type="button"
            aria-label="開始"
            onClick={start}
            style={{
              width: 18,
              height: 18,
              borderRadius: "50%",
              border: "2px solid #fff",
              background: "#e00",
              cursor: "pointer",
              padding: 0,
              boxShadow: "0 0 0 4px rgba(0,0,0,.35)",
            }}
          />
          <div style={{ ...label, top: "calc(50% + 22px)" }}>{startLabel}</div>
        </div>
      )}
      {phase === "loading" && <div style={{ ...overlay, color: "#fff" }}>読み込み中…</div>}
      {phase === "error" && <div style={{ ...overlay, color: "#f88", padding: 16 }}>{error}</div>}
    </div>
  );
}

const overlay: CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 14,
  fontFamily: "system-ui, sans-serif",
};

const label: CSSProperties = {
  position: "absolute",
  left: 0,
  right: 0,
  textAlign: "center",
  color: "#fff",
  textShadow: "0 1px 3px #000",
};
