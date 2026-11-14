import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { SaliconAnnotator } from "./SaliconAnnotator";
import type { AnnotationImage, FoveationParams, SessionResult, TrialResult } from "./types";

export interface SaliconSessionProps {
  images: AnnotationImage[];
  /** 本番前の練習用画像。結果には含めない。論文の実験室条件では 10 枚。 */
  practiceImages?: AnnotationImage[];
  /** 提示時間（ms）。論文は 5000。 */
  durationMs?: number;
  /** 試行間の待ち時間（ms）。論文は 2000。 */
  intervalMs?: number;
  /** 画像の順序をランダムにする。論文はランダム順。 */
  shuffle?: boolean;
  /** 乱数のシード。参加者ごとに順序を再現したい場合に指定する。 */
  seed?: number;
  /** 表示幅（CSS px） */
  width?: number;
  foveation?: Partial<FoveationParams>;
  participantId?: string;
  /** 1 試行ごとに呼ばれる。サーバーへ逐次送信する場合に使う。 */
  onTrial?: (trial: TrialResult) => void;
  /** すべての試行が終わったときに呼ばれる */
  onFinish?: (result: SessionResult) => void;
  /** 開始前に表示する説明文 */
  instructions?: React.ReactNode;
  style?: CSSProperties;
}

type Stage = "intro" | "practice" | "main" | "interval" | "finished";

/** 決定的な乱数（mulberry32） */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled<T>(xs: T[], seed: number): T[] {
  const r = rng(seed);
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const DEFAULT_INSTRUCTIONS = (
  <>
    <p>これから画像が 1 枚ずつ表示されます。画像はぼやけていて、マウスカーソルの周りだけがはっきり見えます。</p>
    <p>見たいところへ自由にマウスを動かして、画像を眺めてください。各画像の表示は数秒で自動的に終わります。</p>
    <p>各画像の前に中央の赤い点をクリックすると表示が始まります。</p>
  </>
);

/**
 * 複数の画像を順に提示するセッション。練習試行、ランダム順、試行間の待ち時間を扱う。
 */
export function SaliconSession({
  images,
  practiceImages = [],
  durationMs = 5000,
  intervalMs = 2000,
  shuffle = true,
  seed,
  width = 800,
  foveation,
  participantId,
  onTrial,
  onFinish,
  instructions = DEFAULT_INSTRUCTIONS,
  style,
}: SaliconSessionProps) {
  const order = useMemo(
    () => (shuffle ? shuffled(images, seed ?? Math.floor(Math.random() * 2 ** 31)) : images),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [images, shuffle, seed],
  );
  const [stage, setStage] = useState<Stage>("intro");
  const [index, setIndex] = useState(0);
  const [isPractice, setIsPractice] = useState(false);
  const trials = useRef<TrialResult[]>([]);
  const startedAt = useRef("");

  const list = isPractice ? practiceImages : order;
  const current = list[index];

  const begin = () => {
    startedAt.current = new Date().toISOString();
    trials.current = [];
    setIndex(0);
    if (practiceImages.length) {
      setIsPractice(true);
      setStage("practice");
    } else {
      setIsPractice(false);
      setStage("main");
    }
  };

  const handleComplete = useCallback(
    (t: TrialResult) => {
      if (!isPractice) {
        trials.current.push(t);
        onTrial?.(t);
      }
      setStage("interval");
    },
    [isPractice, onTrial],
  );

  // 試行間の待ち時間が過ぎたら次へ進む
  useEffect(() => {
    if (stage !== "interval") return;
    const id = window.setTimeout(() => {
      const next = index + 1;
      if (next < list.length) {
        setIndex(next);
        setStage(isPractice ? "practice" : "main");
      } else if (isPractice) {
        setIsPractice(false);
        setIndex(0);
        setStage("main");
      } else {
        setStage("finished");
        onFinish?.({
          participantId,
          startedAt: startedAt.current,
          finishedAt: new Date().toISOString(),
          trials: trials.current,
        });
      }
    }, intervalMs);
    return () => window.clearTimeout(id);
  }, [stage, index, list.length, isPractice, intervalMs, onFinish, participantId]);

  const box: CSSProperties = {
    width,
    fontFamily: "system-ui, sans-serif",
    ...style,
  };

  if (stage === "intro") {
    return (
      <div style={box}>
        {instructions}
        <button type="button" onClick={begin} style={btn}>
          {practiceImages.length ? "練習を始める" : "始める"}
        </button>
      </div>
    );
  }

  if (stage === "finished") {
    return (
      <div style={box}>
        <p>すべての画像が終わりました。ご協力ありがとうございました。</p>
      </div>
    );
  }

  const progress = isPractice
    ? `練習 ${index + 1} / ${practiceImages.length}`
    : `${index + 1} / ${order.length}`;

  return (
    <div style={box}>
      <div style={{ fontSize: 12, color: "#888", marginBottom: 6 }}>{progress}</div>
      {current && (
        <SaliconAnnotator
          key={`${isPractice ? "p" : "m"}-${index}`}
          src={current.src}
          imageId={current.id}
          width={width}
          durationMs={durationMs}
          foveation={foveation}
          active={stage !== "interval"}
          onComplete={handleComplete}
          onError={() => setStage("interval")}
          style={{ visibility: stage === "interval" ? "hidden" : "visible" }}
        />
      )}
    </div>
  );
}

const btn: CSSProperties = {
  marginTop: 12,
  padding: "8px 20px",
  fontSize: 14,
  cursor: "pointer",
};
