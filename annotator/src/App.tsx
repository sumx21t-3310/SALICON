import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { SaliconSession, type SessionResult, type TrialResult } from "./lib";
import {
  inTauri,
  loadFileListImages,
  loadFolderImages,
  pickFolder,
  revokeImages,
  saveResult,
  type LoadedImage,
} from "./backend";
import { Heatmap } from "./Heatmap";

interface Settings {
  imageDir: string;
  practiceDir: string;
  outputDir: string;
  durationMs: number;
  intervalMs: number;
  width: number;
  pixelsPerDegree: number;
  shuffle: boolean;
}

const DEFAULTS: Settings = {
  imageDir: "",
  practiceDir: "",
  outputDir: "",
  durationMs: 5000,
  intervalMs: 2000,
  width: 800,
  pixelsPerDegree: 7.5,
  shuffle: true,
};

const STORAGE_KEY = "salicon-annotator/settings";

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : DEFAULTS;
  } catch {
    return DEFAULTS;
  }
}

type Screen = "setup" | "running" | "result";

const stamp = () => new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "");

/** 保存用に Blob URL を外す（再起動後は意味が無いため） */
const slimTrial = (t: TrialResult): TrialResult => ({ ...t, imageSrc: t.imageId });

export function App() {
  const tauri = inTauri();
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [participantId, setParticipantId] = useState("");
  const [images, setImages] = useState<LoadedImage[]>([]);
  const [practice, setPractice] = useState<LoadedImage[]>([]);
  const [screen, setScreen] = useState<Screen>("setup");
  const [result, setResult] = useState<SessionResult | null>(null);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const session = useRef({ fileName: "", startedAt: "", trials: [] as TrialResult[] });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* 保存できなくても動作には影響しない */
    }
  }, [settings]);

  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setSettings((s) => ({ ...s, [k]: v }));

  // 前回のフォルダーを自動で読み込む
  useEffect(() => {
    if (!tauri) return;
    if (settings.imageDir) void reload("imageDir", settings.imageDir);
    if (settings.practiceDir) void reload("practiceDir", settings.practiceDir);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** 画像の一覧を差し替え、古い Blob URL を解放する */
  function replaceImages(key: "imageDir" | "practiceDir", next: LoadedImage[]) {
    const setter = key === "imageDir" ? setImages : setPractice;
    setter((prev) => {
      revokeImages(prev);
      return next;
    });
  }

  // アンマウント時に解放する
  const latest = useRef({ images, practice });
  latest.current = { images, practice };
  useEffect(
    () => () => {
      revokeImages(latest.current.images);
      revokeImages(latest.current.practice);
    },
    [],
  );

  /** ブラウザー用: 選んだファイルを読み込む（ID が重複していれば読み込まない） */
  function pickFiles(key: "imageDir" | "practiceDir", files: FileList | null) {
    if (!files) return;
    try {
      replaceImages(key, loadFileListImages(files));
      setMessage(null);
    } catch (e) {
      replaceImages(key, []);
      setMessage(e instanceof Error ? e.message : String(e));
    }
  }

  async function reload(key: "imageDir" | "practiceDir", dir: string) {
    setBusy(true);
    setMessage(null);
    try {
      const imgs = await loadFolderImages(dir);
      replaceImages(key, imgs);
      set(key, dir);
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function choose(key: keyof Pick<Settings, "imageDir" | "practiceDir" | "outputDir">) {
    const label = { imageDir: "画像フォルダー", practiceDir: "練習用画像フォルダー", outputDir: "保存先フォルダー" }[key];
    const dir = await pickFolder(`${label}を選択`);
    if (!dir) return;
    if (key === "outputDir") set(key, dir);
    else await reload(key, dir);
  }

  const srcById = useMemo(() => new Map(images.map((i) => [i.id, i.src])), [images]);

  const canStart =
    images.length > 0 && participantId.trim() !== "" && (!tauri || settings.outputDir !== "") && !busy;

  function start() {
    session.current = {
      fileName: `${participantId.trim()}_${stamp()}`,
      startedAt: new Date().toISOString(),
      trials: [],
    };
    setSavedPath(null);
    setMessage(null);
    setScreen("running");
  }

  const snapshot = (finishedAt: string): SessionResult => ({
    participantId: participantId.trim(),
    startedAt: session.current.startedAt,
    finishedAt,
    trials: session.current.trials.map(slimTrial),
  });

  // 試行ごとに同じファイルへ上書き保存し、途中で終了してもデータを残す
  async function onTrial(t: TrialResult) {
    session.current.trials.push(t);
    try {
      const p = await saveResult(settings.outputDir, session.current.fileName, snapshot(""), true);
      if (p) setSavedPath(p);
    } catch (e) {
      setMessage(`途中保存に失敗しました: ${e}`);
    }
  }

  async function onFinish(r: SessionResult) {
    setResult(r);
    setScreen("result");
    try {
      const p = await saveResult(
        settings.outputDir,
        session.current.fileName,
        snapshot(r.finishedAt),
        tauri, // Tauri では途中保存したファイルを上書き、ブラウザーではダウンロード
      );
      setSavedPath(p);
    } catch (e) {
      setMessage(`保存に失敗しました: ${e}`);
    }
  }

  async function abort() {
    const r = snapshot(new Date().toISOString());
    setResult(r);
    setScreen("result");
    if (r.trials.length === 0) {
      setMessage("セッションを中断しました。完了した試行はありません。");
      return;
    }
    try {
      // Tauri では途中保存済みのファイルに終了時刻を入れて上書き、ブラウザーではダウンロード
      const p = await saveResult(settings.outputDir, session.current.fileName, r, tauri);
      setSavedPath(p);
      setMessage(`セッションを中断しました。中断までの ${r.trials.length} 試行を保存しました。`);
    } catch (e) {
      setMessage(`セッションを中断しましたが、保存に失敗しました: ${e}`);
    }
  }

  /** ブラウザー用: 結果画面から JSON をもう一度ダウンロードする */
  async function exportAgain() {
    if (!result) return;
    const p = await saveResult(null, session.current.fileName, result, false);
    setSavedPath(p);
  }

  // Esc で中断
  useEffect(() => {
    if (screen !== "running") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") void abort();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screen]);

  if (screen === "running") {
    return (
      <div style={{ ...page, display: "flex", justifyContent: "center", paddingTop: 24 }}>
        <SaliconSession
          images={images}
          practiceImages={practice}
          durationMs={settings.durationMs}
          intervalMs={settings.intervalMs}
          shuffle={settings.shuffle}
          width={settings.width}
          foveation={{ pixelsPerDegree: settings.pixelsPerDegree }}
          participantId={participantId.trim()}
          onTrial={onTrial}
          onFinish={onFinish}
        />
      </div>
    );
  }

  if (screen === "result" && result) {
    return (
      <div style={page}>
        <h1 style={h1}>結果（{result.participantId}）</h1>
        {message && <p style={note}>{message}</p>}
        {savedPath && <p style={{ fontSize: 13 }}>保存先: {savedPath}</p>}
        {!tauri && result.trials.length > 0 && (
          <button type="button" style={{ ...btn, marginRight: 8 }} onClick={exportAgain}>
            JSON を保存
          </button>
        )}
        <button type="button" style={btn} onClick={() => setScreen("setup")}>
          設定に戻る
        </button>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginTop: 16 }}>
          {result.trials.map((t, i) => (
            <figure key={`${t.imageId}-${i}`} style={{ margin: 0 }}>
              <Heatmap trial={t} src={srcById.get(t.imageId) ?? t.imageSrc} />
              <figcaption style={{ fontSize: 12, marginTop: 4 }}>
                {t.imageId} / {t.samples.length} サンプル / {t.quality.samplingRateHz} Hz
                {t.quality.samplingRateOk ? "" : "（12 Hz 未満のため無効）"}
                {t.quality.lostFocus ? "（フォーカス喪失）" : ""}
              </figcaption>
            </figure>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={page}>
      <h1 style={h1}>SALICON Annotator</h1>
      <p style={{ fontSize: 13, color: "#555", marginTop: 0 }}>
        SALICON（Jiang et al., CVPR 2015）方式のマウス追跡で、画像ごとの注視データを集めます。
      </p>

      <section style={card}>
        <h2 style={h2}>画像</h2>
        {tauri ? (
          <>
            <Row label="本番用">
              <button type="button" style={btn} onClick={() => choose("imageDir")} disabled={busy}>
                フォルダーを選択
              </button>
              <Path value={settings.imageDir} count={images.length} />
            </Row>
            <Row label="練習用（任意）">
              <button type="button" style={btn} onClick={() => choose("practiceDir")} disabled={busy}>
                フォルダーを選択
              </button>
              {settings.practiceDir && (
                <button
                  type="button"
                  style={btnLink}
                  onClick={() => {
                    set("practiceDir", "");
                    replaceImages("practiceDir", []);
                  }}
                >
                  解除
                </button>
              )}
              <Path value={settings.practiceDir} count={practice.length} />
            </Row>
            <Row label="保存先">
              <button type="button" style={btn} onClick={() => choose("outputDir")}>
                フォルダーを選択
              </button>
              <Path value={settings.outputDir} />
            </Row>
          </>
        ) : (
          <>
            <Row label="本番用">
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={(e) => pickFiles("imageDir", e.target.files)}
              />
              <span style={muted}>{images.length} 枚</span>
            </Row>
            <Row label="練習用（任意）">
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={(e) => pickFiles("practiceDir", e.target.files)}
              />
              <span style={muted}>{practice.length} 枚</span>
            </Row>
            <p style={muted}>ブラウザーで開いているため、結果は最後に JSON でダウンロードします。</p>
          </>
        )}
      </section>

      <section style={card}>
        <h2 style={h2}>実験条件</h2>
        <Row label="参加者 ID">
          <input value={participantId} onChange={(e) => setParticipantId(e.target.value)} placeholder="p001" />
        </Row>
        <Row label="提示時間（ms）">
          <Num value={settings.durationMs} min={500} step={500} onChange={(v) => set("durationMs", v)} />
          <span style={muted}>論文: 5000</span>
        </Row>
        <Row label="試行間隔（ms）">
          <Num value={settings.intervalMs} min={0} step={500} onChange={(v) => set("intervalMs", v)} />
          <span style={muted}>論文: 2000</span>
        </Row>
        <Row label="表示幅（px）">
          <Num value={settings.width} min={200} step={10} onChange={(v) => set("width", v)} />
        </Row>
        <Row label="1 度あたりの画素数">
          <Num value={settings.pixelsPerDegree} min={1} step={0.5} onChange={(v) => set("pixelsPerDegree", v)} />
          <span style={muted}>画像の元解像度で指定。論文: 640 px 幅で 7.5</span>
        </Row>
        <Row label="ランダム順">
          <input type="checkbox" checked={settings.shuffle} onChange={(e) => set("shuffle", e.target.checked)} />
        </Row>
      </section>

      {message && <p style={note}>{message}</p>}
      <button type="button" style={{ ...btn, ...primary }} disabled={!canStart} onClick={start}>
        セッションを始める
      </button>
      {!canStart && !busy && (
        <p style={muted}>
          {tauri ? "画像フォルダー、保存先、参加者 ID を設定してください。" : "画像と参加者 ID を設定してください。"}
        </p>
      )}
      <p style={muted}>セッション中は Esc キーで中断できます。</p>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "8px 0", flexWrap: "wrap" }}>
      <span style={{ width: 150, fontSize: 13 }}>{label}</span>
      {children}
    </div>
  );
}

function Path({ value, count }: { value: string; count?: number }) {
  if (!value) return <span style={muted}>未選択</span>;
  return (
    <span style={{ ...muted, wordBreak: "break-all" }}>
      {value}
      {count !== undefined && `（${count} 枚）`}
    </span>
  );
}

function Num({
  value,
  onChange,
  min,
  step,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  step: number;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      step={step}
      style={{ width: 90 }}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (Number.isFinite(v) && v >= min) onChange(v);
      }}
    />
  );
}

const page: CSSProperties = {
  fontFamily: "system-ui, -apple-system, 'Hiragino Sans', 'Yu Gothic UI', sans-serif",
  padding: "16px 24px",
  maxWidth: 1000,
  margin: "0 auto",
  color: "#222",
};
const h1: CSSProperties = { fontSize: 20, margin: "8px 0" };
const h2: CSSProperties = { fontSize: 15, margin: "0 0 8px" };
const card: CSSProperties = { border: "1px solid #ddd", borderRadius: 8, padding: "12px 16px", margin: "12px 0" };
const btn: CSSProperties = { padding: "6px 14px", fontSize: 13, cursor: "pointer" };
const btnLink: CSSProperties = { ...btn, background: "none", border: "none", color: "#06c", padding: "6px 4px" };
const primary: CSSProperties = { padding: "8px 20px", fontSize: 14 };
const muted: CSSProperties = { fontSize: 12, color: "#777" };
const note: CSSProperties = { fontSize: 13, color: "#b44" };
