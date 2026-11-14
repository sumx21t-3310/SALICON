/**
 * Tauri バックエンドとのやり取り。
 * ブラウザーで `npm run dev` したときは、ファイル選択とダウンロードで代替する。
 */
import { invoke, isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

export interface LoadedImage {
  id: string;
  /** Blob URL（同一オリジンなので WebGL でそのまま使える） */
  src: string;
}

export const inTauri = (): boolean => {
  try {
    return isTauri();
  } catch {
    return false;
  }
};

interface ImageEntry {
  id: string;
  path: string;
}

const MIME: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  bmp: "image/bmp",
  gif: "image/gif",
};

/** フォルダー選択ダイアログを開く。キャンセル時は null */
export async function pickFolder(title: string): Promise<string | null> {
  const r = await open({ directory: true, multiple: false, title });
  return typeof r === "string" ? r : null;
}

/** フォルダー内の画像を読み込み、Blob URL にする */
export async function loadFolderImages(dir: string): Promise<LoadedImage[]> {
  const entries = await invoke<ImageEntry[]>("list_images", { dir });
  const out: LoadedImage[] = [];
  for (const e of entries) {
    const buf = await invoke<ArrayBuffer>("read_image", { path: e.path });
    const ext = e.path.split(".").pop()?.toLowerCase() ?? "";
    const blob = new Blob([buf], { type: MIME[ext] ?? "application/octet-stream" });
    out.push({ id: e.id, src: URL.createObjectURL(blob) });
  }
  return out;
}

/** ブラウザー用: input[type=file] で選んだファイルを読み込む */
export function loadFileListImages(files: FileList): LoadedImage[] {
  return Array.from(files)
    .filter((f) => f.type.startsWith("image/"))
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((f) => ({ id: f.name.replace(/\.[^.]+$/, ""), src: URL.createObjectURL(f) }));
}

/**
 * 結果を保存する。Tauri では出力フォルダーへ書き込み、保存先のパスを返す。
 * ブラウザーでは overwrite=false（最終保存）のときだけダウンロードする。
 */
export async function saveResult(
  dir: string | null,
  fileName: string,
  data: unknown,
  overwrite: boolean,
): Promise<string | null> {
  const contents = JSON.stringify(data, null, 2);
  if (inTauri() && dir) {
    return invoke<string>("save_result", { dir, fileName, contents, overwrite });
  }
  if (!overwrite) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([contents], { type: "application/json" }));
    a.download = `${fileName}.json`;
    a.click();
    return a.download;
  }
  return null;
}
