//! SALICON Annotator のバックエンド。
//!
//! フロントエンド（React）から呼ぶコマンドを提供する。
//! - 画像フォルダーの一覧取得
//! - 画像ファイルの読み込み（WebGL で使うため、バイト列で返して Blob URL にする）
//! - 結果 JSON の保存

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::ipc::Response;

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "gif"];

#[derive(Debug, Serialize, PartialEq)]
pub struct ImageEntry {
    /// ファイル名の stem。学習データの画像 ID になる。
    pub id: String,
    /// 絶対パス
    pub path: String,
}

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// フォルダー直下の画像をファイル名順に返す
pub fn list_images_in(dir: &Path) -> Result<Vec<ImageEntry>, String> {
    let rd = fs::read_dir(dir).map_err(|e| format!("フォルダーを開けません: {}: {e}", dir.display()))?;
    let mut out: Vec<ImageEntry> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_image(p))
        .filter_map(|p| {
            let id = p.file_stem()?.to_str()?.to_string();
            Some(ImageEntry { id, path: p.to_string_lossy().into_owned() })
        })
        .collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// ファイル名に使えない文字を置き換える
pub fn sanitize_file_name(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || "-_.".contains(c) { c } else { '_' })
        .collect();
    let s = s.trim_matches('.').to_string();
    if s.is_empty() { "result".into() } else { s }
}

/// 結果を書き込む。overwrite が false で同名のファイルがあれば、連番を付けて上書きを避ける。
/// セッション中は試行ごとに同じファイルへ上書き保存し、途中で落ちてもデータが残るようにする。
pub fn write_result_in(dir: &Path, file_name: &str, contents: &str, overwrite: bool) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|e| format!("フォルダーを作れません: {e}"))?;
    let base = sanitize_file_name(file_name.trim_end_matches(".json"));
    let mut path = dir.join(format!("{base}.json"));
    let mut n = 1;
    while !overwrite && path.exists() {
        path = dir.join(format!("{base}_{n}.json"));
        n += 1;
    }
    fs::write(&path, contents).map_err(|e| format!("保存できません: {e}"))?;
    Ok(path)
}

#[tauri::command]
fn list_images(dir: String) -> Result<Vec<ImageEntry>, String> {
    list_images_in(Path::new(&dir))
}

#[tauri::command]
fn read_image(path: String) -> Result<Response, String> {
    let p = Path::new(&path);
    if !is_image(p) {
        return Err(format!("画像ファイルではありません: {path}"));
    }
    let bytes = fs::read(p).map_err(|e| format!("画像を読み込めません: {path}: {e}"))?;
    Ok(Response::new(bytes))
}

#[tauri::command]
fn save_result(dir: String, file_name: String, contents: String, overwrite: bool) -> Result<String, String> {
    write_result_in(Path::new(&dir), &file_name, &contents, overwrite).map(|p| p.to_string_lossy().into_owned())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![list_images, read_image, save_result])
        .run(tauri::generate_context!())
        .expect("アプリの起動に失敗しました");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_only_images_sorted() {
        let d = tempfile::tempdir().unwrap();
        for f in ["b.PNG", "a.jpg", "note.txt"] {
            fs::write(d.path().join(f), b"x").unwrap();
        }
        fs::create_dir(d.path().join("sub.png")).unwrap();
        let ids: Vec<_> = list_images_in(d.path()).unwrap().into_iter().map(|e| e.id).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn does_not_overwrite_results() {
        let d = tempfile::tempdir().unwrap();
        let p1 = write_result_in(d.path(), "p001.json", "{}", false).unwrap();
        let p2 = write_result_in(d.path(), "p001.json", "{}", false).unwrap();
        assert_ne!(p1, p2);
        assert!(p2.ends_with("p001_1.json"));
        let p3 = write_result_in(d.path(), "p001.json", "[1]", true).unwrap();
        assert_eq!(p1, p3);
        assert_eq!(fs::read_to_string(p3).unwrap(), "[1]");
    }

    #[test]
    fn sanitizes_names() {
        assert_eq!(sanitize_file_name("../a b/c"), "_a_b_c");
        assert_eq!(sanitize_file_name(""), "result");
    }
}
