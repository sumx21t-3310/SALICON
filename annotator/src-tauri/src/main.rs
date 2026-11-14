// リリースビルドで Windows のコンソールを出さない
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    salicon_annotator_lib::run()
}
