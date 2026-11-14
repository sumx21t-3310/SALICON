import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri の開発サーバー設定（https://v2.tauri.app/start/frontend/vite/）
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: 1421 } : undefined,
    watch: { ignored: ["**/src-tauri/**"] },
  },
  envPrefix: ["VITE_", "TAURI_ENV_*"],
  build: {
    target: "es2020",
    outDir: "dist",
  },
});
