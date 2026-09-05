/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Environment-driven configuration: no environment-specific URLs are
// hardcoded in source. VITE_PROXY_TARGET overrides the backend the dev
// proxy forwards /api requests to.
//
// Default points at the DB-backed FastAPI instance on 127.0.0.1:8001. Port
// 8000 has historically hosted a stale instance started WITHOUT DATABASE_URL
// (reports "degraded" / database:false); the proxy must never silently fall
// back to it.
const proxyTarget = process.env.VITE_PROXY_TARGET ?? "http://127.0.0.1:8001";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: proxyTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    css: false,
  },
});
