import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Development: `npm run dev` serves the app on :5173 and proxies /api to the
// Rust server started with `latticefsm serve` (127.0.0.1:8000, or VITE_PROXY_TARGET).
// Production: `npm run build` writes ./dist, which `latticefsm serve` serves
// directly, so relative /api URLs work without any configuration.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
