import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Development: `npm run dev` serves the app on :5173 and proxies /api to the
// Python API started with `python -m radixnet serve` (127.0.0.1:8000).
// Production: `npm run build` writes ./dist, which the Python server serves
// directly, so relative /api URLs work without any configuration.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
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
