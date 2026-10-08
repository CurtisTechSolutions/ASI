import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The Rust server embeds dist/index.html, dist/app.js and dist/app.css, so the output names are fixed.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8080' } },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: { entryFileNames: 'app.js', chunkFileNames: 'app-[name].js', assetFileNames: 'app.[ext]', codeSplitting: false },
    },
  },
});
