import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Builds into ../web, which distil/serve.py serves directly -- so the Python
// server is the only thing a user needs to run. `npm run dev` proxies /api to
// that same server, so the dev server and the built app talk to one backend.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../web',
    emptyOutDir: true,
    // Readable output. This is a research repo: someone reading the built file
    // to see what it does should not meet a single minified line.
    minify: false,
    rollupOptions: { output: { entryFileNames: 'app.js', assetFileNames: '[name][extname]' } },
  },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
})
