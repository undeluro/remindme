import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Docker on Windows doesn't forward file-change events into the container — poll instead,
    // otherwise edits never reach the browser (even after F5)
    watch: { usePolling: true, interval: 300 },
  },
})
