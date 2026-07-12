import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy API + generated images to the local Flask backend (backend/server.py).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8787',
      '/outputs': 'http://127.0.0.1:8787',
    },
  },
})
