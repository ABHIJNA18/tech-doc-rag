// Vite configuration: React support, Tailwind CSS and the dev server port.
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Fixed port, so it does not clash with other Vite apps on the default 5173.
    port: 5180,
    strictPort: true,
  },
})
