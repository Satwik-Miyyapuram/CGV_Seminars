import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    // Honor the PORT env var when provided (e.g. by the Claude preview launcher's autoPort);
    // falls back to Vite's default 5173 for a normal `npm run dev`.
    port: process.env.PORT ? Number(process.env.PORT) : undefined,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});