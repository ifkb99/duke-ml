import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// Cross-origin isolation lets onnxruntime-web use several CPU threads for the
// neural network AI (production needs the same headers: see nginx.conf).
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [
    react(),
    // Installable, and playable with no connection: the service worker keeps a
    // copy of everything the app loads, the network (models/) and onnxruntime's
    // 28 MB WebAssembly included. A new build (or a retrained network: the worker
    // lists each file's hash) installs beside the old one and takes over when the
    // person accepts the update or reopens the app, so the page, its lazy chunks
    // and the three network files always come from the same build.
    VitePWA({
      registerType: 'prompt',
      includeManifestIcons: false,   // globPatterns already has them
      workbox: {
        globPatterns: ['**/*.{js,mjs,css,html,wasm,woff2,ico,png,json,onnx,bin}'],
        globIgnores: ['og.png'],
        maximumFileSizeToCacheInBytes: 64 * 1024 * 1024,
      },
      manifest: {
        name: 'The Duke',
        short_name: 'The Duke',
        description: 'The Duke, the tile-flipping strategy game, against a neural network that runs on your device.',
        id: '/',
        start_url: '/',
        display: 'standalone',
        theme_color: '#0b0a11',
        background_color: '#0b0a11',
        icons: [
          {src: 'icon-192.png', sizes: '192x192', type: 'image/png'},
          {src: 'icon-512.png', sizes: '512x512', type: 'image/png'},
          {src: 'icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable'},
        ],
      },
    }),
  ],
  worker: { format: 'es' },
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
