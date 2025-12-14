import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    // 🚨 TRAFFIC CONTROLLER: Middleware to redirect ALL /viewer/* PAGE requests to /viewer/index.html
    {
      name: 'viewer-redirect',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          // 🚨 LOGGING: Check if the server even sees the request
          if (req.url.includes('viewer')) {
            console.log('🚨 VITE SERVER: Request received for:', req.url);
          }

          // Get just the pathname without query params for extension checking
          const pathname = req.url.split('?')[0];

          // SKIP static assets - don't rewrite .js, .css, .wasm, .map, .json, .png, .ico, etc.
          const hasFileExtension = /\.(js|css|wasm|map|json|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot|txt|LICENSE)$/i.test(pathname);
          if (hasFileExtension) {
            console.log('⏭️ SKIPPING static asset:', req.url);
            next();
            return;
          }

          // Match ALL viewer paths: /viewer, /viewer/, /viewer?..., /viewer/anything, /viewer/anything?...
          // But NOT /viewer/something.js (those are caught above by extension check)
          const isViewerPageRequest = pathname === '/viewer' ||
            pathname === '/viewer/' ||
            pathname.startsWith('/viewer/');

          if (isViewerPageRequest) {
            console.log('✅ REWRITING PAGE to /viewer/index.html');
            // Keep query parameters intact
            if (req.url.includes('?')) {
              const query = req.url.split('?')[1];
              req.url = '/viewer/index.html?' + query;
            } else {
              req.url = '/viewer/index.html';
            }
          }
          next();
        });
      }
    }
  ],
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    }
  }
})