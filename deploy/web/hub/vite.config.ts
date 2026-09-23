import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// base "/hub/" because authproxy mounts the built bundle under /hub/ - the dashboard
// itself owns "/" and is proxied to the GPU box, so the hub cannot take the root.
export default defineConfig({
  base: "/hub/",
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
  server: {
    port: 5173,
    // `npm run dev` talks to a local authproxy so the API shape is exercised for real.
    proxy: { "/_jlens/api": "http://127.0.0.1:7870" },
  },
});
