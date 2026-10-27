import { defineConfig, loadEnv } from "vite";
import { resolve } from "path";
import fs from "node:fs";
import process from "process";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Serve the generated agent-discovery documents with the content types the
// agent-readiness checks expect. Vite's static handler would otherwise answer
// an extensionless file such as .well-known/api-catalog as octet-stream and
// llms.txt as text/plain, and it has no markdown negotiation for `/`.
function agentDiscovery() {
  return {
    name: "agent-discovery",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url || "").split("?")[0];
        const root = process.cwd();
        const accept = req.headers.accept || "";

        if (url === "/" && accept.includes("text/markdown")) {
          const file = resolve(root, "public/llms.txt");
          if (!fs.existsSync(file)) return next();
          const body = fs.readFileSync(file, "utf8");
          res.setHeader("Content-Type", "text/markdown; charset=utf-8");
          res.setHeader("X-Markdown-Tokens", String(Math.max(1, Math.round(body.split(/\s+/).length / 0.75))));
          res.end(body);
          return;
        }

        let file = null;
        let type = null;
        if (url === "/llms.txt") {
          file = "public/llms.txt";
          type = "text/markdown; charset=utf-8";
        } else if (url === "/auth.md") {
          file = "public/auth.md";
          type = "text/markdown; charset=utf-8";
        } else if (url.startsWith("/.well-known/")) {
          file = `public${url}`;
          type = url === "/.well-known/api-catalog"
            ? "application/linkset+json; charset=utf-8"
            : url.endsWith(".md")
              ? "text/markdown; charset=utf-8"
              : "application/json; charset=utf-8";
        }
        if (!file) return next();

        const abs = resolve(root, file);
        if (!fs.existsSync(abs) || !fs.statSync(abs).isFile()) return next();
        res.setHeader("Content-Type", type);
        res.setHeader("Access-Control-Allow-Origin", "*");
        res.end(fs.readFileSync(abs));
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  // Use process.env directly — loadEnv only reads .env files, not Docker env vars.
  // Only set allowedHosts when the env var is explicitly provided; otherwise Vite
  // defaults handle localhost.
  const allowedHosts = (process.env.VITE_ALLOWED_HOSTS || env.VITE_ALLOWED_HOSTS || "")
    .split(",")
    .map((h) => h.trim())
    .filter(Boolean);

  return {
    plugins: [
      agentDiscovery(),
      react(),
      tailwindcss(),
    ],
    // Two entry points: the static landing page and the React chat app.
    build: {
      rollupOptions: {
        input: {
          main: resolve(process.cwd(), "index.html"),
          chat: resolve(process.cwd(), "chat.html"),
          salah: resolve(process.cwd(), "salah.html"),
        },
      },
    },
    server: {
      port: 5173,
      host: true,
      ...(allowedHosts.length > 0 && { allowedHosts }),
      proxy: {
        "/api": {
          target: process.env.VITE_API_PROXY_TARGET || env.VITE_API_PROXY_TARGET || env.VITE_API_BASE_URL || "http://localhost:8000",
          // Preserve the browser's host/port so Django's same-origin CSRF
          // check also works when Vite chooses a different development port.
          changeOrigin: false,
        },
      },
    },
  };
});
