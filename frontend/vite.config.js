import { defineConfig, loadEnv } from "vite";
import process from "process";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

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
      react(),
      tailwindcss(),
    ],
    server: {
      port: 5173,
      host: true,
      ...(allowedHosts.length > 0 && { allowedHosts }),
      proxy: {
        "/api": {
          target: env.VITE_API_BASE_URL || "http://localhost:8000",
          changeOrigin: true,
        },
      },
    },
  };
});