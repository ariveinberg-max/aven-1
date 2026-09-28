import { defineConfig, devices } from "@playwright/test";

// End-to-end: browser → Next.js (production build) → API (uvicorn, demo mode) → model.
// Needs a demo bundle in artifacts/models (`make demo-model`).
const API_PORT = 8765;
const WEB_PORT = 3100;

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    trace: "retain-on-failure",
    launchOptions: process.env.PW_CHROMIUM_PATH ? { executablePath: process.env.PW_CHROMIUM_PATH } : {},
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `uv run uvicorn neurolayer_api.app:app --port ${API_PORT}`,
      cwd: "../..",
      url: `http://127.0.0.1:${API_PORT}/healthz`,
      env: {
        NEUROLAYER_AUTH_DISABLED: "1",
        NEUROLAYER_DEMO: "1",
        NEUROLAYER_CORS_ORIGINS: `http://127.0.0.1:${WEB_PORT}`,
        NEUROLAYER_MODELS_DIR: "artifacts/models",
      },
      timeout: 120_000,
      reuseExistingServer: false,
    },
    {
      // NEXT_PUBLIC_API_URL is inlined at build time (and pinned in the CSP), so the
      // e2e run builds against the test API.
      command: `npx next build && npx next start --port ${WEB_PORT} --hostname 127.0.0.1`,
      url: `http://127.0.0.1:${WEB_PORT}`,
      env: { NEXT_PUBLIC_API_URL: `http://127.0.0.1:${API_PORT}` },
      timeout: 180_000,
      reuseExistingServer: false,
    },
  ],
});
