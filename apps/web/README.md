# apps/web: dashboard (Next.js + TypeScript)

Stage 7 grows this into the product dashboard (upload or stream neural data → visualize → decode). Today it is a status shell that shows the pipeline, CAP-1 and the roadmap, and pings the API.

```bash
npm install
npm run dev            # http://localhost:3000 (expects the API at NEXT_PUBLIC_API_URL, default http://localhost:8000)
npm run lint && npm run typecheck && npm run build
```

Rules: the web app talks to the API only. It never reads datasets or model files directly, and it never sends neural data to third-party services.
