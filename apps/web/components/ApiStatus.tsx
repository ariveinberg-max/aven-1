"use client";

import { useEffect, useState } from "react";

type State = "checking" | "online" | "offline";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function ApiStatus() {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_URL}/healthz`, { signal: controller.signal })
      .then((res) => setState(res.ok ? "online" : "offline"))
      .catch(() => {
        if (!controller.signal.aborted) setState("offline");
      });
    return () => controller.abort();
  }, []);

  return (
    <span className={`pill pill-${state}`} aria-live="polite">
      API {state}
    </span>
  );
}
