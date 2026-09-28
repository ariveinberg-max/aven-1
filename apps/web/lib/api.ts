// Typed client for the neurolayer API (mirrors src/neurolayer_api/app.py schemas).
// The access token lives in localStorage on this device only; it is never sent anywhere
// except the configured API.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "neurolayer.token";

export interface ModelInfo {
  id: string;
  labels: string[];
  sfreq: number;
  trial_seconds: number;
  window_offset_s: number;
  trained_channels: string[];
  synthetic_only: boolean;
}

export interface Trial {
  data: number[][]; // microvolts, channels × samples
  label: string | null;
}

export interface DemoTrials {
  synthetic: true;
  ch_names: string[];
  sfreq: number;
  trial_seconds: number;
  trials: Trial[];
}

export interface SessionInfo {
  id: string;
  model: string;
  ch_names: string[];
  sfreq: number;
  calibration_counts: Record<string, number>;
  unlabeled_trials: number;
  decoded_trials: number;
}

export interface Prediction {
  label: string;
  probabilities: Record<string, number>;
}

export interface InspectChannel {
  name: string;
  canonical: string | null;
  mu_db: number | null;
  beta_db: number | null;
  flat: boolean;
  noisy: boolean;
}

export interface InspectReport {
  format: "edf" | "bdf";
  sfreq: number;
  duration_s: number;
  n_channels: number;
  channels: InspectChannel[];
  annotations: Record<string, number>;
  line_noise_ratio: Record<string, number | null>;
  median_abs_amplitude_uv: number;
  issues: string[];
  preview: { sfreq: number; channels: string[]; data_uv: number[][] };
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    detail: string,
  ) {
    super(`${status}: ${detail}`);
  }
}

export function getToken(): string {
  try {
    return window.localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setToken(token: string): void {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Storage unavailable (private mode): the token lasts for this page only.
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && typeof init.body === "string") headers.set("Content-Type", "application/json");
  const response = await fetch(`${API_URL}${path}`, { ...init, headers });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body: unknown = await response.json();
      if (body && typeof body === "object" && "detail" in body) detail = String(body.detail);
    } catch {
      // Non-JSON error body.
    }
    throw new ApiError(response.status, detail);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

export const api = {
  models: () => request<ModelInfo[]>("/v1/models"),
  demoTrials: (model: string, nPerClass: number, seed: number) =>
    request<DemoTrials>(
      `/v1/demo/trials?model=${encodeURIComponent(model)}&n_per_class=${nPerClass}&seed=${seed}`,
    ),
  createSession: (model: string, chNames: string[], sfreq: number) =>
    request<SessionInfo>("/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ model, ch_names: chNames, sfreq }),
    }),
  calibrate: (sessionId: string, trials: Trial[]) =>
    request<SessionInfo>(`/v1/sessions/${sessionId}/calibration`, {
      method: "POST",
      body: JSON.stringify({ trials }),
    }),
  decode: (sessionId: string, trials: Trial[]) =>
    request<{ predictions: Prediction[]; latency_ms: number }>(`/v1/sessions/${sessionId}/decode`, {
      method: "POST",
      body: JSON.stringify({ trials: trials.map((t) => ({ data: t.data, label: null })) }),
    }),
  deleteSession: (sessionId: string) =>
    request<void>(`/v1/sessions/${sessionId}`, { method: "DELETE" }),
  inspect: (file: File) =>
    request<InspectReport>("/v1/recordings/inspect", {
      method: "POST",
      body: file,
      headers: { "Content-Type": "application/octet-stream" },
    }),
};

export const LABEL_TEXT: Record<string, string> = { left_hand: "Left hand", right_hand: "Right hand" };
