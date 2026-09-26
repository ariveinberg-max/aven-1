export type StageStatus = "done" | "next" | "planned";

export interface Stage {
  id: number;
  name: string;
  summary: string;
  gate: string;
  status: StageStatus;
}

// Mirrors docs/product/roadmap.md. Update both together.
export const STAGES: readonly Stage[] = [
  { id: 0, name: "Research & foundations", summary: "Landscape, CAP-1 spec, architecture, harness", gate: "Harness runs in CI", status: "done" },
  { id: 1, name: "Data ingestion", summary: "Public datasets → canonical recordings (license-gated)", gate: "Datasets load + QA", status: "next" },
  { id: 2, name: "Signal processing", summary: "Versioned transforms, epoching, artifacts", gate: "Gate 0: reproduce MOABB", status: "planned" },
  { id: 3, name: "Neural representation", summary: "Encoders, alignment, training scaffold", gate: "Contract tests", status: "planned" },
  { id: 4, name: "Baseline models", summary: "B0–B6 on regimes R0–R3", gate: "Gate 1: thresholds fixed", status: "planned" },
  { id: 5, name: "Proprietary model", summary: "Montage-agnostic, calibration-efficient", gate: "Gate 2: locked holdout", status: "planned" },
  { id: 6, name: "API", summary: "Registry, inference, auth, tenancy", gate: "Latency + security review", status: "planned" },
  { id: 7, name: "Product", summary: "Dashboard, calibration game, device & OS bridges", gate: "Dogfooding", status: "planned" },
  { id: 8, name: "Real-world testing", summary: "Consumer devices, consented pilot", gate: "Gate 3: real device", status: "planned" },
];
