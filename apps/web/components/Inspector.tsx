"use client";

import { useState } from "react";

import { Traces } from "@/components/Traces";
import { api, type InspectReport } from "@/lib/api";

// WP-7.1: upload → validate → visualize. The API parses the file in a sandboxed,
// resource-limited worker and returns a summary; nothing is stored server-side.

const MAX_BYTES = 8 * 1024 * 1024; // matches the API's default request-size limit

function fmt(value: number | null, digits = 1): string {
  if (value === null) return "–";
  const rounded = Number(value.toFixed(digits));
  return (rounded === 0 ? 0 : rounded).toFixed(digits); // never show "-0.0"
}

export function Inspector() {
  const [report, setReport] = useState<InspectReport | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function upload(file: File) {
    setError("");
    setReport(null);
    if (!/\.(edf|bdf)$/i.test(file.name)) {
      setError("Only EDF/EDF+ (.edf) and BDF (.bdf) files are supported.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setError(`File is ${(file.size / 1e6).toFixed(1)} MB; the limit is ${MAX_BYTES / 1e6} MB.`);
      return;
    }
    setBusy(true);
    try {
      setReport(await api.inspect(file));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="card setup">
        <label htmlFor="recording-file">EEG recording (.edf or .bdf, up to 8 MB)</label>
        <input
          id="recording-file"
          type="file"
          accept=".edf,.bdf"
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void upload(file);
          }}
        />
        <p className="muted">
          Parsed by the API in an isolated worker with memory and time limits. The file is deleted right after
          parsing and never logged. Only upload data you are allowed to share with this server.
        </p>
        {busy ? <p aria-live="polite">Inspecting…</p> : null}
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
      </div>

      {report ? (
        <section aria-labelledby="report-title">
          <h2 id="report-title">Summary</h2>
          <dl className="facts" data-testid="inspect-summary">
            <dt>Format</dt>
            <dd>{report.format.toUpperCase()}</dd>
            <dt>Channels</dt>
            <dd>{report.n_channels}</dd>
            <dt>Sampling rate</dt>
            <dd>{report.sfreq} Hz</dd>
            <dt>Duration</dt>
            <dd>{report.duration_s.toFixed(1)} s</dd>
            <dt>Median |amplitude|</dt>
            <dd>{report.median_abs_amplitude_uv.toFixed(1)} µV</dd>
            <dt>Annotations</dt>
            <dd>
              {Object.keys(report.annotations).length
                ? Object.entries(report.annotations)
                    .map(([k, v]) => `${k} × ${v}`)
                    .join(", ")
                : "none"}
            </dd>
          </dl>
          <h2>Quality</h2>
          {report.issues.length ? (
            <ul className="issues">
              {report.issues.map((issue) => (
                <li key={issue}>{issue}</li>
              ))}
            </ul>
          ) : (
            <p>No issues found.</p>
          )}
          <h2>First {Math.round((report.preview.data_uv[0]?.length ?? 0) / report.preview.sfreq)} seconds</h2>
          <Traces channels={report.preview.channels} data={report.preview.data_uv} />
          <h2>Channels</h2>
          <table className="stages">
            <thead>
              <tr>
                <th scope="col">Channel</th>
                <th scope="col">10-05 name</th>
                <th scope="col">Mu 8–13 Hz (dB µV²/Hz)</th>
                <th scope="col">Beta 13–30 Hz (dB µV²/Hz)</th>
                <th scope="col">Flags</th>
              </tr>
            </thead>
            <tbody>
              {report.channels.map((ch) => (
                <tr key={ch.name}>
                  <td>{ch.name}</td>
                  <td>{ch.canonical ?? "unknown"}</td>
                  <td>{fmt(ch.mu_db)}</td>
                  <td>{fmt(ch.beta_db)}</td>
                  <td>{[ch.flat ? "flat" : "", ch.noisy ? "noisy" : ""].filter(Boolean).join(", ") || "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
    </div>
  );
}
