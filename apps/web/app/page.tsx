import { ApiStatus } from "@/components/ApiStatus";
import { STAGES } from "@/lib/stages";

const PIPELINE = ["Brain activity", "Neural sensors", "Signal processing", "Neural representation", "Intent decoding", "Action"];

export default function Home() {
  return (
    <main className="container">
      <header className="header">
        <div>
          <h1>neurolayer</h1>
          <p className="muted">From neural signals to computer interactions.</p>
        </div>
        <ApiStatus />
      </header>

      <section aria-labelledby="pipeline">
        <h2 id="pipeline">Pipeline</h2>
        <ol className="pipeline">
          {PIPELINE.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="cap1" className="card">
        <h2 id="cap1">CAP-1 · Calibration-efficient motor intent</h2>
        <p>
          A new person, on an electrode layout we never trained on, gets left/right intent control with as few
          calibration trials as possible — measured as balanced accuracy at 0, 5, 10, 20 and 40 trials per class on
          held-out subjects and datasets.
        </p>
        <p className="muted">Status: evaluation harness ready; baselines pending real-data ingestion (Stage 1).</p>
      </section>

      <section aria-labelledby="stages">
        <h2 id="stages">Roadmap</h2>
        <table className="stages">
          <thead>
            <tr>
              <th scope="col">Stage</th>
              <th scope="col">Scope</th>
              <th scope="col">Exit gate</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {STAGES.map((stage) => (
              <tr key={stage.id}>
                <td>
                  {stage.id}. {stage.name}
                </td>
                <td className="muted">{stage.summary}</td>
                <td>{stage.gate}</td>
                <td>
                  <span className={`pill pill-${stage.status}`}>{stage.status}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </main>
  );
}
