"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Traces } from "@/components/Traces";
import { api, ApiError, LABEL_TEXT, type DemoTrials, type ModelInfo, type Trial } from "@/lib/api";

// WP-7.2: the same cue → imagery protocol as the pilot recorder and CAP-1, played
// against the API's synthetic demo "headset". Before calibration we decode the play
// trials once (k = 0) so the gain from calibration is visible at the end.

type Phase = "setup" | "loading" | "calibrating" | "adapting" | "playing" | "done" | "error";

interface Outcome {
  truth: string;
  predicted: string;
  confidence: number;
}

const BUDGETS = [5, 10, 20];
const PLAY_TRIALS = 20;
const SPEEDS = { normal: 1100, fast: 120 } as const;
const SHOWN_CHANNELS = ["C3", "Cz", "C4"];

function split(demo: DemoTrials, perClass: number): { calibration: Trial[]; play: Trial[] } {
  const counts: Record<string, number> = {};
  const calibration: Trial[] = [];
  const play: Trial[] = [];
  for (const trial of demo.trials) {
    const label = trial.label ?? "";
    if ((counts[label] ?? 0) < perClass) {
      counts[label] = (counts[label] ?? 0) + 1;
      calibration.push(trial);
    } else if (play.length < PLAY_TRIALS) {
      play.push(trial);
    }
  }
  return { calibration, play };
}

function accuracy(outcomes: Outcome[]): number {
  if (outcomes.length === 0) return 0;
  return outcomes.filter((o) => o.truth === o.predicted).length / outcomes.length;
}

function Arrow({ label }: { label: string | null }) {
  if (!label) return <div className="cue cue-rest" aria-hidden="true">+</div>;
  return (
    <div className="cue" aria-hidden="true">
      {label === "left_hand" ? "←" : "→"}
    </div>
  );
}

export function CalibrationGame() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState("");
  const [perClass, setPerClass] = useState(10);
  const [speed, setSpeed] = useState<keyof typeof SPEEDS>("normal");
  const [phase, setPhase] = useState<Phase>("setup");
  const [error, setError] = useState("");
  const [demo, setDemo] = useState<DemoTrials | null>(null);
  const [calibration, setCalibration] = useState<Trial[]>([]);
  const [play, setPlay] = useState<Trial[]>([]);
  const [index, setIndex] = useState(0);
  const [baseline, setBaseline] = useState<Outcome[]>([]);
  const [outcomes, setOutcomes] = useState<Outcome[]>([]);
  const session = useRef<string | null>(null);
  const busy = useRef(false);

  const fail = useCallback((err: unknown) => {
    const message =
      err instanceof ApiError && err.status === 404 && String(err.message).includes("demo")
        ? "Demo data is disabled on this API. Start it with NEUROLAYER_DEMO=1 (make api)."
        : err instanceof Error
          ? err.message
          : String(err);
    setError(message);
    setPhase("error");
  }, []);

  useEffect(() => {
    api
      .models()
      .then((list) => {
        setModels(list);
        const preferred = list.find((m) => m.synthetic_only) ?? list[0];
        if (preferred) setModel(preferred.id);
      })
      .catch(fail);
  }, [fail]);

  useEffect(
    () => () => {
      if (session.current) void api.deleteSession(session.current).catch(() => undefined);
    },
    [],
  );

  async function start() {
    setPhase("loading");
    setError("");
    setIndex(0);
    setOutcomes([]);
    try {
      if (session.current) {
        const previous = session.current;
        session.current = null;
        await api.deleteSession(previous).catch(() => undefined);
      }
      const trials = await api.demoTrials(model, 30, Math.floor(Math.random() * 1000));
      const parts = split(trials, perClass);
      const created = await api.createSession(model, trials.ch_names, trials.sfreq);
      session.current = created.id;
      const zeroShot = await api.decode(created.id, parts.play);
      setBaseline(
        zeroShot.predictions.map((p, i) => ({
          truth: parts.play[i]?.label ?? "",
          predicted: p.label,
          confidence: Math.max(...Object.values(p.probabilities)),
        })),
      );
      setDemo(trials);
      setCalibration(parts.calibration);
      setPlay(parts.play);
      setPhase("calibrating");
    } catch (err) {
      fail(err);
    }
  }

  const step = useCallback(async () => {
    if (busy.current || !session.current) return;
    busy.current = true;
    try {
      if (phase === "calibrating") {
        if (index + 1 < calibration.length) {
          setIndex(index + 1);
        } else {
          setPhase("adapting");
          await api.calibrate(session.current, calibration);
          setIndex(0);
          setPhase("playing");
        }
      } else if (phase === "playing") {
        const trial = play[index];
        if (!trial) return;
        const { predictions } = await api.decode(session.current, [trial]);
        const p = predictions[0];
        if (!p) throw new Error("the API returned no prediction");
        setOutcomes((prev) => [
          ...prev,
          { truth: trial.label ?? "", predicted: p.label, confidence: Math.max(...Object.values(p.probabilities)) },
        ]);
        if (index + 1 < play.length) {
          setIndex(index + 1);
        } else {
          setPhase("done");
          const id = session.current;
          session.current = null;
          await api.deleteSession(id);
        }
      }
    } catch (err) {
      fail(err);
    } finally {
      busy.current = false;
    }
  }, [phase, index, calibration, play, fail]);

  useEffect(() => {
    if (phase !== "calibrating" && phase !== "playing") return;
    const timer = window.setTimeout(() => void step(), SPEEDS[speed]);
    return () => window.clearTimeout(timer);
  }, [phase, index, speed, step]);

  const channelIdx = demo
    ? SHOWN_CHANNELS.map((c) => demo.ch_names.indexOf(c)).filter((i) => i >= 0)
    : [];
  const current = phase === "calibrating" ? calibration[index] : phase === "playing" ? play[index] : undefined;
  const last = outcomes[outcomes.length - 1];

  return (
    <div className="game">
      {phase === "setup" || phase === "error" || phase === "loading" ? (
        <form
          className="card setup"
          onSubmit={(event) => {
            event.preventDefault();
            void start();
          }}
        >
          <p>
            Imagine squeezing your <strong>left</strong> or <strong>right</strong> hand when the arrow appears. In
            this demo a simulated headset (synthetic EEG) plays your part; with a real device the{" "}
            <code>neurolayer bridge</code> command runs the same protocol.
          </p>
          <label>
            Model{" "}
            <select value={model} onChange={(e) => setModel(e.target.value)} disabled={!models.length}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.id}
                  {m.synthetic_only ? " (synthetic demo)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label>
            Calibration trials per hand{" "}
            <select value={perClass} onChange={(e) => setPerClass(Number(e.target.value))}>
              {BUDGETS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <label>
            Speed{" "}
            <select value={speed} onChange={(e) => setSpeed(e.target.value as keyof typeof SPEEDS)}>
              <option value="normal">normal</option>
              <option value="fast">fast</option>
            </select>
          </label>
          <button type="submit" disabled={!model || phase === "loading"}>
            {phase === "loading" ? "Connecting…" : "Start calibration"}
          </button>
          {!models.length && phase !== "error" ? (
            <p className="muted">No models found yet. Train the demo bundle with <code>make demo-model</code>.</p>
          ) : null}
          {phase === "error" ? (
            <p className="error" role="alert">
              {error}
            </p>
          ) : null}
        </form>
      ) : null}

      {phase === "calibrating" || phase === "adapting" ? (
        <section className="card stage" aria-labelledby="cal-title">
          <h2 id="cal-title">
            Calibration · trial {Math.min(index + 1, calibration.length)} of {calibration.length}
          </h2>
          <Arrow label={phase === "adapting" ? null : (current?.label ?? null)} />
          <p className="instruction" aria-live="polite">
            {phase === "adapting"
              ? "Adapting the decoder to you…"
              : `Imagine: ${LABEL_TEXT[current?.label ?? ""] ?? current?.label}`}
          </p>
          <progress max={calibration.length} value={index + (phase === "adapting" ? 1 : 0)} />
          {current && demo ? (
            <Traces
              channels={channelIdx.map((i) => demo.ch_names[i] ?? "")}
              data={channelIdx.map((i) => current.data[i] ?? [])}
              label="Simulated signal for this trial"
            />
          ) : null}
        </section>
      ) : null}

      {phase === "playing" ? (
        <section className="card stage" aria-labelledby="play-title">
          <h2 id="play-title">
            Play · trial {index + 1} of {play.length}
          </h2>
          <div className="track" aria-hidden="true">
            <div
              className="ball"
              style={{
                left: last ? (last.predicted === "left_hand" ? "15%" : "85%") : "50%",
              }}
            />
          </div>
          <p className="instruction" aria-live="polite" data-testid="last-prediction">
            {last
              ? `Decoded ${LABEL_TEXT[last.predicted] ?? last.predicted} (${Math.round(last.confidence * 100)}%) · you imagined ${LABEL_TEXT[last.truth] ?? last.truth} ${last.truth === last.predicted ? "✓" : "✗"}`
              : "Decoding…"}
          </p>
          <p>
            Running accuracy: <strong data-testid="running-accuracy">{Math.round(accuracy(outcomes) * 100)}%</strong>{" "}
            over {outcomes.length} trials
          </p>
        </section>
      ) : null}

      {phase === "done" ? (
        <section className="card stage" aria-labelledby="done-title">
          <h2 id="done-title">Result</h2>
          <table className="stages">
            <tbody>
              <tr>
                <th scope="row">Before calibration (0 trials)</th>
                <td data-testid="baseline-accuracy">{Math.round(accuracy(baseline) * 100)}%</td>
              </tr>
              <tr>
                <th scope="row">After calibration ({perClass} per hand)</th>
                <td data-testid="final-accuracy">{Math.round(accuracy(outcomes) * 100)}%</td>
              </tr>
            </tbody>
          </table>
          <p className="muted">
            {outcomes.length} held-out trials from a simulated user. Synthetic data only: this shows the product
            flow, not a capability claim. The session and its calibration data were deleted on the server.
          </p>
          <button type="button" onClick={() => setPhase("setup")}>
            Play again
          </button>
        </section>
      ) : null}
    </div>
  );
}
