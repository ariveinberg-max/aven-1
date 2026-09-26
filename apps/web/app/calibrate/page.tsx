import type { Metadata } from "next";

import { CalibrationGame } from "@/components/CalibrationGame";

export const metadata: Metadata = { title: "Calibration game · neurolayer" };

export default function CalibratePage() {
  return (
    <main className="container">
      <h1>Calibration game</h1>
      <p className="muted">
        A two-minute cue-based calibration that produces CAP-1-compatible trials, then live decoding with your
        personalized model.
      </p>
      <CalibrationGame />
    </main>
  );
}
