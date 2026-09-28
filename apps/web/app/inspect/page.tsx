import type { Metadata } from "next";

import { Inspector } from "@/components/Inspector";

export const metadata: Metadata = { title: "Inspect a recording · neurolayer" };

export default function InspectPage() {
  return (
    <main className="container">
      <h1>Inspect a recording</h1>
      <p className="muted">Channels, sampling rate, events, band power and quality flags for an EDF or BDF file.</p>
      <Inspector />
    </main>
  );
}
