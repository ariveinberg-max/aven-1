// Minimal EDF writer for tests (EEG data must never be committed, so fixtures are generated).

function field(value: string | number, width: number): string {
  return String(value).padEnd(width, " ").slice(0, width);
}

export function syntheticEdf(labels: string[], sfreq: number, seconds: number): Buffer {
  const ns = labels.length;
  const header =
    field("0", 8) +
    field("X X X X", 80) +
    field("Startdate 01-JAN-2026 X X X", 80) +
    field("01.01.26", 8) +
    field("00.00.00", 8) +
    field(256 + ns * 256, 8) +
    field("", 44) +
    field(seconds, 8) +
    field(1, 8) +
    field(ns, 4) +
    labels.map((l) => field(l, 16)).join("") +
    labels.map(() => field("AgAgCl electrode", 80)).join("") +
    labels.map(() => field("uV", 8)).join("") +
    labels.map(() => field(-500, 8)).join("") +
    labels.map(() => field(500, 8)).join("") +
    labels.map(() => field(-32768, 8)).join("") +
    labels.map(() => field(32767, 8)).join("") +
    labels.map(() => field("", 80)).join("") +
    labels.map(() => field(sfreq, 8)).join("") +
    labels.map(() => field("", 32)).join("");
  const samples = sfreq; // one-second records
  const body = Buffer.alloc(seconds * ns * samples * 2);
  let offset = 0;
  for (let record = 0; record < seconds; record++) {
    for (let ch = 0; ch < ns; ch++) {
      for (let i = 0; i < samples; i++) {
        const t = record + i / sfreq;
        const microvolts = ch === ns - 1 ? 0 : 20 * Math.sin(2 * Math.PI * 10 * t + ch) + 5 * Math.sin(2 * Math.PI * 3 * t);
        body.writeInt16LE(Math.round((microvolts / 500) * 32767), offset);
        offset += 2;
      }
    }
  }
  return Buffer.concat([Buffer.from(header, "ascii"), body]);
}
