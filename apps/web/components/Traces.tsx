// Stacked signal traces as SVG (each channel scaled to its own range).

interface TracesProps {
  channels: string[];
  data: number[][];
  height?: number;
  label?: string;
}

const WIDTH = 600;

function path(values: number[], top: number, rowHeight: number): string {
  if (values.length === 0) return "";
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = max - min || 1;
  const step = WIDTH / Math.max(1, values.length - 1);
  return values
    .map((v, i) => {
      const x = i * step;
      const y = top + rowHeight * 0.9 - ((v - min) / span) * rowHeight * 0.8;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function Traces({ channels, data, height = 28, label = "Signal preview" }: TracesProps) {
  const total = height * data.length;
  return (
    <svg
      className="traces"
      viewBox={`-60 0 ${WIDTH + 60} ${total}`}
      role="img"
      aria-label={`${label}: ${channels.join(", ")}`}
      preserveAspectRatio="none"
    >
      {data.map((values, i) => (
        <g key={channels[i] ?? i}>
          <text x={-8} y={i * height + height * 0.6} textAnchor="end" className="trace-label">
            {channels[i]}
          </text>
          <path d={path(values, i * height, height)} className="trace" />
        </g>
      ))}
    </svg>
  );
}
