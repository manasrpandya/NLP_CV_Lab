"use client";

import { color, names } from "@/lib/emotion-classification/meta";

export type occ = { parts: string[]; widx: number[]; d: number[][] };

// split into words while keeping the whitespace, so the text can be put back together exactly
export function splitWords(s: string) {
  const parts = s.split(/(\s+)/).filter((x) => x.length);
  const widx = parts.flatMap((x, i) => (/\S/.test(x) ? [i] : []));
  return { parts, widx };
}

// one variant per word with that word dropped
export function variants(parts: string[], widx: number[]) {
  return widx.map((i) =>
    parts
      .filter((_, j) => j !== i)
      .join("")
      .replace(/\s+/g, " ")
      .trim(),
  );
}

export function Explain({ o, sel, pick, top }: { o: occ; sel: number; pick: (i: number) => void; top: number[] }) {
  const ds = o.d.map((r) => r[sel]);
  const mx = Math.max(0.02, ...ds.map(Math.abs));
  const at = new Map(o.widx.map((pi, k) => [pi, k]));
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <span className="text-mute">explain</span>
        {top.map((i) => (
          <button
            key={i}
            onClick={() => pick(i)}
            className={`rounded-full border px-2.5 py-0.5 transition-colors ${
              i === sel ? "border-2 font-medium text-ink" : "border-line text-mute hover:text-ink"
            }`}
            style={i === sel ? { borderColor: color(names[i]) } : undefined}
          >
            {names[i]}
          </button>
        ))}
      </div>
      <p className="leading-8">
        {o.parts.map((x, j) => {
          const k = at.get(j);
          if (k === undefined) return <span key={j}>{x}</span>;
          const v = ds[k] / mx;
          return (
            <span
              key={j}
              title={`${v >= 0 ? "+" : ""}${(ds[k] * 100).toFixed(1)} pts ${names[sel]}`}
              className={`rounded px-0.5 py-0.5 ${v < -0.15 ? "underline decoration-dotted decoration-mute" : ""}`}
              style={v > 0 ? { background: `color-mix(in srgb, ${color(names[sel])} ${Math.round(v * 70)}%, transparent)` } : undefined}
            >
              {x}
            </span>
          );
        })}
      </p>
      <p className="mt-2 text-xs text-mute">
        shading = how much the {names[sel]} probability drops when that word is removed. dotted words push it up
        when removed (they argue against it).
      </p>
    </div>
  );
}
