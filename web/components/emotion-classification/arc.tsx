"use client";

import { useState } from "react";
import { active, color, names, rel } from "@/lib/emotion-classification/meta";

export function sentences(s: string) {
  return s
    .split(/\n+/)
    .flatMap((l) => l.split(/(?<=[.!?])\s+(?=\S)/))
    .map((x) => x.trim())
    .filter((x) => x.length > 1)
    .slice(0, 80);
}

const w = 640;
const h = 220;
const pad = { l: 30, r: 12, t: 12, b: 22 };

// emotion trajectory: one line per emotion that is strong somewhere in the text
export function Arc({ xs, ps }: { xs: string[]; ps: number[][] }) {
  const [hi, setHi] = useState<number | null>(null);
  const n = ps.length;
  const neutral = names.indexOf("neutral");
  const peak = names.map((_, i) => Math.max(...ps.map((p) => rel(p)[i])));
  const show = names
    .map((_, i) => i)
    .filter((i) => i !== neutral)
    .sort((a, b) => peak[b] - peak[a])
    .slice(0, 5);
  const X = (k: number) => pad.l + (n === 1 ? 0.5 : k / (n - 1)) * (w - pad.l - pad.r);
  const Y = (v: number) => pad.t + (1 - v) * (h - pad.t - pad.b);
  const dom = ps.map((p) => {
    const on = active(p);
    return p.reduce((b, v, i) => (on[i] && (v > p[b] || !on[b]) ? i : b), neutral);
  });

  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {show.map((i) => (
          <span key={i} className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4" style={{ background: color(names[i]) }} />
            {names[i]}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="emotion arc">
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={w - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--line)" />
            <text x={pad.l - 6} y={Y(v)} fontSize={10} textAnchor="end" dominantBaseline="middle" fill="var(--mute)">
              {v * 100}
            </text>
          </g>
        ))}
        {hi !== null && <line x1={X(hi)} x2={X(hi)} y1={pad.t} y2={h - pad.b} stroke="var(--mute)" strokeDasharray="3 3" />}
        {show.map((i) => (
          <polyline
            key={i}
            fill="none"
            stroke={color(names[i])}
            strokeWidth={2}
            strokeLinejoin="round"
            points={ps.map((p, k) => `${X(k)},${Y(p[i])}`).join(" ")}
          />
        ))}
        {ps.map((_, k) => (
          <rect
            key={k}
            x={X(k) - (w - pad.l - pad.r) / Math.max(n - 1, 1) / 2}
            width={(w - pad.l - pad.r) / Math.max(n - 1, 1)}
            y={pad.t}
            height={h - pad.t - pad.b}
            fill="transparent"
            onMouseEnter={() => setHi(k)}
            onClick={() => setHi(k)}
          />
        ))}
        {dom.map((d, k) => (
          <rect key={k} x={X(k) - 3} y={h - pad.b + 8} width={6} height={6} rx={1} fill={color(names[d])} />
        ))}
      </svg>
      <ol className="mt-3 max-h-72 space-y-1 overflow-y-auto pr-1 text-sm">
        {xs.map((x, k) => (
          <li
            key={k}
            onMouseEnter={() => setHi(k)}
            className={`grid grid-cols-[1.5rem_6.5rem_1fr] gap-2 rounded px-1 py-0.5 ${hi === k ? "bg-line/50" : ""}`}
          >
            <span className="font-mono text-xs text-mute">{k + 1}</span>
            <span className="text-xs font-medium" style={{ color: color(names[dom[k]]) }}>
              {names[dom[k]]}
            </span>
            <span>{x}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
