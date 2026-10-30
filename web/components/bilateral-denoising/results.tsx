"use client";

import { useState } from "react";
import res from "@/lib/bilateral-denoising/results.json";

const sets = [
  ["set12", "Set12"],
  ["bsd68", "BSD68"],
  ["kodak24", "Kodak24"],
] as const;
const sigs = [15, 25, 50];
const plotKeys = ["bf_oracle", "bf_joint", "nbf", "bm3d", "dncnn"];

type row = (typeof res.rows)[number];
const get = (r: row, k: string) => (r.psnr as Record<string, number | undefined>)[k];

function Plot({ ds }: { ds: string }) {
  const rows = res.rows.filter((r) => plotKeys.includes(r.key));
  const w = 320;
  const h = 200;
  const m = { l: 34, r: 10, t: 10, b: 26 };
  const ys = rows.flatMap((r) => sigs.map((s) => get(r, `${ds}/${s}`) ?? NaN)).filter((v) => !isNaN(v));
  const lo = Math.floor(Math.min(...ys)) - 0.5;
  const hi = Math.ceil(Math.max(...ys)) + 0.5;
  const px = (s: number) => m.l + ((s - 15) / 35) * (w - m.l - m.r);
  const py = (v: number) => m.t + (1 - (v - lo) / (hi - lo)) * (h - m.t - m.b);
  const ticks = [];
  for (let v = Math.ceil(lo); v <= hi; v += 2) ticks.push(v);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-md" role="img" aria-label="psnr vs noise sigma">
      {ticks.map((v) => (
        <g key={v}>
          <line x1={m.l} x2={w - m.r} y1={py(v)} y2={py(v)} className="stroke-line" />
          <text x={m.l - 4} y={py(v) + 3} textAnchor="end" className="fill-mute font-mono text-[9px]">
            {v}
          </text>
        </g>
      ))}
      {sigs.map((s) => (
        <text key={s} x={px(s)} y={h - 8} textAnchor="middle" className="fill-mute font-mono text-[9px]">
          σ={s}
        </text>
      ))}
      {rows.map((r) => {
        const ours = r.key === "nbf";
        const cls = ours ? "stroke-acc" : r.group === "e" ? "stroke-mute" : "stroke-ink";
        const pts = sigs.map((s) => [px(s), py(get(r, `${ds}/${s}`) ?? lo)]);
        return (
          <g key={r.key}>
            <polyline
              points={pts.map((p) => p.join(",")).join(" ")}
              fill="none"
              strokeWidth={ours ? 2.2 : 1.2}
              strokeDasharray={r.group === "e" ? "4 3" : undefined}
              className={cls}
            />
            <text x={pts[2][0] - 4} y={pts[2][1] - 4} textAnchor="end" className={`font-mono text-[8.5px] ${ours ? "fill-acc" : "fill-mute"}`}>
              {r.key}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function Results() {
  const [ds, setDs] = useState<string>("bsd68");
  return (
    <section className="mt-16">
      <h2 className="font-serif text-3xl tracking-tight">results</h2>
      <p className="mt-2 max-w-2xl text-sm text-mute">
        PSNR in dB on grayscale test sets with gaussian noise (fixed seed), and on 320 real-noise crops from the SIDD
        validation set. methods (a)-(d) are bilateral variants, (e) are reference denoisers for context.
      </p>
      <div className="mt-4 inline-flex overflow-hidden rounded-md border border-line text-sm">
        {sets.map(([k, l]) => (
          <button key={k} onClick={() => setDs(k)} className={`px-3 py-1.5 ${ds === k ? "bg-ink text-bg" : "bg-card"}`}>
            {l}
          </button>
        ))}
      </div>
      <div className="mt-4 grid gap-8 lg:grid-cols-[1fr_auto]">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[420px] text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs text-mute">
                <th className="py-1.5 pr-2 font-normal"></th>
                <th className="py-1.5 pr-2 font-normal">method</th>
                {sigs.map((s) => (
                  <th key={s} className="py-1.5 pr-2 text-right font-normal">
                    σ={s}
                  </th>
                ))}
                <th className="py-1.5 text-right font-normal">SIDD</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {res.rows.map((r) => (
                <tr key={r.key} className={`border-b border-line/60 ${r.key === "nbf" ? "text-acc" : ""}`}>
                  <td className="py-1.5 pr-2 text-xs text-mute">{r.group}</td>
                  <td className="py-1.5 pr-2 font-sans">{r.label}</td>
                  {sigs.map((s) => (
                    <td key={s} className="py-1.5 pr-2 text-right">
                      {get(r, `${ds}/${s}`)?.toFixed(2) ?? "-"}
                    </td>
                  ))}
                  <td className="py-1.5 text-right">{get(r, "sidd/0")?.toFixed(2) ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Plot ds={ds} />
      </div>
    </section>
  );
}
