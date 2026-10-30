"use client";

import { useState } from "react";
import res from "@/lib/bilateral-denoising/results.json";
import { More, Tip } from "./tip";

const sets = [
  ["set12", "Set12"],
  ["bsd68", "BSD68"],
  ["kodak24", "Kodak24"],
] as const;
const sigs = [15, 25, 50];
const plotKeys = ["bf_oracle", "bf_joint", "nbf", "bm3d", "dncnn"];
// stroke class + dash per plotted method
const sty: Record<string, [string, string | undefined, string]> = {
  bf_oracle: ["stroke-ink", undefined, "border-ink border-solid"],
  bf_joint: ["stroke-ink", "1.5 2.5", "border-ink border-dotted"],
  nbf: ["stroke-acc", undefined, "border-acc border-solid"],
  bm3d: ["stroke-mute", "5 3", "border-mute border-dashed"],
  dncnn: ["stroke-mute", "1.5 2.5", "border-mute border-dotted"],
};

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
        const pts = sigs.map((s) => [px(s), py(get(r, `${ds}/${s}`) ?? lo)]);
        return (
          <g key={r.key}>
            <polyline
              points={pts.map((p) => p.join(",")).join(" ")}
              fill="none"
              strokeWidth={ours ? 2.2 : 1.2}
              strokeDasharray={sty[r.key][1]}
              className={sty[r.key][0]}
            />
          </g>
        );
      })}
    </svg>
  );
}

function Legend() {
  return (
    <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-mute">
      {res.rows
        .filter((r) => plotKeys.includes(r.key))
        .map((r) => (
          <li key={r.key} className="flex items-center gap-1.5">
            <span
              className={`inline-block w-5 border-t-2 ${sty[r.key][2]}`}
            />
            {r.label}
          </li>
        ))}
    </ul>
  );
}

function Full() {
  const [ds, setDs] = useState<string>("bsd68");
  return (
    <div>
      <p className="max-w-2xl">
        PSNR in dB on grayscale test sets with gaussian noise (fixed seed), and on 320 real-noise crops from the SIDD
        validation set. (a)-(d) are bilateral variants, (e) are reference denoisers. the last three (d) rows are
        ablations of the network.
      </p>
      <div className="mt-4 inline-flex overflow-hidden rounded-md border border-line text-sm">
        {sets.map(([k, l]) => (
          <button key={k} onClick={() => setDs(k)} className={`px-3 py-1.5 ${ds === k ? "bg-ink text-bg" : "bg-card"}`}>
            {l}
          </button>
        ))}
      </div>
      <div className="mt-4 grid gap-8 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
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
        <div>
          <Plot ds={ds} />
          <Legend />
        </div>
      </div>
    </div>
  );
}

// short names for the summary bars
const plain: [string, string][] = [
  ["noisy", "noisy photo"],
  ["bf_oracle", "classic, hand-tuned"],
  ["bf_auto", "classic, auto"],
  ["nbf", "neural (this page)"],
  ["bm3d", "BM3D"],
  ["dncnn", "DnCNN"],
];

const refNote: Record<string, string> = {
  bf_oracle: "the best fixed settings, found by trying every combination. it is told the true noise level.",
  bm3d: "a well-known non-neural denoiser from 2007, a common yardstick. slow (about 5 s per image here).",
  dncnn: "a standard denoising network with 7 times more weights and no bilateral structure, for reference.",
};

function Bars({ real }: { real: boolean }) {
  const k = real ? "sidd/0" : "bsd68/25";
  const rows = plain.map(([key, name]) => ({ key, name, v: get(res.rows.find((r) => r.key === key)!, k) ?? 0 }));
  const lo = real ? 25 : 18;
  const hi = real ? 34 : 30;
  return (
    <ul className="space-y-2">
      {rows.map((r) => (
        <li key={r.key} className="grid grid-cols-[9.5rem_1fr] items-center gap-3 text-sm sm:grid-cols-[11rem_1fr]">
          <span className={`flex items-center gap-1.5 ${r.key === "nbf" ? "font-medium text-acc" : ""}`}>
            {r.name}
            {refNote[r.key] && <Tip label={`about ${r.name}`}>{refNote[r.key]}</Tip>}
          </span>
          <span className="flex items-center gap-2">
            <span
              className={`h-4 rounded-sm transition-all duration-500 ${r.key === "nbf" ? "bg-acc" : r.key === "noisy" ? "bg-line" : "bg-mute/50"}`}
              style={{ width: `${Math.max(2, ((r.v - lo) / (hi - lo)) * 100)}%` }}
            />
            <span className="font-mono text-xs">{r.v.toFixed(1)}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}

export function Results() {
  const [real, setReal] = useState(false);
  return (
    <section className="mt-16">
      <h2 className="font-serif text-3xl tracking-tight">how well does it work?</h2>
      <p className="mt-2 max-w-2xl text-sm text-mute">
        average quality score (PSNR, higher is better) on standard test photos.
      </p>
      <div className="mt-4 inline-flex overflow-hidden rounded-md border border-line text-sm">
        {(
          [
            [false, "added noise"],
            [true, "real camera noise"],
          ] as const
        ).map(([k, l]) => (
          <button key={l} onClick={() => setReal(k)} className={`px-3 py-1.5 ${real === k ? "bg-ink text-bg" : "bg-card"}`}>
            {l}
          </button>
        ))}
      </div>
      <div className="mt-5 max-w-2xl">
        <Bars real={real} />
      </div>
      <p className="mt-4 max-w-2xl text-sm">
        {real
          ? "on real phone photos (SIDD) the ranking flips: the classic filters win, because the network only ever saw artificial grain while training, and real camera noise looks different."
          : "the neural version beats every classic setting by about 1.4 dB and comes close to BM3D, while staying a filter that only averages nearby pixels. a bigger plain network (DnCNN) is still better."}
      </p>
      <div className="mt-6 max-w-5xl border-t border-line">
        <More title="full results table and plot">
          <Full />
        </More>
      </div>
    </section>
  );
}
