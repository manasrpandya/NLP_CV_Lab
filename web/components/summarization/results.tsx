"use client";

import { useState } from "react";
import { More, Tip } from "@/components/ui/tip";
import res from "@/lib/summarization/results.json";

type row = {
  k: string;
  label: string;
  group: string;
  rouge1: number;
  rouge2: number;
  rougeLsum: number;
  bertscore?: number;
  consistency?: number;
  novel_bigram: number;
  words: number;
  ms?: number;
  mb?: number;
};
const rows = res.rows as row[];
const get = (k: string) => rows.find((r) => r.k === k);

// plain names for the summary bars
const plain: [string, string, string?][] = [
  ["lead3", "first 3 sentences", "just copies the first three sentences of the article. a surprisingly strong baseline for news."],
  ["textrank3", "quick (textrank)", "the quick method on this page: copies the 3 most central sentences."],
  ["ours-medium-greedy", "ai summary (this page)", "our small model, run the way it runs in your browser (greedy decoding, 8-bit)."],
  ["bart-large-cnn", "big model", "facebook's bart-large-cnn, the usual choice for this task. 3x larger, about 2.5 gb in memory."],
];

type view = "rouge" | "faith";
const views: [view, string][] = [
  ["rouge", "how close to a human summary"],
  ["faith", "does it stick to the article"],
];
const sub: Record<view, string> = {
  rouge:
    "rouge-2 on 2,000 test articles the models never saw: how many two-word phrases the summary shares with the human-written highlights. higher is closer.",
  faith:
    "share of the summary's sentences that the article clearly backs up, as judged by a language-inference model, on 500 of the same articles. higher means fewer made-up details.",
};

function Bars({ v }: { v: view }) {
  const xs = plain
    .map(([k, name, note]) => {
      const r = get(k);
      return r && { k, name, note, s: v === "rouge" ? r.rouge2 : (r.consistency ?? 0) * 100 };
    })
    .filter(Boolean) as { k: string; name: string; note?: string; s: number }[];
  const hi = v === "rouge" ? Math.max(...xs.map((d) => d.s)) * 1.15 : 100;
  return (
    <ul className="space-y-2">
      {xs.map((d) => (
        <li key={d.k} className="grid grid-cols-[9.5rem_1fr] items-center gap-3 text-sm sm:grid-cols-[12rem_1fr]">
          <span className={`flex items-center gap-1.5 ${d.k.startsWith("ours") ? "font-medium text-acc" : ""}`}>
            {d.name}
            {d.note && <Tip label={`about ${d.name}`}>{d.note}</Tip>}
          </span>
          <span className="flex items-center gap-2">
            <span
              className={`h-4 rounded-sm transition-all duration-500 ${d.k.startsWith("ours") ? "bg-acc" : "bg-mute/50"}`}
              style={{ width: `${Math.max(2, (d.s / hi) * 100)}%` }}
            />
            <span className="font-mono text-xs">{d.s.toFixed(v === "rouge" ? 1 : 0)}{v === "faith" && "%"}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}

const f = (v: number | undefined, d = 1) => (v === undefined ? "" : v.toFixed(d));

function Full() {
  return (
    <div>
      <p className="max-w-2xl">
        CNN/DailyMail test set, a fixed random 2,000 articles for every model (faithfulness on the first 500). rouge is
        f1 with stemming, rouge-lsum splits the summary into sentences. bertscore uses roberta-large. faithfulness is
        the share of summary sentences with an entailing source sentence (summac-zs style, deberta-v3-xsmall). novel =
        share of summary bigrams that are not in the article. ms = cpu time per summary with 4 threads.
      </p>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs text-mute">
              <th className="py-1.5 pr-2 font-normal">method</th>
              <th className="py-1.5 pr-2 text-right font-normal">rouge-1</th>
              <th className="py-1.5 pr-2 text-right font-normal">rouge-2</th>
              <th className="py-1.5 pr-2 text-right font-normal">rouge-lsum</th>
              <th className="py-1.5 pr-2 text-right font-normal">bertscore</th>
              <th className="py-1.5 pr-2 text-right font-normal">faithful</th>
              <th className="py-1.5 pr-2 text-right font-normal">novel</th>
              <th className="py-1.5 pr-2 text-right font-normal">words</th>
              <th className="py-1.5 text-right font-normal">ms</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((r) => (
              <tr key={r.k} className={`border-b border-line/60 ${r.k.startsWith("ours") ? "text-acc" : ""}`}>
                <td className="py-1.5 pr-2 font-sans">{r.label}</td>
                <td className="py-1.5 pr-2 text-right">{f(r.rouge1)}</td>
                <td className="py-1.5 pr-2 text-right">{f(r.rouge2)}</td>
                <td className="py-1.5 pr-2 text-right">{f(r.rougeLsum)}</td>
                <td className="py-1.5 pr-2 text-right">{f(r.bertscore)}</td>
                <td className="py-1.5 pr-2 text-right">{r.consistency === undefined ? "" : f(r.consistency * 100, 0)}</td>
                <td className="py-1.5 pr-2 text-right">{f(r.novel_bigram, 0)}</td>
                <td className="py-1.5 pr-2 text-right">{f(r.words, 0)}</td>
                <td className="py-1.5 text-right">{r.ms ? r.ms.toFixed(0) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function Results() {
  const [v, setV] = useState<view>("rouge");
  return (
    <section className="mt-16 max-w-2xl">
      <h2 className="font-serif text-3xl tracking-tight">how good is it?</h2>
      <div className="mt-4 inline-flex overflow-hidden rounded-md border border-line">
        {views.map(([k, name]) => (
          <button
            key={k}
            onClick={() => setV(k)}
            aria-pressed={v === k}
            className={`px-3 py-1.5 text-sm ${v === k ? "bg-acc text-bg" : "bg-card text-mute hover:text-ink"}`}
          >
            {name}
          </button>
        ))}
      </div>
      <p className="mt-3 text-sm text-mute">{sub[v]}</p>
      <div className="mt-4 rounded-lg border border-line bg-card p-4">
        <Bars v={v} />
      </div>
      <p className="mt-3 text-sm leading-relaxed">{res.take[v]}</p>
      <div className="mt-4 border-t border-line">
        <More title="full table">
          <Full />
        </More>
      </div>
    </section>
  );
}
