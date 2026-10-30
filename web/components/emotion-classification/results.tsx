"use client";

import { useState } from "react";
import { data } from "@/lib/emotion-classification/meta";
import { More, Tip } from "./tip";

type row = {
  k: string;
  label: string;
  macro_f1: number;
  macro_f1_ci: number[];
  micro_f1: number;
  ekman_macro_f1: number;
  ood_acc: number;
  ece_raw: number;
  ece_cal: number;
  mb: number;
};
const rows = data.rows as row[];
const get = (k: string) => rows.find((r) => r.k === k);

// short plain names for the summary bars. paper = the GoEmotions paper's BERT baseline (no ood number)
const plain: [string, string, string?][] = [
  ["tfidf", "word counting", "tf-idf + logistic regression: counts which words and letter groups appear. no deep learning."],
  ["bigru", "older neural net", "a recurrent network (bigru) on top of pretrained glove word vectors, the pre-transformer way."],
  ["paper", "original paper", "the BERT-base model from the GoEmotions paper (2020), the usual number to beat."],
  ["student-int8", "this page (79 mb)"],
  ["teacher-ens", "big model (2.7 gb)", "two roberta-large models averaged. the small model on this page learned from them."],
];
const paper: Record<string, number> = { macro_f1: 0.46, ekman_macro_f1: 0.64 };

type view = "macro_f1" | "ekman_macro_f1" | "ood_acc";
const views: [view, string][] = [
  ["macro_f1", "28 emotions"],
  ["ekman_macro_f1", "the big six"],
  ["ood_acc", "tweets, never seen"],
];
const sub: Record<view, string> = {
  macro_f1: "score on 5,427 held-out reddit comments, averaged over all 28 emotions so rare ones count as much as common ones.",
  ekman_macro_f1: "the same comments, but only asking for the big six (plus neutral), which is an easier question.",
  ood_acc: "share of 2,000 tweets from a different dataset where the main emotion is right. the models never trained on tweets.",
};
const take: Record<view, string> = {
  macro_f1:
    "the small model on this page scores about the same as the big one it learned from, while being 17x smaller and about 45x faster than a single big model, and both clearly beat the original paper.",
  ekman_macro_f1: "on the coarse version of the question every transformer lands close together, just above the paper's 0.64.",
  ood_acc:
    "on tweets it gets the main emotion right about 63% of the time with no extra training. word counting drops to 45%, it relies on reddit-specific words.",
};

function Bars({ v }: { v: view }) {
  const xs = plain
    .map(([k, name, note]) => ({ k, name, note, s: k === "paper" ? paper[v] : get(k)?.[v] }))
    .filter((d) => d.s !== undefined) as { k: string; name: string; note?: string; s: number }[];
  const lo = v === "ood_acc" ? 0.3 : 0.35;
  const hi = 0.7;
  return (
    <ul className="space-y-2">
      {xs.map((d) => (
        <li key={d.k} className="grid grid-cols-[9.5rem_1fr] items-center gap-3 text-sm sm:grid-cols-[11rem_1fr]">
          <span className={`flex items-center gap-1.5 ${d.k === "student-int8" ? "font-medium text-acc" : ""}`}>
            {d.name}
            {d.note && <Tip label={`about ${d.name}`}>{d.note}</Tip>}
          </span>
          <span className="flex items-center gap-2">
            <span
              className={`h-4 rounded-sm transition-all duration-500 ${
                d.k === "student-int8" ? "bg-acc" : d.k === "paper" ? "bg-line" : "bg-mute/50"
              }`}
              style={{ width: `${Math.max(2, ((d.s - lo) / (hi - lo)) * 100)}%` }}
            />
            <span className="font-mono text-xs">{d.s.toFixed(2)}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}

const f = (v: number, d = 3) => v.toFixed(d);

function Full() {
  return (
    <div>
      <p className="max-w-2xl">
        GoEmotions test set, 28 labels. every model gets its own temperature and per-label thresholds tuned on the
        validation set. ci = 95% bootstrap interval over test comments. ood = zero-shot accuracy on the dair-ai/emotion
        twitter test set after mapping both label sets to ekman groups. ece = calibration error before / after
        temperature scaling.
      </p>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[680px] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs text-mute">
              <th className="py-1.5 pr-2 font-normal">model</th>
              <th className="py-1.5 pr-2 text-right font-normal">macro-f1</th>
              <th className="py-1.5 pr-2 text-right font-normal">ci</th>
              <th className="py-1.5 pr-2 text-right font-normal">micro-f1</th>
              <th className="py-1.5 pr-2 text-right font-normal">ekman</th>
              <th className="py-1.5 pr-2 text-right font-normal">ood acc</th>
              <th className="py-1.5 pr-2 text-right font-normal">ece</th>
              <th className="py-1.5 text-right font-normal">mb</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((r) => (
              <tr key={r.k} className={`border-b border-line/60 ${r.k === "student-int8" ? "text-acc" : ""}`}>
                <td className="py-1.5 pr-2 font-sans">{r.label}</td>
                <td className="py-1.5 pr-2 text-right">{f(r.macro_f1)}</td>
                <td className="py-1.5 pr-2 text-right text-xs text-mute">
                  {f(r.macro_f1_ci[0], 2)}-{f(r.macro_f1_ci[1], 2)}
                </td>
                <td className="py-1.5 pr-2 text-right">{f(r.micro_f1)}</td>
                <td className="py-1.5 pr-2 text-right">{f(r.ekman_macro_f1)}</td>
                <td className="py-1.5 pr-2 text-right">{f(r.ood_acc)}</td>
                <td className="py-1.5 pr-2 text-right">
                  {f(r.ece_raw)} / {f(r.ece_cal)}
                </td>
                <td className="py-1.5 text-right">{r.mb ? r.mb.toFixed(0) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function Results() {
  const [v, setV] = useState<view>("macro_f1");
  if (!rows.length) return null;
  return (
    <section className="mt-16">
      <h2 className="flex items-center gap-2 font-serif text-3xl tracking-tight">
        how well does it work?
        <Tip label="what is the score">
          the scores are f1: a mix of &ldquo;when it says sadness, is it right?&rdquo; and &ldquo;does it find the
          sadness that is there?&rdquo;. 1 is perfect. people often disagree on emotions too, so nothing gets close to 1
          on this data.
        </Tip>
      </h2>
      <div className="mt-4 inline-flex overflow-hidden rounded-md border border-line text-sm">
        {views.map(([k, l]) => (
          <button key={k} onClick={() => setV(k)} className={`px-3 py-1.5 ${v === k ? "bg-ink text-bg" : "bg-card"}`}>
            {l}
          </button>
        ))}
      </div>
      <p className="mt-3 max-w-2xl text-sm text-mute">{sub[v]}</p>
      <div className="mt-5 max-w-2xl">
        <Bars v={v} />
      </div>
      <p className="mt-4 max-w-2xl text-sm">{take[v]}</p>
      <div className="mt-6 max-w-5xl border-t border-line">
        <More title="full results table">
          <Full />
        </More>
      </div>
    </section>
  );
}
