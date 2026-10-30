"use client";

import { useEffect, useRef, useState } from "react";
import { active, names, probs } from "@/lib/emotion-classification/meta";
import { Arc, sentences } from "./arc";
import { Bars } from "./bars";
import { Explain, splitWords, variants, type occ } from "./explain";
import { useModel } from "./use-model";
import { Wheel } from "./wheel";

const examples = [
  "thank you so much for helping me move, honestly couldn't have done it without you",
  "i can't believe they cancelled the show after that cliffhanger. so annoying",
  "wait, you're telling me the whole thing was a prank all along??",
  "my dog passed away this morning. the house feels so empty",
  "i have my first job interview tomorrow and i can't stop shaking lol",
  "that is honestly the grossest thing i've ever seen someone eat",
];

const chat = `hey are you coming tonight?
i thought we agreed on 8. it's already 9 and nobody told me anything.
sorry!! my train got cancelled and my phone died. i feel terrible.
ok that's not your fault, i'm just tired and a bit grumpy.
let me make it up to you, dinner on me this weekend?
haha deal. thank you, that's really sweet of you.
also guess what, i got the job!!!
no way, that's amazing! i'm so proud of you`;

const maxWords = 80;

function Loader({ loaded, total, phase, msg }: { loaded: number; total: number; phase: string; msg?: string }) {
  if (phase === "error") return <p className="text-sm text-acc">could not load the model: {msg}</p>;
  const f = total ? loaded / total : 0;
  return (
    <div className="text-sm text-mute">
      <div className="h-1.5 overflow-hidden rounded-full bg-line">
        <div className="h-full bg-acc transition-[width]" style={{ width: `${Math.max(3, f * 100)}%` }} />
      </div>
      <p className="mt-2 font-mono text-xs">
        loading model {total ? `${(loaded / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} mb` : "..."}{" "}
        (cached after the first visit)
      </p>
    </div>
  );
}

export function Demo() {
  const { st, run } = useModel();
  const [mode, setMode] = useState<"text" | "arc">("text");
  const [x, setX] = useState(examples[0]);
  const [long, setLong] = useState(chat);
  const [p, setP] = useState<number[] | null>(null);
  const [o, setO] = useState<occ | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [arc, setArc] = useState<{ xs: string[]; ps: number[][] } | null>(null);
  const [ms, setMs] = useState(0);
  const req = useRef(0);
  const ready = st.phase === "ready";

  useEffect(() => {
    if (!ready || mode !== "text") return;
    const id = ++req.current;
    const t = setTimeout(async () => {
      const s = x.trim();
      if (!s) return;
      const t0 = performance.now();
      const pf = probs(await run([s]))[0];
      if (id !== req.current) return;
      setP(pf);
      setMs(performance.now() - t0);
      // occlusion: rerun with each word dropped, all in one batch
      const { parts, widx } = splitWords(s);
      if (widx.length < 2 || widx.length > maxWords) {
        setO(null);
        return;
      }
      const pv = probs(await run(variants(parts, widx)));
      if (id !== req.current) return;
      setO({ parts, widx, d: pv.map((q) => q.map((v, i) => pf[i] - v)) });
    }, 250);
    return () => clearTimeout(t);
  }, [x, ready, mode, run]);

  useEffect(() => {
    if (!ready || mode !== "arc") return;
    const id = ++req.current;
    const t = setTimeout(async () => {
      const xs = sentences(long);
      if (!xs.length) return;
      const ps = probs(await run(xs));
      if (id === req.current) setArc({ xs, ps });
    }, 400);
    return () => clearTimeout(t);
  }, [long, ready, mode, run]);

  const on = p ? active(p) : [];
  const top = p
    ? names
        .map((_, i) => i)
        .sort((a, b) => p[b] - p[a])
        .slice(0, 4)
    : [];
  const shown = sel !== null && top.includes(sel) ? sel : top[0];

  return (
    <section className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div className="inline-flex rounded-md border border-line p-0.5 text-sm">
          {(["text", "arc"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`rounded px-3 py-1 ${mode === m ? "bg-ink text-bg" : "text-mute hover:text-ink"}`}
            >
              {m === "text" ? "single text" : "emotion arc"}
            </button>
          ))}
        </div>
        {ready && (
          <span className="font-mono text-xs text-mute">
            {st.backend}
            {mode === "text" && ms > 0 && ` · ${ms.toFixed(0)} ms`}
          </span>
        )}
      </div>

      {!ready && <Loader {...st} />}

      {mode === "text" ? (
        <>
          <div>
            <textarea
              value={x}
              onChange={(e) => setX(e.target.value)}
              rows={3}
              placeholder="type something..."
              className="w-full resize-y rounded-lg border border-line bg-card p-3 text-base outline-none focus:border-mute"
            />
            <div className="mt-2 flex flex-wrap gap-2">
              {examples.map((e) => (
                <button
                  key={e}
                  onClick={() => setX(e)}
                  className="max-w-full truncate rounded-full border border-line px-3 py-1 text-xs text-mute hover:text-ink sm:max-w-[16rem]"
                >
                  {e}
                </button>
              ))}
            </div>
          </div>

          {p && (
            <>
              <div className="flex flex-wrap gap-2">
                {names
                  .map((n, i) => ({ n, i }))
                  .filter(({ i }) => on[i])
                  .sort((a, b) => p[b.i] - p[a.i])
                  .map(({ n, i }) => (
                    <span key={n} className="rounded-md bg-card px-2.5 py-1 text-sm ring-1 ring-line">
                      {n} <span className="font-mono text-xs text-mute">{(p[i] * 100).toFixed(0)}%</span>
                    </span>
                  ))}
              </div>
              <div className="rounded-lg border border-line bg-card p-4">
                <h3 className="mb-3 font-serif text-xl">which words matter</h3>
                {o ? (
                  <Explain o={o} sel={shown} pick={setSel} top={top} />
                ) : (
                  <p className="text-sm text-mute">needs 2 to {maxWords} words.</p>
                )}
              </div>
              <div className="grid gap-6 md:grid-cols-[1fr_300px]">
                <div className="rounded-lg border border-line bg-card p-4">
                  <h3 className="mb-3 font-serif text-xl">all 28 labels</h3>
                  <Bars p={p} />
                  <p className="mt-3 text-xs text-mute">
                    calibrated probabilities. the thin tick is each label&apos;s threshold tuned on the validation
                    set; bars past it count as predicted.
                  </p>
                </div>
                <div className="self-start rounded-lg border border-line bg-card p-4">
                  <h3 className="mb-1 font-serif text-xl">ekman wheel</h3>
                  <Wheel p={p} />
                  <p className="mt-2 text-xs text-mute">
                    petals show the strongest emotion in each ekman group, spokes are the 27 fine labels.
                  </p>
                </div>
              </div>
            </>
          )}
        </>
      ) : (
        <>
          <textarea
            value={long}
            onChange={(e) => setLong(e.target.value)}
            rows={8}
            className="w-full resize-y rounded-lg border border-line bg-card p-3 font-mono text-sm outline-none focus:border-mute"
          />
          <p className="-mt-4 text-xs text-mute">
            paste a chat log, a diary entry or a story. it gets split into sentences (and lines) and each one is
            classified on its own.
          </p>
          {arc && (
            <div className="rounded-lg border border-line bg-card p-4">
              <Arc xs={arc.xs} ps={arc.ps} />
            </div>
          )}
        </>
      )}
    </section>
  );
}
