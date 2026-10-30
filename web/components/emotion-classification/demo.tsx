"use client";

import { useEffect, useRef, useState } from "react";
import { active, huecss, names, probs } from "@/lib/emotion-classification/meta";
import { Arc, sentences } from "./arc";
import { Bars } from "./bars";
import { Explain, splitWords, variants, type occ } from "./explain";
import { Tip } from "./tip";
import { useModel, type mstate } from "./use-model";
import { Wheel } from "./wheel";

const chat = `hey are you coming tonight?
i thought we agreed on 8. it's already 9 and nobody told me anything.
sorry!! my train got cancelled and my phone died. i feel terrible.
ok that's not your fault, i'm just tired and a bit grumpy.
let me make it up to you, dinner on me this weekend?
haha deal. thank you, that's really sweet of you.
also guess what, i got the job!!!
no way, that's amazing! i'm so proud of you`;

const diary = `woke up late again and missed the bus. great start.
the exam was way harder than i expected and i blanked on the last question.
honestly i'm scared i failed it.
then sam showed up at lunch with my favourite cake, i had completely forgotten it was my birthday.
we laughed about the exam the whole afternoon.
not sure how i did, but today ended a lot better than it started.`;

type ex = { name: string; x: string; arc?: boolean };
const examples: ex[] = [
  { name: "thank you", x: "thank you so much for helping me move, honestly couldn't have done it without you" },
  { name: "cancelled show", x: "i can't believe they cancelled the show after that cliffhanger. so annoying" },
  { name: "a prank?", x: "wait, you're telling me the whole thing was a prank all along??" },
  { name: "lost a pet", x: "my dog passed away this morning. the house feels so empty" },
  { name: "interview", x: "i have my first job interview tomorrow and i can't stop shaking lol" },
  { name: "gross food", x: "that is honestly the grossest thing i've ever seen someone eat" },
  { name: "a chat", x: chat, arc: true },
  { name: "a diary entry", x: diary, arc: true },
];

type mode = "one" | "arc";
const modes: { k: mode; name: string; sub: string; best?: boolean }[] = [
  { k: "one", name: "emotions", sub: "what the whole text feels like", best: true },
  { k: "arc", name: "emotion arc", sub: "how it changes line by line" },
];
const about: Record<mode, string> = {
  one: "reads the text as one message and shows every emotion it picks up, plus which words gave it away.",
  arc: "splits the text into sentences and lines, reads each one separately and draws how the feelings move.",
};

const maxWords = 80;

function Step({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <p className="mb-2 flex items-center gap-2 text-sm">
      <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-ink font-mono text-[11px] text-bg">
        {n}
      </span>
      {children}
    </p>
  );
}

function Loading({ st }: { st: mstate }) {
  if (st.phase === "error")
    return <p className="p-6 text-sm text-acc">the model could not be loaded: {st.msg}. try reloading the page.</p>;
  const f = st.total ? st.loaded / st.total : 0;
  return (
    <div className="flex min-h-48 flex-col items-center justify-center gap-3 p-6 text-sm text-mute">
      <div className="h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-line">
        <div className="h-full bg-acc transition-[width]" style={{ width: `${Math.max(3, f * 100)}%` }} />
      </div>
      <p className="text-center">
        {st.total
          ? `downloading the model ${Math.round(f * 100)}% (${(st.total / 2 ** 20).toFixed(0)} mb, only the first time)`
          : "starting the model (only slow the first time)"}
      </p>
    </div>
  );
}

export function Demo() {
  const { st, run } = useModel();
  const [mode, setMode] = useState<mode>("one");
  const [x, setX] = useState(examples[0].x);
  const [p, setP] = useState<number[] | null>(null);
  const [o, setO] = useState<occ | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [arc, setArc] = useState<{ xs: string[]; ps: number[][] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [ms, setMs] = useState(0);
  const req = useRef(0);
  const ready = st.phase === "ready";

  useEffect(() => {
    if (!ready) return;
    const id = ++req.current;
    const t = setTimeout(async () => {
      const s = x.trim();
      if (!s) return;
      setBusy(true);
      const t0 = performance.now();
      if (mode === "arc") {
        const xs = sentences(s);
        const ps = probs(await run(xs));
        if (id !== req.current) return;
        setArc({ xs, ps });
        setMs(performance.now() - t0);
        setBusy(false);
        return;
      }
      const pf = probs(await run([s]))[0];
      if (id !== req.current) return;
      setP(pf);
      setMs(performance.now() - t0);
      // occlusion: rerun with each word dropped, all in one batch
      const { parts, widx } = splitWords(s);
      if (widx.length < 2 || widx.length > maxWords) {
        setO(null);
        setBusy(false);
        return;
      }
      const pv = probs(await run(variants(parts, widx)));
      if (id !== req.current) return;
      // log-odds instead of raw probability, otherwise a saturated 99% label shows no word effects at all
      const lo = (v: number) => Math.log(Math.max(v, 1e-6) / Math.max(1 - v, 1e-6));
      setO({ parts, widx, d: pv.map((q) => q.map((v, i) => lo(pf[i]) - lo(v))) });
      setBusy(false);
    }, 250);
    return () => clearTimeout(t);
  }, [x, ready, mode, run]);

  const top = p
    ? names
        .map((_, i) => i)
        .sort((a, b) => p[b] - p[a])
        .slice(0, 4)
    : [];
  const shown = sel !== null && top.includes(sel) ? sel : top[0];
  const on = p ? active(p) : [];
  const found = p
    ? names
        .map((n, i) => ({ n, v: p[i] }))
        .filter((_, i) => on[i])
        .sort((a, b) => b.v - a.v)
    : [];

  return (
    <section>
      <style>{huecss}</style>
      <Step n={1}>write something, or pick an example</Step>
      <textarea
        value={x}
        onChange={(e) => setX(e.target.value)}
        rows={mode === "arc" ? 7 : 3}
        placeholder="type a message, a tweet, a few lines of a chat..."
        className="w-full resize-y rounded-lg border border-line bg-card p-3 text-base outline-none focus:border-mute"
      />
      <div className="mt-2 flex flex-wrap gap-2">
        {examples.map((e) => (
          <button
            key={e.name}
            onClick={() => {
              setX(e.x);
              setMode(e.arc ? "arc" : "one");
            }}
            className={`rounded-full border px-3 py-1 text-xs hover:text-ink ${
              x === e.x ? "border-acc text-ink" : "border-line text-mute"
            }`}
          >
            {e.name}
          </button>
        ))}
      </div>

      <div className="mt-6">
        <Step n={2}>
          pick what to show
          <Tip label="what is the difference">
            &ldquo;emotions&rdquo; treats everything as one message. &ldquo;emotion arc&rdquo; is for longer text like a
            chat log or a diary: every sentence gets its own reading, so you can see the mood shift.
          </Tip>
        </Step>
        <div className="grid grid-cols-2 gap-2">
          {modes.map((m) => (
            <button
              key={m.k}
              onClick={() => setMode(m.k)}
              aria-pressed={mode === m.k}
              className={`relative rounded-lg border-2 p-2.5 text-left transition-colors sm:p-3 ${
                mode === m.k ? "border-acc bg-acc/10" : "border-line bg-card hover:border-mute"
              }`}
            >
              <span className="block text-sm font-medium sm:text-base">{m.name}</span>
              <span className="mt-0.5 block text-[11px] leading-snug text-mute sm:text-xs">{m.sub}</span>
              {m.best && (
                <span className="absolute top-2 right-2 hidden rounded-sm bg-acc px-1 font-mono text-[10px] text-bg sm:inline">
                  start here
                </span>
              )}
            </button>
          ))}
        </div>
        <p className="mt-2 text-sm text-mute">{about[mode]}</p>
      </div>

      <div className="mt-5 flex h-4 items-center justify-end gap-1.5 font-mono text-xs text-mute">
        {ready && (
          <>
            {busy ? "reading..." : `${ms.toFixed(0)} ms`} · {st.backend.startsWith("webgpu") ? "on your gpu" : "on your cpu"}
            <Tip label="where it runs">
              the model ({st.backend}) runs inside this browser tab, nothing you type is sent anywhere. with webgpu it
              uses your graphics card, otherwise an 8-bit version (79 mb) runs on the cpu with webassembly.
            </Tip>
          </>
        )}
      </div>

      {!ready ? (
        <div className="mt-2 rounded-lg border border-line bg-card">
          <Loading st={st} />
        </div>
      ) : mode === "one" ? (
        p && (
          <div className={`mt-2 space-y-6 transition-opacity ${busy ? "opacity-70" : ""}`}>
            <div className="rounded-lg border border-line bg-card p-4">
              <p className="text-sm text-mute">this sounds like</p>
              <p className="mt-1 font-serif text-3xl leading-tight">
                {found.map((d, k) => (
                  <span key={d.n}>
                    {k > 0 && <span className="text-mute">{k === found.length - 1 ? " and " : ", "}</span>}
                    {d.n}
                  </span>
                ))}
              </p>
              <p className="mt-1 font-mono text-xs text-mute">
                {found.map((d) => `${d.n} ${(d.v * 100).toFixed(0)}%`).join(" · ")}
              </p>
            </div>

            <div className="rounded-lg border border-line bg-card p-4">
              <h3 className="mb-3 flex items-center gap-2 font-serif text-xl">
                which words gave it away
                <Tip label="how this works">
                  the text is read again once for every word, with that word deleted. the more the emotion drops without
                  a word, the stronger its shading. dotted words point the other way. all the variants run in one batch,
                  so it stays fast.
                </Tip>
              </h3>
              {o ? (
                <Explain o={o} sel={shown} pick={setSel} top={top} />
              ) : (
                <p className="text-sm text-mute">works for 2 to {maxWords} words.</p>
              )}
            </div>

            <div className="grid gap-6 md:grid-cols-[1fr_300px]">
              <div className="rounded-lg border border-line bg-card p-4">
                <h3 className="mb-3 flex items-center gap-2 font-serif text-xl">
                  all 28 emotions
                  <Tip label="what the bars mean">
                    each bar is how likely that emotion is. the percentages are calibrated: of all the times it says
                    70%, the emotion is really there about 70% of the time. the thin line is the cutoff for that emotion,
                    picked on held-out data. rare ones like grief need a much lower cutoff than common ones like
                    gratitude.
                  </Tip>
                </h3>
                <Bars p={p} />
              </div>
              <div className="self-start rounded-lg border border-line bg-card p-4">
                <h3 className="mb-1 flex items-center gap-2 font-serif text-xl">
                  the big six
                  <Tip label="what is this wheel">
                    psychologist paul ekman&apos;s six basic emotions. each of the 27 fine emotions belongs to one of
                    them and is drawn as a spoke in that slice. the shaded petal is the strongest emotion in the group,
                    neutral sits in the middle.
                  </Tip>
                </h3>
                <Wheel p={p} />
              </div>
            </div>
          </div>
        )
      ) : (
        arc && (
          <div className={`mt-2 rounded-lg border border-line bg-card p-4 transition-opacity ${busy ? "opacity-70" : ""}`}>
            <h3 className="mb-3 flex items-center gap-2 font-serif text-xl">
              how the feelings move
              <Tip label="how to read this">
                one line per emotion that is strong somewhere in the text, sentence by sentence from left to right. the
                squares under the chart and the list below show the main emotion of each sentence. hover or tap to
                highlight one.
              </Tip>
            </h3>
            <Arc xs={arc.xs} ps={arc.ps} />
          </div>
        )
      )}
    </section>
  );
}
