"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import samples from "@/lib/summarization/samples.json";
import { origin, sents, textrank } from "@/lib/summarization/text";
import { Tip } from "@/components/ui/tip";
import { useModel } from "./use-model";

type method = "quick" | "ai";
type len = "short" | "medium" | "long";

const methods: { k: method; name: string; sub: string; best?: boolean }[] = [
  { k: "quick", name: "quick", sub: "picks key sentences, instant" },
  { k: "ai", name: "ai summary", sub: "writes its own, small model", best: true },
];
const about: Record<method, string> = {
  quick:
    "scores every sentence by how much it has in common with the others (an old algorithm called textrank) and copies the top ones. instant and nothing to download, but it can only copy.",
  ai: "a small transformer reads the start of the text and writes a new summary word by word. it downloads about 150 mb the first time, then runs in this tab.",
};
const lens: { k: len; name: string; n: number }[] = [
  { k: "short", name: "short", n: 2 },
  { k: "medium", name: "medium", n: 3 },
  { k: "long", name: "long", n: 4 },
];
// the model sees 512 tokens, about this many words of news text
const readWords = 360;

type out = { key: string; text: string; done: boolean; ms?: number; ntok?: number; method: method };

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

const nw = (x: string) => (x.trim() ? x.trim().split(/\s+/).length : 0);

// entailment cutoffs for the fact check
const ok = 0.5;
const maybe = 0.1;

export function Demo() {
  const { st, nst, gen, nli } = useModel();
  const [x, setX] = useState(samples[0].text);
  const [pick, setPick] = useState<string | null>(samples[0].id);
  const [method, setMethod] = useState<method>("ai");
  const [len, setLen] = useState<len>("medium");
  const [res, setRes] = useState<out | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [facts, setFacts] = useState(false);
  const [mat, setMat] = useState<{ key: string; m: number[][] } | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const req = useRef(0);

  const src = useMemo(() => sents(x), [x]);
  const key = `${method}|${len}|${x}`;
  const cur = res && res.key === key ? res : null;
  const outSents = useMemo(() => (cur?.done ? sents(cur.text) : []), [cur]);
  const words = nw(x);

  async function run() {
    const id = ++req.current;
    const k = key;
    setErr("");
    setSel(null);
    setMat(null);
    if (!x.trim()) return;
    if (method === "quick") {
      const t0 = performance.now();
      const ss = sents(x);
      const n = lens.find((l) => l.k === len)!.n;
      const t = textrank(ss, n).map((i) => ss[i]).join(" ");
      setRes({ key: k, text: t, done: true, ms: performance.now() - t0, method });
      return;
    }
    setBusy(true);
    setRes({ key: k, text: "", done: false, method });
    try {
      const r = await gen(x, len, (s) => id === req.current && setRes({ key: k, text: s, done: false, method }));
      if (id === req.current) setRes({ key: k, text: r.text, done: true, ms: r.ms, ntok: r.ntok, method });
    } catch (e) {
      if (id === req.current) {
        setErr(String((e as Error).message ?? e));
        setRes(null);
      }
    } finally {
      if (id === req.current) setBusy(false);
    }
  }

  // fact check: entailment of every summary sentence against every source sentence
  useEffect(() => {
    if (!facts || !cur?.done || !outSents.length || mat?.key === key + cur.text) return;
    const k = key + cur.text;
    const ps: [string, string][] = [];
    const ss = src.slice(0, 60);
    for (const h of outSents) for (const s of ss) ps.push([s, h]);
    let dead = false;
    nli(ps)
      .then((z) => {
        if (dead) return;
        const m = outSents.map((_, a) => Array.from(z.slice(a * ss.length, (a + 1) * ss.length)));
        setMat({ key: k, m });
      })
      .catch((e) => !dead && setErr(`the fact checker failed: ${e.message}`));
    return () => {
      dead = true;
    };
  }, [facts, cur, outSents, src, key, mat, nli]);

  const checked = facts && cur?.done && mat?.key === key + cur.text ? mat.m : null;
  const hl = useMemo(() => {
    if (sel === null || !outSents[sel]) return new Set<number>();
    return new Set(origin(outSents[sel], src, checked?.[sel]));
  }, [sel, outSents, src, checked]);
  const firstHl = useRef<HTMLSpanElement | null>(null);
  useEffect(() => {
    if (sel !== null) firstHl.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [sel]);

  const loading = busy && st.phase === "loading";
  const f = st.total ? st.loaded / st.total : 0;
  const offs = useMemo(() => {
    const c = src.map((s) => nw(s));
    return c.map((_, i) => c.slice(0, i).reduce((a, b) => a + b, 0));
  }, [src]);
  const cut = method === "ai" && words > readWords;

  return (
    <section>
      <Step n={1}>pick a text, or paste your own</Step>
      <div className="mb-2 flex flex-wrap gap-2">
        {samples.map((s) => (
          <button
            key={s.id}
            onClick={() => {
              setX(s.text);
              setPick(s.id);
            }}
            className={`rounded-full border px-3 py-1 text-xs hover:text-ink ${
              pick === s.id ? "border-acc text-ink" : "border-line text-mute"
            }`}
          >
            {s.name}
          </button>
        ))}
      </div>
      <textarea
        value={x}
        onChange={(e) => {
          setX(e.target.value);
          setPick(null);
        }}
        rows={7}
        placeholder="paste an article, a report, a long email..."
        className="w-full resize-y rounded-lg border border-line bg-card p-3 text-base outline-none focus:border-mute"
      />
      <p className="mt-1 flex items-center gap-1.5 font-mono text-xs text-mute">
        {words} words · {src.length} sentences
        {pick && " · cnn/dailymail test article"}
        <Tip label="where the examples come from">
          the examples are news articles from the CNN/DailyMail test set, which the model never trained on. they are
          shown as published, so the odd formatting is original.
        </Tip>
      </p>

      <div className="mt-6">
        <Step n={2}>
          pick how to summarize
          <Tip label="what is the difference">
            &ldquo;quick&rdquo; picks existing sentences, which is why it never gets a fact wrong but often reads
            clunky. &ldquo;ai summary&rdquo; writes new sentences, which reads better but can occasionally invent or
            mix up details, so there is a fact check below.
          </Tip>
        </Step>
        <div className="grid grid-cols-2 gap-2">
          {methods.map((m) => (
            <button
              key={m.k}
              onClick={() => setMethod(m.k)}
              aria-pressed={method === m.k}
              className={`relative rounded-lg border-2 p-2.5 text-left transition-colors sm:p-3 ${
                method === m.k ? "border-acc bg-acc/10" : "border-line bg-card hover:border-mute"
              }`}
            >
              <span className="block text-sm font-medium sm:text-base">{m.name}</span>
              <span className="mt-0.5 block text-[11px] leading-snug text-mute sm:text-xs">{m.sub}</span>
              {m.best && (
                <span className="absolute top-2 right-2 hidden rounded-sm bg-acc px-1 font-mono text-[10px] text-bg sm:inline">
                  best
                </span>
              )}
            </button>
          ))}
        </div>
        <p className="mt-2 text-sm text-mute">{about[method]}</p>
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="flex items-center gap-1.5 text-sm">
            length
            <Tip label="how length works">
              {method === "ai"
                ? "the model was trained with the length written in front of the text (summarize short / medium / long), so the choice really changes how much it writes: about 40, 55 and 80 words on average."
                : "the number of sentences copied: 2, 3 or 4."}
            </Tip>
          </span>
          <div className="inline-flex overflow-hidden rounded-md border border-line">
            {lens.map((l) => (
              <button
                key={l.k}
                onClick={() => setLen(l.k)}
                aria-pressed={len === l.k}
                className={`px-3 py-1 text-sm ${len === l.k ? "bg-acc text-bg" : "bg-card text-mute hover:text-ink"}`}
              >
                {l.name}
              </button>
            ))}
          </div>
          <button
            onClick={run}
            disabled={busy || !x.trim()}
            className="ml-auto rounded-md bg-ink px-4 py-1.5 text-sm text-bg disabled:opacity-50"
          >
            {busy ? "writing..." : "summarize"}
          </button>
        </div>
      </div>

      <div className="mt-5 overflow-hidden rounded-lg border border-line bg-card">
        <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2 text-sm">
          <span className="font-serif text-lg">summary</span>
          <span className="font-mono text-xs text-mute">
            {cur?.done && cur.ms !== undefined
              ? `${cur.method === "quick" ? "<1 ms" : `${(cur.ms / 1000).toFixed(1)} s, ${cur.ntok} tokens`} · ${nw(cur.text)} words`
              : busy && st.phase === "ready"
                ? "writing..."
                : ""}
          </span>
          <label className="ml-auto flex cursor-pointer items-center gap-2 text-sm">
            <input type="checkbox" checked={facts} onChange={(e) => setFacts(e.target.checked)} className="accent-[var(--acc)]" />
            check the facts
            <Tip label="what is the fact check">
              a small language-inference model (deberta, 8-bit, about 70 mb) is asked, for every summary sentence,
              whether any sentence of the text implies it. sentences it cannot find support for get a warning mark. it
              catches wrong numbers and invented claims fairly often, but it is not perfect.
            </Tip>
          </label>
        </div>
        <div className="relative min-h-40 p-4">
          {loading ? (
            <div className="flex min-h-32 flex-col items-center justify-center gap-3 text-sm text-mute">
              <div className="h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-line">
                <div className="h-full bg-acc transition-[width]" style={{ width: `${Math.max(3, f * 100)}%` }} />
              </div>
              <p className="text-center">
                {st.total
                  ? `downloading the model ${Math.round(f * 100)}% (${(st.total / 2 ** 20).toFixed(0)} mb, only the first time)`
                  : "starting the model (only slow the first time)"}
              </p>
            </div>
          ) : err ? (
            <p className="text-sm text-acc">something went wrong: {err}</p>
          ) : !cur ? (
            <p className="text-sm text-mute">
              {res ? "the text or settings changed. press summarize to update." : "press summarize."}
            </p>
          ) : (
            <>
              <p className="text-base leading-relaxed">
                {cur.done ? (
                  outSents.map((s, i) => {
                    const sc = checked ? Math.max(...checked[i], 0) : null;
                    const bad = sc !== null && sc < maybe;
                    const mid = sc !== null && sc >= maybe && sc < ok;
                    return (
                      <span key={i}>
                        <button
                          onClick={() => setSel(sel === i ? null : i)}
                          className={`rounded-sm text-left transition-colors hover:bg-acc/15 ${sel === i ? "bg-acc/25" : ""} ${
                            bad ? "underline decoration-red-600 decoration-wavy underline-offset-4 dark:decoration-red-400" : ""
                          } ${mid ? "underline decoration-amber-600 decoration-dotted underline-offset-4 dark:decoration-amber-400" : ""}`}
                        >
                          {s}
                        </button>
                        {sc !== null && (
                          <span
                            title={`support ${sc.toFixed(2)}`}
                            className={`mx-1 align-super font-mono text-[10px] ${
                              bad ? "text-red-600 dark:text-red-400" : mid ? "text-amber-600 dark:text-amber-400" : "text-emerald-700 dark:text-emerald-400"
                            }`}
                          >
                            {bad ? "not found in text" : mid ? "weak" : "ok"}
                          </span>
                        )}{" "}
                      </span>
                    );
                  })
                ) : (
                  <>
                    {cur.text}
                    <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-acc align-middle" />
                  </>
                )}
              </p>
              {facts && cur.done && !checked && (
                <p className="mt-3 font-mono text-xs text-mute">
                  {nst.phase === "error"
                    ? "the fact checker could not be loaded."
                    : nst.total && nst.loaded < nst.total
                      ? `downloading the fact checker ${Math.round((nst.loaded / nst.total) * 100)}% (${(nst.total / 2 ** 20).toFixed(0)} mb, only the first time)`
                      : "checking..."}
                </p>
              )}
              {cur.done && (
                <p className="mt-3 text-xs text-mute">
                  {checked
                    ? "tap a sentence to see where in the text it came from. a warning means no sentence of the text clearly backs it up."
                    : "tap a sentence to see where in the text it probably came from."}
                </p>
              )}
            </>
          )}
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-line bg-card p-4">
        <h3 className="mb-2 flex items-center gap-2 font-serif text-xl">
          the text
          <Tip label="what is highlighted">
            the sentences the picked summary sentence most likely came from. without the fact check this is a simple
            word-overlap match, with it the match comes from the language-inference model.
            {cut && " the grey part is beyond what the ai summary reads."}
          </Tip>
        </h3>
        <p className="max-h-72 overflow-y-auto text-sm leading-relaxed">
          {src.map((s, i) => {
            const out = cut && offs[i] >= readWords;
            const on = hl.has(i);
            const first = on && Math.min(...hl) === i;
            return (
              <span
                key={i}
                ref={first ? firstHl : undefined}
                className={`${on ? "rounded-sm bg-acc/25" : ""} ${out ? "text-mute/60" : ""}`}
              >
                {s}{" "}
              </span>
            );
          })}
        </p>
      </div>
    </section>
  );
}
