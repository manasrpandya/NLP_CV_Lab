"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { bgl } from "@/lib/bilateral-denoising/gl";
import { autoParams, chan, estSigma, psnr } from "@/lib/bilateral-denoising/est";
import { Tip } from "./tip";

const samples = ["houses", "lighthouse", "parrots", "monarch"];
const maxSide = 768;

type mode = "manual" | "auto" | "neural";
type heat = "none" | "sr" | "ss" | "guide";
// channel order of the onnx "maps" output: sig_s1, sig_s2, sig_r1, sig_r2, guide
const heatIdx: Record<Exclude<heat, "none">, number> = { ss: 0, sr: 2, guide: 4 };

const modes: { k: mode; name: string; sub: string; best?: boolean }[] = [
  { k: "manual", name: "hand-tuned", sub: "classic filter, you set it" },
  { k: "auto", name: "auto", sub: "classic filter, sets itself" },
  { k: "neural", name: "neural", sub: "best quality, takes a few seconds", best: true },
];

const about: Record<mode, string> = {
  manual: "the classic bilateral filter: it blurs noise but stops at edges. move the sliders and watch it change.",
  auto: "same classic filter, but it first measures how noisy the photo is and picks the settings for you.",
  neural:
    "a tiny neural network picks the filter settings separately for every pixel, so flat sky and sharp edges get different treatment.",
};

const heatNote: Record<Exclude<heat, "none">, string> = {
  sr: "bright = smooths across bigger brightness jumps (flat areas), dark = protects edges and texture.",
  ss: "bright = averages over a wider area, dark = stays local.",
  guide: "the image the network uses to decide what counts as an edge. it is a cleaned-up, contrast-boosted copy.",
};

const heatName: Record<Exclude<heat, "none">, string> = { sr: "edge threshold", ss: "blur distance", guide: "guide" };

type nnRes = { key: string; rgba: Uint8Array; maps: Float32Array; ms: number };

function Slider(p: {
  label: string;
  v: number;
  min: number;
  max: number;
  step: number;
  on: (v: number) => void;
  off?: boolean;
  fmt?: (v: number) => string;
  tip?: React.ReactNode;
}) {
  return (
    <div className={p.off ? "opacity-50" : ""}>
      <span className="flex items-center justify-between gap-2 text-xs">
        <span className="flex items-center gap-1.5 text-mute">
          {p.label}
          {p.tip && <Tip label={`about ${p.label}`}>{p.tip}</Tip>}
        </span>
        <span className="font-mono">{p.fmt ? p.fmt(p.v) : p.v}</span>
      </span>
      <input
        type="range"
        aria-label={p.label}
        className="mt-1 w-full accent-[var(--acc)]"
        min={p.min}
        max={p.max}
        step={p.step}
        value={p.v}
        disabled={p.off}
        onChange={(e) => p.on(Number(e.target.value))}
      />
    </div>
  );
}

function Seg<T extends string>(p: { opts: [T, string][]; v: T; on: (v: T) => void }) {
  return (
    <div className="inline-flex flex-wrap overflow-hidden rounded-md border border-line text-xs">
      {p.opts.map(([k, l]) => (
        <button
          key={k}
          onClick={() => p.on(k)}
          className={`px-2.5 py-1.5 ${p.v === k ? "bg-ink text-bg" : "bg-card hover:bg-line/50"}`}
        >
          {l}
        </button>
      ))}
    </div>
  );
}

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

export function Demo() {
  const cv = useRef<HTMLCanvasElement>(null);
  const work = useRef<HTMLCanvasElement | null>(null);
  const g = useRef<bgl | null>(null);
  const wk = useRef<Worker | null>(null);

  const [src, setSrc] = useState<string>("houses");
  const [mode, setMode] = useState<mode>("auto");
  const [sig, setSig] = useState(60);
  const [r, setR] = useState(3);
  const [ss, setSs] = useState(1.6);
  const [sr, setSr] = useState(62);
  const [passes, setPasses] = useState(1);
  const [split, setSplit] = useState(0.5);
  const [heat, setHeat] = useState<heat>("none");
  const [left, setLeft] = useState<"noisy" | "clean">("noisy");
  const [size, setSize] = useState<[number, number]>([768, 512]);
  const [err, setErr] = useState("");
  const [stats, setStats] = useState({ pn: NaN, po: NaN, est: NaN, ms: NaN });
  const [nn, setNN] = useState<{ st: "idle" | "loading" | "ready" | "error"; p: number; ep: string }>({
    st: "idle",
    p: 0,
    ep: "",
  });
  const [busy, setBusy] = useState(-1);
  const [heatRange, setHeatRange] = useState<[number, number]>([0, 0]);
  const [hasNN, setHasNN] = useState(false);

  // mirror of the controls for the draw callback
  const st = useRef({ src, mode, sig, r, ss, sr, passes, split, heat, left, imgId: 0 });
  st.current = { ...st.current, src, mode, sig, r, ss, sr, passes, split, heat, left };
  const last = useRef<nnRes | null>(null);
  const inflight = useRef<string | null>(null);
  const estKey = useRef("");
  const estVal = useRef(NaN);
  const nnReady = useRef(false);

  const key = () => `${st.current.imgId}|${st.current.sig}`;

  const pushHeat = useCallback(() => {
    const G = g.current;
    const res = last.current;
    if (!G || !res || st.current.heat === "none") return;
    const m = res.maps.subarray(heatIdx[st.current.heat] * G.w * G.h, (heatIdx[st.current.heat] + 1) * G.w * G.h);
    // 2-98 percentile stretch so the map is readable
    const s = Float32Array.from(m.filter((_, i) => i % 7 === 0)).sort();
    const lo = s[Math.floor(s.length * 0.02)];
    const hi = s[Math.floor(s.length * 0.98)] + 1e-6;
    const u8 = new Uint8Array(m.length);
    for (let i = 0; i < m.length; i++) u8[i] = Math.max(0, Math.min(255, ((m[i] - lo) / (hi - lo)) * 255));
    G.setNN(res.rgba, u8);
    const sc = st.current.heat === "sr" ? 255 : 1;
    setHeatRange([lo * sc, hi * sc]);
  }, []);

  const draw = useCallback(() => {
    const G = g.current;
    const s = st.current;
    if (!G || !G.w) return;
    G.prep(s.sig / 255, 7);
    const k = key();

    if (s.mode === "auto" && (estKey.current !== k || isNaN(estVal.current))) {
      estKey.current = k;
      const px = G.read("noisy");
      estVal.current = [0, 1, 2].reduce((a, c) => a + estSigma(chan(px, G.w, G.h, c), G.w, G.h), 0) / 3;
    }
    let q = { sig: s.sig, r: s.r, ss: s.ss, sr: s.sr / 255, passes: s.passes };
    if (s.mode === "auto" && !isNaN(estVal.current)) {
      const a = autoParams(estVal.current);
      q = { ...q, r: a.r, ss: a.ss, sr: a.sr / 255, passes: 1 };
    }

    let raw = false;
    if (s.mode === "neural") {
      const fresh = last.current?.key === k;
      if (wk.current && nnReady.current && inflight.current === null && !fresh) {
        const px = G.read("noisy");
        const n = G.w * G.h;
        const x = new Float32Array(3 * n);
        for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) x[c * n + i] = px[4 * i + c] / 255;
        inflight.current = k;
        setBusy(0);
        wk.current.postMessage({ type: "run", id: k, x, w: G.w, h: G.h }, [x.buffer]);
      }
      G.useNN = fresh;
      raw = !fresh;
      setHasNN(fresh);
    } else {
      G.useNN = false;
      G.filter(q);
    }
    G.show(s.split, s.mode === "neural" && s.heat !== "none" && G.useNN ? 0.85 : 0, s.left === "clean", raw);

    const out = G.useNN ? last.current!.rgba : G.read("out");
    const clean = G.read("clean");
    const ok = s.sig > 0;
    setStats((o) => ({
      ...o,
      pn: ok ? psnr(clean, G.read("noisy")) : NaN,
      po: ok && !raw ? psnr(clean, out) : NaN,
      est: s.mode === "auto" ? estVal.current : o.est,
    }));
  }, []);

  useEffect(() => {
    try {
      g.current = new bgl(cv.current!);
    } catch (e) {
      queueMicrotask(() => setErr(String(e)));
    }
    work.current = document.createElement("canvas");
    return () => wk.current?.terminate();
  }, []);

  const loadNN = useCallback(() => {
    if (wk.current) return;
    const w = new Worker(new URL("../../lib/bilateral-denoising/worker.ts", import.meta.url), { type: "module" });
    wk.current = w;
    setNN({ st: "loading", p: 0, ep: "" });
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") setNN((o) => ({ ...o, p: m.p }));
      if (m.type === "tile") setBusy(m.p);
      if (m.type === "ep") setNN((o) => ({ ...o, ep: m.ep }));
      if (m.type === "ready") {
        nnReady.current = true;
        setNN({ st: "ready", p: 1, ep: m.ep });
        requestAnimationFrame(draw);
      }
      if (m.type === "error") {
        inflight.current = null;
        setBusy(-1);
        setNN((o) => ({ ...o, st: "error" }));
        setErr(m.msg);
      }
      if (m.type === "done") {
        const G = g.current!;
        const n = G.w * G.h;
        inflight.current = null;
        setBusy(-1);
        const y = m.y as Float32Array;
        // the image may have changed while the model ran
        if (y.length !== 3 * n) return requestAnimationFrame(draw);
        const rgba = new Uint8Array(4 * n);
        for (let i = 0; i < n; i++) {
          for (let c = 0; c < 3; c++) rgba[4 * i + c] = Math.max(0, Math.min(255, Math.round(y[c * n + i] * 255)));
          rgba[4 * i + 3] = 255;
        }
        last.current = { key: m.id, rgba, maps: m.maps, ms: m.ms };
        G.setNN(rgba, null);
        pushHeat();
        setStats((o) => ({ ...o, ms: m.ms }));
        requestAnimationFrame(draw);
      }
    };
    // ?ep=wasm forces the cpu backend, handy for debugging
    w.postMessage({ type: "load", ep: new URLSearchParams(location.search).get("ep") ?? undefined });
  }, [draw, pushHeat]);

  useEffect(() => {
    if (src === "upload") return;
    let dead = false;
    const im = new Image();
    im.src = `/bilateral-denoising/${src}.webp`;
    im.decode().then(() => {
      if (!dead) setImage(im, im.naturalWidth, im.naturalHeight);
    });
    return () => {
      dead = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  function setImage(el: CanvasImageSource, w0: number, h0: number) {
    const G = g.current;
    if (!G || !work.current) return;
    const k = Math.min(1, maxSide / Math.max(w0, h0));
    const w = Math.round(w0 * k);
    const h = Math.round(h0 * k);
    work.current.width = w;
    work.current.height = h;
    work.current.getContext("2d")!.drawImage(el, 0, 0, w, h);
    cv.current!.width = w;
    cv.current!.height = h;
    G.resize(w, h);
    G.upload(work.current);
    st.current.imgId++;
    last.current = null;
    estVal.current = NaN;
    setSize([w, h]);
    requestAnimationFrame(draw);
  }

  useEffect(() => {
    requestAnimationFrame(draw);
  }, [draw, src, mode, sig, r, ss, sr, passes, split, heat, left]);

  useEffect(() => {
    if (mode === "neural") loadNN();
  }, [mode, loadNN]);

  useEffect(() => {
    pushHeat();
  }, [heat, pushHeat]);

  async function onFile(f: File | undefined) {
    if (!f) return;
    const bm = await createImageBitmap(f);
    setSrc("upload");
    setImage(bm, bm.width, bm.height);
  }

  function download() {
    const G = g.current;
    if (!G) return;
    const px = G.useNN && last.current ? last.current.rgba : G.read("out");
    const c = document.createElement("canvas");
    c.width = G.w;
    c.height = G.h;
    c.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(px), G.w, G.h), 0, 0);
    c.toBlob((b) => {
      if (!b) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(b);
      a.download = `denoised-${mode}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
  }

  const drag = (e: React.PointerEvent) => {
    if (e.type === "pointermove" && e.buttons === 0 && e.pointerType === "mouse") return;
    const b = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setSplit(Math.max(0, Math.min(1, (e.clientX - b.left) / b.width)));
  };

  const auto = mode === "auto" && !isNaN(stats.est) ? autoParams(stats.est) : null;
  const f1 = (v: number) => v.toFixed(1);
  const working =
    mode !== "neural"
      ? ""
      : nn.st === "loading"
        ? `downloading the model ${Math.round(nn.p * 100)}%`
        : busy >= 0
          ? `cleaning ${Math.round(busy * 100)}%`
          : nn.st === "ready" && !hasNN
            ? "cleaning"
            : "";

  return (
    <section>
      <Step n={1}>pick a photo, or use your own</Step>
      <div className="flex flex-wrap items-center gap-2">
        {samples.map((s) => (
          <button
            key={s}
            onClick={() => setSrc(s)}
            title={s}
            className={`h-12 w-16 overflow-hidden rounded border-2 ${src === s ? "border-acc" : "border-transparent"}`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`/bilateral-denoising/${s}-t.jpg`} alt={s} className="h-full w-full object-cover" />
          </button>
        ))}
        <label
          className={`flex h-12 cursor-pointer items-center rounded-md border-2 border-dashed px-3 text-sm ${src === "upload" ? "border-acc" : "border-line hover:border-mute"}`}
        >
          upload
          <input type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
      </div>

      <div className="mt-6">
        <Step n={2}>
          pick how to clean it
          <Tip label="what is the difference">
            all three use the same idea: replace each pixel by an average of similar nearby pixels. they differ in who
            picks the settings. hand-tuned: you. auto: a noise estimate plus a lookup table. neural: a small network,
            separately for every pixel. neural is the most accurate, the other two are instant.
          </Tip>
        </Step>
        <div className="grid grid-cols-3 gap-2">
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
                  best
                </span>
              )}
            </button>
          ))}
        </div>
        <p className="mt-2 text-sm text-mute">{about[mode]}</p>
      </div>

      <div className="mt-5 grid gap-6 lg:grid-cols-[1fr_280px]">
        <div>
          <div
            className="relative w-full touch-none select-none overflow-hidden rounded-lg border border-line bg-card"
            style={{ aspectRatio: `${size[0]} / ${size[1]}` }}
            onPointerDown={drag}
            onPointerMove={drag}
          >
            <canvas ref={cv} className="h-full w-full cursor-ew-resize" />
            <span className="pointer-events-none absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white">
              {left === "noisy" ? "noisy" : "original"}
            </span>
            <span className="pointer-events-none absolute top-2 right-2 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white">
              {mode === "neural" && !hasNN ? "noisy" : mode === "neural" && heat !== "none" ? `${heatName[heat]} map` : "cleaned"}
            </span>
            <span
              className="pointer-events-none absolute top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white/90 font-mono text-xs text-black shadow"
              style={{ left: `${split * 100}%` }}
            >
              &lsaquo;&rsaquo;
            </span>
            {working && (
              <span className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center gap-2 bg-black/65 px-3 py-2 text-xs text-white">
                <span className="h-3 w-3 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                {working}
                {nn.st === "loading" && <span className="text-white/70">(460 kB, only the first time)</span>}
              </span>
            )}
          </div>
          <p className="mt-1.5 text-xs text-mute">drag across the photo to compare before and after.</p>

          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
            <span className="flex items-center gap-1.5">
              <span className="text-mute">quality</span>
              <span className="font-mono">{isNaN(stats.pn) ? "n/a" : f1(stats.pn)}</span>
              <span className="text-mute">&rarr;</span>
              <span className="font-mono text-acc">{isNaN(stats.po) ? "..." : f1(stats.po)}</span>
              <span className="text-mute">dB</span>
              <Tip label="about the quality score">
                PSNR: how close the picture is to the original, noise-free photo. higher is better, and every +3 dB
                halves the leftover error. it needs the original, so it is only shown when noise was added here.
              </Tip>
            </span>
            {mode === "neural" && nn.st === "ready" && !isNaN(stats.ms) && (
              <span className="text-xs text-mute">
                took {(stats.ms / 1000).toFixed(1)} s on {nn.ep === "webgpu" ? "your gpu" : "your cpu"}
              </span>
            )}
          </div>
          {sig === 0 && (
            <p className="mt-1 text-xs text-mute">no noise added, so there is no original to score against.</p>
          )}
          {err && <p className="mt-2 text-sm text-acc">{err}</p>}
        </div>

        <div className="space-y-4">
          <Slider
            label="noise to add"
            v={sig}
            min={0}
            max={60}
            step={1}
            on={setSig}
            tip="adds random grain to the photo, like a camera in low light. σ is its strength on the 0-255 scale. set it to 0 for your own already-noisy photo."
          />

          {mode === "manual" && (
            <div className="space-y-4 rounded-lg border border-line p-3">
              <p className="text-xs text-mute">filter settings</p>
              <Slider
                label="window radius"
                v={r}
                min={1}
                max={8}
                step={1}
                on={setR}
                tip="how many pixels around each pixel are looked at."
              />
              <Slider
                label="blur distance σ_s"
                v={ss}
                min={0.5}
                max={6}
                step={0.1}
                on={setSs}
                tip="how fast the weight of a neighbour drops with distance, in pixels. bigger = blurrier."
              />
              <Slider
                label="edge threshold σ_r"
                v={sr}
                min={2}
                max={160}
                step={1}
                on={setSr}
                tip="how different two pixels can be and still get averaged. too low leaves noise, too high blurs edges away. a good start is about 2.5 times the noise level."
              />
              <Slider
                label="passes"
                v={passes}
                min={1}
                max={3}
                step={1}
                on={setPasses}
                tip="run the filter again on its own output. two passes with a lower edge threshold often beat one."
              />
            </div>
          )}

          {mode === "auto" && (
            <div className="rounded-lg border border-line p-3 text-sm">
              <p className="flex items-center gap-1.5">
                <span className="text-mute">measured noise</span>
                <span className="font-mono">{isNaN(stats.est) ? "..." : f1(stats.est)}</span>
                {sig > 0 && <span className="text-xs text-mute">(actual {sig})</span>}
                <Tip label="how the noise is measured">
                  a small edge-cancelling filter (immerkaer&apos;s method) responds mostly to noise, and its average
                  response gives the noise level. the settings then come from a table found by testing every
                  combination on 32 training photos.
                </Tip>
              </p>
              {auto && (
                <p className="mt-2 font-mono text-xs text-mute">
                  chose radius {auto.r}, σ_s {auto.ss.toFixed(1)}, σ_r {Math.round(auto.sr)}
                </p>
              )}
            </div>
          )}

          {mode === "neural" && (
            <div className="space-y-2 rounded-lg border border-line p-3">
              <p className="flex items-center gap-1.5 text-xs text-mute">
                see what the network chose
                <Tip label="about the overlay">
                  the network outputs the filter settings for every pixel. colour them over the result to see where it
                  smooths hard and where it protects detail.
                </Tip>
              </p>
              <Seg
                opts={[
                  ["none", "off"],
                  ["sr", "edge threshold"],
                  ["ss", "blur distance"],
                  ["guide", "guide"],
                ]}
                v={heat}
                on={setHeat}
              />
              {heat !== "none" && (
                <>
                  <div className="h-2 rounded bg-[linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)]" />
                  <p className="flex justify-between font-mono text-[11px] text-mute">
                    <span>{heatRange[0].toFixed(heat === "sr" ? 1 : 2)}</span>
                    <span>{heatRange[1].toFixed(heat === "sr" ? 1 : 2)}</span>
                  </p>
                  <p className="text-xs text-mute">{heatNote[heat]}</p>
                </>
              )}
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setLeft(left === "noisy" ? "clean" : "noisy")}
              className="rounded-md border border-line px-3 py-1.5 text-sm hover:bg-line/50"
            >
              compare with {left === "noisy" ? "original" : "noisy"}
            </button>
            <button onClick={download} className="rounded-md bg-ink px-3 py-1.5 text-sm text-bg hover:opacity-85">
              download
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
