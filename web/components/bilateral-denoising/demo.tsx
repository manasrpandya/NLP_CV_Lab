"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { bgl } from "@/lib/bilateral-denoising/gl";
import { autoParams, estSigma, luma, psnr } from "@/lib/bilateral-denoising/est";

const samples = ["parrots", "lighthouse", "houses", "monarch"];
const maxSide = 768;
const camSide = 640;

type mode = "manual" | "auto" | "neural";
type heat = "none" | "sr" | "ss" | "guide";
// channel order of the onnx "maps" output: sig_s1, sig_s2, sig_r1, sig_r2, guide
const heatIdx: Record<Exclude<heat, "none">, number> = { ss: 0, sr: 2, guide: 4 };

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
}) {
  return (
    <label className={`block ${p.off ? "opacity-50" : ""}`}>
      <span className="flex justify-between text-xs">
        <span className="text-mute">{p.label}</span>
        <span className="font-mono">{p.fmt ? p.fmt(p.v) : p.v}</span>
      </span>
      <input
        type="range"
        className="mt-1 w-full accent-[var(--acc)]"
        min={p.min}
        max={p.max}
        step={p.step}
        value={p.v}
        disabled={p.off}
        onChange={(e) => p.on(Number(e.target.value))}
      />
    </label>
  );
}

function Seg<T extends string>(p: { opts: [T, string][]; v: T; on: (v: T) => void }) {
  return (
    <div className="inline-flex overflow-hidden rounded-md border border-line text-sm">
      {p.opts.map(([k, l]) => (
        <button
          key={k}
          onClick={() => p.on(k)}
          className={`px-3 py-1.5 ${p.v === k ? "bg-ink text-bg" : "bg-card hover:bg-line/50"}`}
        >
          {l}
        </button>
      ))}
    </div>
  );
}

export function Demo() {
  const cv = useRef<HTMLCanvasElement>(null);
  const work = useRef<HTMLCanvasElement | null>(null);
  const vid = useRef<HTMLVideoElement | null>(null);
  const g = useRef<bgl | null>(null);
  const wk = useRef<Worker | null>(null);

  const [src, setSrc] = useState<string>("parrots");
  const [mode, setMode] = useState<mode>("manual");
  const [sig, setSig] = useState(25);
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
  const [heatRange, setHeatRange] = useState<[number, number]>([0, 0]);

  // mirror of everything the render loop needs, so the camera loop never sees stale state
  const st = useRef({ src, mode, sig, r, ss, sr, passes, split, heat, left, imgId: 0, frame: 0 });
  st.current = { ...st.current, src, mode, sig, r, ss, sr, passes, split, heat, left };
  const last = useRef<nnRes | null>(null);
  const inflight = useRef<string | null>(null);
  const estKey = useRef("");
  const estVal = useRef(NaN);

  const key = () => `${st.current.imgId}|${st.current.sig}|${st.current.src === "cam" ? st.current.frame : 0}`;

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
    const cam = s.src === "cam";
    if (cam && vid.current && work.current && vid.current.readyState >= 2) {
      work.current.getContext("2d")!.drawImage(vid.current, 0, 0, G.w, G.h);
      G.upload(work.current);
      s.frame++;
    }
    G.prep(s.sig / 255, cam ? s.frame : 7);
    const k = key();

    // blind noise estimate for auto mode (every frame would be wasteful on video)
    if (s.mode === "auto" && (estKey.current !== k || isNaN(estVal.current)) && (!cam || s.frame % 20 === 1)) {
      estKey.current = k;
      const y = luma(G.read("noisy"), G.w, G.h);
      estVal.current = estSigma(y, G.w, G.h);
    }
    let q = { sig: s.sig, r: s.r, ss: s.ss, sr: s.sr / 255, passes: s.passes };
    if (s.mode === "auto" && !isNaN(estVal.current)) {
      const a = autoParams(estVal.current);
      q = { ...q, r: a.r, ss: a.ss, sr: a.sr / 255, passes: 1 };
    }

    if (s.mode === "neural") {
      if (wk.current && nnReady.current && inflight.current === null && last.current?.key !== k) {
        const px = G.read("noisy");
        const n = G.w * G.h;
        const x = new Float32Array(3 * n);
        for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) x[c * n + i] = px[4 * i + c] / 255;
        inflight.current = k;
        wk.current.postMessage({ type: "run", id: k, x, w: G.w, h: G.h }, [x.buffer]);
      }
      if (last.current && last.current.rgba.length === G.w * G.h * 4) G.useNN = true;
    } else {
      G.useNN = false;
      G.filter(q);
    }
    G.show(s.split, s.mode === "neural" && s.heat !== "none" && G.useNN ? 0.65 : 0, s.left === "clean");

    if (!cam || s.frame % 15 === 1) {
      const out = G.useNN ? last.current!.rgba : G.read("out");
      const clean = G.read("clean");
      const ok = s.sig > 0;
      setStats((o) => ({
        ...o,
        pn: ok ? psnr(clean, G.read("noisy")) : NaN,
        po: ok ? psnr(clean, out) : NaN,
        est: s.mode === "auto" ? estVal.current : o.est,
      }));
    }
  }, []);

  const nnReady = useRef(false);

  // gl + worker setup
  useEffect(() => {
    try {
      g.current = new bgl(cv.current!);
    } catch (e) {
      queueMicrotask(() => setErr(String(e)));
    }
    work.current = document.createElement("canvas");
    return () => {
      wk.current?.terminate();
      const v = vid.current;
      (v?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const loadNN = useCallback(() => {
    if (wk.current) return;
    const w = new Worker(new URL("../../lib/bilateral-denoising/worker.ts", import.meta.url), { type: "module" });
    wk.current = w;
    setNN({ st: "loading", p: 0, ep: "" });
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") setNN((o) => ({ ...o, p: m.p }));
      if (m.type === "ready") {
        nnReady.current = true;
        setNN({ st: "ready", p: 1, ep: m.ep });
        requestAnimationFrame(draw);
      }
      if (m.type === "error") {
        inflight.current = null;
        setNN((o) => ({ ...o, st: "error" }));
        setErr(m.msg);
      }
      if (m.type === "done") {
        const G = g.current!;
        const n = G.w * G.h;
        const rgba = new Uint8Array(4 * n);
        const y = m.y as Float32Array;
        for (let i = 0; i < n; i++) {
          for (let c = 0; c < 3; c++) rgba[4 * i + c] = Math.max(0, Math.min(255, Math.round(y[c * n + i] * 255)));
          rgba[4 * i + 3] = 255;
        }
        inflight.current = null;
        if (rgba.length !== n * 4) return;
        last.current = { key: m.id, rgba, maps: m.maps, ms: m.ms };
        G.setNN(rgba, null);
        pushHeat();
        setStats((o) => ({ ...o, ms: m.ms }));
        if (st.current.src !== "cam") requestAnimationFrame(draw);
      }
    };
    w.postMessage({ type: "load" });
  }, [draw, pushHeat]);

  // load still images
  useEffect(() => {
    if (src === "cam" || src === "upload") return;
    let dead = false;
    const im = new Image();
    im.src = `/bilateral-denoising/${src}.webp`;
    im.decode().then(() => {
      if (dead) return;
      setImage(im, im.naturalWidth, im.naturalHeight);
    });
    return () => {
      dead = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  function setImage(el: CanvasImageSource, w0: number, h0: number, cap = maxSide) {
    const G = g.current;
    if (!G || !work.current) return;
    const k = Math.min(1, cap / Math.max(w0, h0));
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

  // redraw stills whenever a control changes
  useEffect(() => {
    if (src !== "cam") requestAnimationFrame(draw);
  }, [draw, src, mode, sig, r, ss, sr, passes, split, heat, left]);

  useEffect(() => {
    if (mode === "neural") loadNN();
  }, [mode, loadNN]);

  useEffect(() => {
    pushHeat();
  }, [heat, pushHeat]);

  // camera loop
  useEffect(() => {
    if (src !== "cam") return;
    let raf = 0;
    let stream: MediaStream | null = null;
    const v = document.createElement("video");
    v.playsInline = true;
    v.muted = true;
    vid.current = v;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment", width: { ideal: 640 } } })
      .then(async (s) => {
        stream = s;
        v.srcObject = s;
        await v.play();
        setImage(v, v.videoWidth, v.videoHeight, camSide);
        const loop = () => {
          draw();
          raf = requestAnimationFrame(loop);
        };
        loop();
      })
      .catch((e) => setErr(`camera: ${e.message ?? e}`));
    return () => {
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, draw]);

  async function onFile(f: File | undefined) {
    if (!f) return;
    const bm = await createImageBitmap(f);
    setSrc("upload");
    setSig(0);
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
  const off = mode !== "manual";
  const f1 = (v: number) => v.toFixed(1);

  return (
    <section>
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
          className={`cursor-pointer rounded-md border px-3 py-1.5 text-sm ${src === "upload" ? "border-acc" : "border-line"}`}
        >
          upload
          <input type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
        <button
          onClick={() => setSrc("cam")}
          className={`rounded-md border px-3 py-1.5 text-sm ${src === "cam" ? "border-acc" : "border-line"}`}
        >
          webcam
        </button>
      </div>

      <div className="mt-4 grid gap-6 lg:grid-cols-[1fr_280px]">
        <div>
          <div
            className="relative w-full touch-none select-none overflow-hidden rounded-lg border border-line bg-card"
            style={{ aspectRatio: `${size[0]} / ${size[1]}` }}
            onPointerDown={drag}
            onPointerMove={drag}
          >
            <canvas ref={cv} className="h-full w-full cursor-ew-resize" />
            <span className="pointer-events-none absolute top-2 left-2 rounded bg-black/55 px-1.5 py-0.5 font-mono text-[11px] text-white">
              {left}
            </span>
            <span className="pointer-events-none absolute top-2 right-2 rounded bg-black/55 px-1.5 py-0.5 font-mono text-[11px] text-white">
              {mode === "neural" ? (nn.st === "ready" ? "neural" : "loading model") : mode}
            </span>
          </div>
          <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1 font-mono text-sm">
            <span>
              <span className="text-mute">psnr in </span>
              {isNaN(stats.pn) ? "n/a" : f1(stats.pn)}
            </span>
            <span>
              <span className="text-mute">psnr out </span>
              <span className="text-acc">{isNaN(stats.po) ? "n/a" : f1(stats.po)}</span>
              {!isNaN(stats.po) && <span className="text-mute"> dB</span>}
            </span>
            {mode === "auto" && (
              <span>
                <span className="text-mute">est. σ </span>
                {isNaN(stats.est) ? "..." : f1(stats.est)}
              </span>
            )}
            {mode === "neural" && nn.st === "ready" && (
              <span className="text-mute">
                {nn.ep} {isNaN(stats.ms) ? "" : `${Math.round(stats.ms)} ms`}
              </span>
            )}
          </div>
          {sig === 0 && <p className="mt-1 text-xs text-mute">no noise added, so there is no ground truth for psnr.</p>}
          {err && <p className="mt-2 text-sm text-acc">{err}</p>}
        </div>

        <div className="space-y-4">
          <Seg
            opts={[
              ["manual", "manual"],
              ["auto", "auto"],
              ["neural", "neural"],
            ]}
            v={mode}
            on={setMode}
          />
          <p className="text-xs text-mute">
            {mode === "manual" && "classic bilateral filter in a webgl shader, set the params yourself."}
            {mode === "auto" &&
              "noise level is estimated blind (immerkaer), then mapped to filter params with the fit from the grid search."}
            {mode === "neural" &&
              "a 93k param unet predicts sig_s, sig_r and a guide per pixel for two bilateral passes. runs in a worker with onnx runtime."}
          </p>
          {mode === "neural" && nn.st === "loading" && (
            <div className="h-1.5 overflow-hidden rounded bg-line">
              <div className="h-full bg-acc transition-all" style={{ width: `${Math.round(nn.p * 100)}%` }} />
            </div>
          )}
          <Slider label="added noise σ" v={sig} min={0} max={60} step={1} on={setSig} />
          <Slider label="radius" v={auto ? auto.r : r} min={1} max={8} step={1} on={setR} off={off} />
          <Slider label="σ_s (px)" v={auto ? +auto.ss.toFixed(2) : ss} min={0.5} max={6} step={0.1} on={setSs} off={off} />
          <Slider label="σ_r (0-255)" v={auto ? Math.round(auto.sr) : sr} min={2} max={160} step={1} on={setSr} off={off} />
          <Slider label="passes" v={mode === "manual" ? passes : 1} min={1} max={3} step={1} on={setPasses} off={off} />
          {mode === "neural" && (
            <div>
              <p className="mb-1 text-xs text-mute">overlay predicted map</p>
              <Seg
                opts={[
                  ["none", "none"],
                  ["sr", "σ_r"],
                  ["ss", "σ_s"],
                  ["guide", "guide"],
                ]}
                v={heat}
                on={setHeat}
              />
              {heat !== "none" && (
                <p className="mt-1 font-mono text-[11px] text-mute">
                  dark {heatRange[0].toFixed(heat === "sr" ? 1 : 2)} &rarr; bright {heatRange[1].toFixed(heat === "sr" ? 1 : 2)}
                </p>
              )}
            </div>
          )}
          <div className="flex flex-wrap gap-2 pt-1">
            <button
              onClick={() => setLeft(left === "noisy" ? "clean" : "noisy")}
              className="rounded-md border border-line px-3 py-1.5 text-sm hover:bg-line/50"
            >
              left side: {left}
            </button>
            <button onClick={download} className="rounded-md bg-ink px-3 py-1.5 text-sm text-bg hover:opacity-85">
              download result
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
