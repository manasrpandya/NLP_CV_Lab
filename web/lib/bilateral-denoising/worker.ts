import * as ort from "onnxruntime-web/webgpu";

// runs the neural-guided bilateral onnx model on rgb planes, tiled so memory stays small on phones

const url = "https://huggingface.co/manasrpandya/bilateral-denoise-nbf/resolve/main/nbf.onnx";
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

const tile = 256;
const pad = 24;
let sess: ort.InferenceSession | null = null;
let ep = "";
let buf: Uint8Array;

// some webgpu stacks (e.g. software adapters) never return from run(), so every gpu call gets a deadline
const timed = <T,>(p: Promise<T>, ms: number) =>
  Promise.race([p, new Promise<never>((_, no) => setTimeout(() => no(new Error("timeout")), ms))]);

async function toWasm() {
  sess = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  ep = "wasm";
  postMessage({ type: "ep", ep });
}

async function load(force?: string) {
  const r = await fetch(url);
  const n = Number(r.headers.get("content-length")) || 0;
  const rd = r.body!.getReader();
  const parts: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    if (n) postMessage({ type: "progress", p: got / n });
  }
  buf = new Uint8Array(got);
  let o = 0;
  for (const p of parts) {
    buf.set(p, o);
    o += p.length;
  }
  const gpu = "gpu" in navigator && (await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter());
  for (const e of force ? [force] : gpu ? ["webgpu", "wasm"] : ["wasm"]) {
    try {
      const sx = await ort.InferenceSession.create(buf, { executionProviders: [e] });
      // warm up on a full tile, this also compiles the gpu shaders
      const k = tile + 2 * pad;
      await timed(sx.run({ x: new ort.Tensor("float32", new Float32Array(3 * k * k), [3, 1, k, k]) }), 20000);
      sess = sx;
      ep = e;
      break;
    } catch (err) {
      console.warn("ep failed", e, err);
    }
  }
  if (!sess) throw new Error("could not create an onnx session");
  postMessage({ type: "ready", ep });
}

// x: (3, h, w) planes in [0, 1]. returns denoised planes and the 5 parameter maps averaged over channels
async function run(x: Float32Array, w: number, h: number) {
  const y = new Float32Array(3 * w * h);
  const maps = new Float32Array(5 * w * h);
  const tw = Math.min(w, tile + 2 * pad);
  const th = Math.min(h, tile + 2 * pad);
  const inp = new Float32Array(3 * tw * th);
  const nt = Math.ceil(w / tile) * Math.ceil(h / tile);
  let done = 0;
  for (let cy = 0; cy < h; cy += tile) {
    for (let cx = 0; cx < w; cx += tile) {
      const x0 = Math.max(0, Math.min(cx - pad, w - tw));
      const y0 = Math.max(0, Math.min(cy - pad, h - th));
      for (let c = 0; c < 3; c++)
        for (let r = 0; r < th; r++) inp.set(x.subarray(c * w * h + (y0 + r) * w + x0, c * w * h + (y0 + r) * w + x0 + tw), (c * th + r) * tw);
      const feed = { x: new ort.Tensor("float32", inp, [3, 1, th, tw]) };
      let out: ort.InferenceSession.OnnxValueMapType;
      try {
        out = ep === "webgpu" ? await timed(sess!.run(feed), 20000) : await sess!.run(feed);
      } catch (err) {
        if (ep !== "webgpu") throw err;
        console.warn("webgpu run failed, falling back to wasm", err);
        await toWasm();
        out = await sess!.run(feed);
      }
      postMessage({ type: "tile", p: ++done / nt });
      const oy = out.y.data as Float32Array;
      const om = out.maps.data as Float32Array;
      const ex = Math.min(cx + tile, w);
      const ey = Math.min(cy + tile, h);
      for (let r = cy; r < ey; r++) {
        for (let q = cx; q < ex; q++) {
          const ti = (r - y0) * tw + (q - x0);
          for (let c = 0; c < 3; c++) y[c * w * h + r * w + q] = oy[c * tw * th + ti];
          for (let m = 0; m < 5; m++) {
            let s = 0;
            for (let c = 0; c < 3; c++) s += om[(c * 5 + m) * tw * th + ti];
            maps[m * w * h + r * w + q] = s / 3;
          }
        }
      }
    }
  }
  return { y, maps };
}

let busy: Promise<unknown> = Promise.resolve();

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === "load") busy = load(m.ep).catch((err) => postMessage({ type: "error", msg: String(err) }));
  if (m.type === "run") {
    busy = busy.then(async () => {
      try {
        const t0 = performance.now();
        const { y, maps } = await run(m.x, m.w, m.h);
        postMessage({ type: "done", id: m.id, y, maps, ms: performance.now() - t0, ep }, { transfer: [y.buffer, maps.buffer] });
      } catch (err) {
        postMessage({ type: "error", msg: String(err) });
      }
    });
  }
};
