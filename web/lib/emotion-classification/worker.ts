// runs the distilled goemotions model off the main thread. returns raw logits, calibration happens in the ui.
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type ProgressInfo,
} from "@huggingface/transformers";

const repo = "manasrpandya/goemotions-distil-onnx";
env.allowLocalModels = false;

let tok: PreTrainedTokenizer | null = null;
let mdl: PreTrainedModel | null = null;
let backend = "";

async function hasGpu() {
  try {
    const g = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu;
    return !!(g && (await g.requestAdapter()));
  } catch {
    return false;
  }
}

async function load(forceWasm = false) {
  const progress_callback = (p: ProgressInfo) => {
    if (p.status === "progress_total") postMessage({ type: "progress", loaded: p.loaded, total: p.total });
  };
  tok ??= await AutoTokenizer.from_pretrained(repo);
  if (!forceWasm && (await hasGpu())) {
    try {
      mdl = await AutoModelForSequenceClassification.from_pretrained(repo, {
        device: "webgpu",
        dtype: "fp16",
        progress_callback,
      });
      backend = "webgpu / fp16";
      await infer(["warm up"]);
      return;
    } catch (e) {
      console.warn("webgpu failed, falling back to wasm", e);
    }
  }
  mdl = await AutoModelForSequenceClassification.from_pretrained(repo, {
    device: "wasm",
    dtype: "q8",
    progress_callback,
  });
  backend = "wasm / int8";
}

async function infer(xs: string[]) {
  const enc = tok!(xs, { padding: true, truncation: true, max_length: 128 });
  const { logits } = await mdl!(enc);
  return Float32Array.from(logits.data as Float32Array);
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      await load();
      postMessage({ type: "ready", backend });
    } else if (m.type === "run") {
      const out: number[] = [];
      // small batches keep latency per call low and memory flat on phones
      for (let i = 0; i < m.xs.length; i += 16) out.push(...(await infer(m.xs.slice(i, i + 16))));
      postMessage({ type: "out", id: m.id, z: Float32Array.from(out) });
    }
  } catch (err) {
    postMessage({ type: "error", id: m.id, msg: String(err) });
  }
};
