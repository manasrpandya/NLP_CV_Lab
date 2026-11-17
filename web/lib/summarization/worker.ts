// runs the summarizer (and, on request, the small nli model) off the main thread.
import {
  AutoModelForSeq2SeqLM,
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  TextStreamer,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type ProgressInfo,
} from "@huggingface/transformers";

const repo = "manasrpandya/cnndm-summ-onnx";
const nrepo = "Xenova/nli-deberta-v3-xsmall";
env.allowLocalModels = false;

let tok: PreTrainedTokenizer | null = null;
let mdl: PreTrainedModel | null = null;
let ntok: PreTrainedTokenizer | null = null;
let nmdl: PreTrainedModel | null = null;
let ent = 1;
let backend = "";

const prog = (kind: string) => (p: ProgressInfo) => {
  if (p.status === "progress_total") postMessage({ type: "progress", kind, loaded: p.loaded, total: p.total });
};

async function hasGpu() {
  try {
    const g = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu;
    return !!(g && (await g.requestAdapter()));
  } catch {
    return false;
  }
}

async function load(gpu: boolean) {
  tok ??= await AutoTokenizer.from_pretrained(repo);
  if (gpu && (await hasGpu())) {
    try {
      mdl = await AutoModelForSeq2SeqLM.from_pretrained(repo, {
        device: "webgpu",
        dtype: "fp16",
        progress_callback: prog("model"),
      });
      // webgpu can hang on odd adapters, so a short warm-up with a timeout decides if we keep it
      const warm = summarize("The quick brown fox jumps over the lazy dog near the river bank today.", "short", () => {});
      const ok = await Promise.race([warm.then(() => true), new Promise<boolean>((r) => setTimeout(() => r(false), 30000))]);
      if (!ok) throw new Error("webgpu warm-up timed out");
      backend = "webgpu / fp16";
      return;
    } catch (e) {
      console.warn("webgpu failed, falling back to wasm", e);
      mdl = null;
    }
  }
  mdl = await AutoModelForSeq2SeqLM.from_pretrained(repo, {
    device: "wasm",
    dtype: "q8",
    progress_callback: prog("model"),
  });
  backend = "wasm / int8";
}

async function summarize(text: string, len: string, push: (s: string) => void) {
  const enc = tok!(`summarize ${len}: ${text}`, { truncation: true, max_length: 512 });
  const streamer = new TextStreamer(tok!, { skip_prompt: true, skip_special_tokens: true });
  let acc = "";
  const ids: bigint[] = [];
  streamer.token_callback_function = (t: bigint[]) => {
    ids.push(...t);
    // decode the whole thing each time so word pieces and spacing come out right
    acc = tok!.decode(ids, { skip_special_tokens: true }).trim();
    push(acc);
  };
  streamer.callback_function = () => {};
  await mdl!.generate({
    ...enc,
    max_new_tokens: 140,
    num_beams: 1,
    do_sample: false,
    no_repeat_ngram_size: 3,
    streamer,
  });
  return { text: acc, ntok: ids.length, inTok: (enc.input_ids.dims as number[])[1] };
}

async function nli(pairs: [string, string][]) {
  if (!nmdl) {
    ntok = await AutoTokenizer.from_pretrained(nrepo);
    nmdl = await AutoModelForSequenceClassification.from_pretrained(nrepo, {
      device: "wasm",
      dtype: "q8",
      progress_callback: prog("nli"),
    });
    const l = (nmdl.config as { id2label?: Record<string, string> }).id2label ?? {};
    ent = Number(Object.keys(l).find((k) => l[k].toLowerCase() === "entailment") ?? 1);
  }
  const out: number[] = [];
  for (let i = 0; i < pairs.length; i += 16) {
    const b = pairs.slice(i, i + 16);
    const enc = ntok!(
      b.map((p) => p[0]),
      { text_pair: b.map((p) => p[1]), padding: true, truncation: true, max_length: 256 },
    );
    const { logits } = await nmdl(enc);
    const d = logits.data as Float32Array;
    const k = logits.dims[1];
    for (let r = 0; r < b.length; r++) {
      const z = Array.from(d.slice(r * k, (r + 1) * k));
      const m = Math.max(...z);
      const e = z.map((v) => Math.exp(v - m));
      out.push(e[ent] / e.reduce((a, c) => a + c, 0));
    }
  }
  return Float32Array.from(out);
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      await load(!!m.gpu);
      postMessage({ type: "ready", backend });
    } else if (m.type === "gen") {
      const t0 = performance.now();
      const r = await summarize(m.text, m.len, (s) => postMessage({ type: "tok", id: m.id, text: s }));
      postMessage({ type: "done", id: m.id, ms: performance.now() - t0, ...r });
    } else if (m.type === "nli") {
      const z = await nli(m.pairs);
      postMessage({ type: "nli", id: m.id, z });
    }
  } catch (err) {
    postMessage({ type: "error", id: m.id, msg: String(err) });
  }
};
