"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type mstate = {
  phase: "loading" | "ready" | "error";
  loaded: number;
  total: number;
  backend: string;
  msg?: string;
};

// thin promise wrapper around the inference worker
export function useModel() {
  const w = useRef<Worker | null>(null);
  const pend = useRef(new Map<number, { ok: (z: Float32Array) => void; no: (e: Error) => void }>());
  const nid = useRef(0);
  const [st, setSt] = useState<mstate>({ phase: "loading", loaded: 0, total: 0, backend: "" });

  useEffect(() => {
    const wk = new Worker(new URL("../../lib/emotion-classification/worker.ts", import.meta.url), {
      type: "module",
    });
    const pd = pend.current;
    wk.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") setSt((s) => ({ ...s, loaded: m.loaded, total: m.total }));
      else if (m.type === "ready") setSt((s) => ({ ...s, phase: "ready", backend: m.backend }));
      else if (m.type === "out") {
        pd.get(m.id)?.ok(m.z);
        pd.delete(m.id);
      } else if (m.type === "error") {
        if (pd.has(m.id)) {
          pd.get(m.id)!.no(new Error(m.msg));
          pd.delete(m.id);
        } else setSt((s) => ({ ...s, phase: "error", msg: m.msg }));
      }
    };
    wk.postMessage({ type: "load" });
    w.current = wk;
    return () => {
      wk.terminate();
      pd.clear();
    };
  }, []);

  const run = useCallback(
    (xs: string[]) =>
      new Promise<Float32Array>((ok, no) => {
        const id = ++nid.current;
        pend.current.set(id, { ok, no });
        w.current?.postMessage({ type: "run", id, xs });
      }),
    [],
  );

  return { st, run };
}
