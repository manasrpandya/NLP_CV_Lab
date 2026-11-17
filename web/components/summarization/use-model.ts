"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type mstate = {
  phase: "idle" | "loading" | "ready" | "error";
  loaded: number;
  total: number;
  backend: string;
  msg?: string;
};

export type gdone = { text: string; ms: number; ntok: number; inTok: number };

const none: mstate = { phase: "idle", loaded: 0, total: 0, backend: "" };

// thin promise wrapper around the worker. the summarizer loads on first use, the nli model on first fact check.
export function useModel() {
  const w = useRef<Worker | null>(null);
  const pend = useRef(new Map<number, { ok: (m: unknown) => void; no: (e: Error) => void; tok?: (s: string) => void }>());
  const nid = useRef(0);
  const ld = useRef<Promise<void> | null>(null);
  const [st, setSt] = useState<mstate>(none);
  const [nst, setNst] = useState<mstate>(none);

  const worker = useCallback(() => {
    if (w.current) return w.current;
    const wk = new Worker(new URL("../../lib/summarization/worker.ts", import.meta.url), { type: "module" });
    const pd = pend.current;
    wk.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") {
        const f = (s: mstate) => ({ ...s, loaded: m.loaded, total: m.total });
        if (m.kind === "nli") setNst(f);
        else setSt(f);
      } else if (m.type === "tok") pd.get(m.id)?.tok?.(m.text);
      else if (m.type === "ready") pd.get(-1)?.ok(m.backend);
      else if (m.type === "done" || m.type === "nli") {
        pd.get(m.id)?.ok(m);
        pd.delete(m.id);
      } else if (m.type === "error") {
        const p = pd.get(m.id ?? -1);
        if (p) {
          p.no(new Error(m.msg));
          pd.delete(m.id ?? -1);
        }
      }
    };
    wk.onerror = (e) => {
      setSt((s) => ({ ...s, phase: "error", msg: e.message || "worker failed to start" }));
    };
    w.current = wk;
    return wk;
  }, []);

  useEffect(() => {
    const pd = pend.current;
    return () => {
      w.current?.terminate();
      w.current = null;
      pd.clear();
    };
  }, []);

  const load = useCallback(
    () => {
      if (ld.current) return ld.current;
      const wk = worker();
      setSt({ ...none, phase: "loading" });
      ld.current = new Promise<void>((ok, no) => {
        pend.current.set(-1, {
          ok: (b) => {
            setSt((s) => ({ ...s, phase: "ready", backend: b as string }));
            ok();
          },
          no,
        });
        wk.postMessage({ type: "load", id: -1, host: new URLSearchParams(location.search).get("host") });
      }).catch((e) => {
        setSt((s) => ({ ...s, phase: "error", msg: String(e.message ?? e) }));
        ld.current = null;
        throw e;
      });
      return ld.current;
    },
    [worker],
  );

  const gen = useCallback(
    async (text: string, len: string, tok: (s: string) => void) => {
      await load();
      return new Promise<gdone>((ok, no) => {
        const id = ++nid.current;
        pend.current.set(id, { ok: (m) => ok(m as gdone), no, tok });
        w.current?.postMessage({ type: "gen", id, text, len });
      });
    },
    [load],
  );

  const nli = useCallback(
    (pairs: [string, string][]) => {
      worker();
      setNst((s) => (s.phase === "ready" ? s : { ...s, phase: "loading" }));
      return new Promise<Float32Array>((ok, no) => {
        const id = ++nid.current;
        pend.current.set(id, {
          ok: (m) => {
            setNst((s) => ({ ...s, phase: "ready" }));
            ok((m as { z: Float32Array }).z);
          },
          no: (e) => {
            setNst((s) => ({ ...s, phase: "error", msg: e.message }));
            no(e);
          },
        });
        w.current?.postMessage({ type: "nli", id, pairs });
      });
    },
    [worker],
  );

  return { st, nst, load, gen, nli };
}
