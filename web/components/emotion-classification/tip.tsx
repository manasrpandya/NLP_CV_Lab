"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// small "?" that shows a note on hover (mouse) or on tap (touch). tap again or tap elsewhere to close.
export function Tip({ children, label = "more info" }: { children: React.ReactNode; label?: string }) {
  const b = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState<{ x: number; y: number; up: boolean; w: number } | null>(null);
  const [pin, setPin] = useState(false);

  const open = () => {
    const r = b.current!.getBoundingClientRect();
    const w = Math.min(280, window.innerWidth - 16);
    const x = Math.max(8, Math.min(r.left + r.width / 2 - w / 2, window.innerWidth - w - 8));
    const up = r.bottom + 180 > window.innerHeight;
    setPos({ x, y: up ? r.top - 6 : r.bottom + 6, up, w });
  };
  const close = useCallback(() => {
    setPos(null);
    setPin(false);
  }, []);

  useEffect(() => {
    if (!pos) return;
    const out = (e: Event) => {
      if (!b.current?.contains(e.target as Node)) close();
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && close();
    document.addEventListener("pointerdown", out);
    document.addEventListener("keydown", esc);
    window.addEventListener("scroll", close, { passive: true });
    return () => {
      document.removeEventListener("pointerdown", out);
      document.removeEventListener("keydown", esc);
      window.removeEventListener("scroll", close);
    };
  }, [pos, close]);

  return (
    <>
      <button
        ref={b}
        type="button"
        aria-label={label}
        aria-expanded={!!pos}
        onClick={(e) => {
          e.stopPropagation();
          if (pin) close();
          else {
            open();
            setPin(true);
          }
        }}
        onPointerEnter={(e) => e.pointerType === "mouse" && !pos && open()}
        onPointerLeave={(e) => e.pointerType === "mouse" && !pin && setPos(null)}
        className={`inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border align-middle font-mono text-[10px] leading-none transition-colors ${
          pos ? "border-acc bg-acc text-bg" : "border-mute/60 text-mute hover:border-acc hover:text-acc"
        }`}
      >
        ?
      </button>
      {pos && (
        <span
          role="tooltip"
          style={{ left: pos.x, top: pos.y, width: pos.w, transform: pos.up ? "translateY(-100%)" : undefined }}
          className="fixed z-50 rounded-md border border-line bg-card p-3 text-left text-xs font-normal leading-relaxed text-ink shadow-lg"
        >
          {children}
        </span>
      )}
    </>
  );
}

// a row that slides open to show more text, for details that most visitors can skip
export function More({ title, children, open }: { title: string; children: React.ReactNode; open?: boolean }) {
  return (
    <details open={open} className="group border-b border-line">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 py-3 text-sm hover:text-acc [&::-webkit-details-marker]:hidden">
        <span>{title}</span>
        <span className="font-mono text-mute transition-transform group-open:rotate-45">+</span>
      </summary>
      <div className="space-y-2 pb-4 text-sm leading-relaxed text-mute">{children}</div>
    </details>
  );
}
