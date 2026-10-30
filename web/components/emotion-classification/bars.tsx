import { active, color, names, th } from "@/lib/emotion-classification/meta";

// all 28 labels, sorted. the tick on each track is that label's tuned threshold.
export function Bars({ p }: { p: number[] }) {
  const on = active(p);
  const order = names.map((_, i) => i).sort((a, b) => p[b] - p[a]);
  return (
    <ul className="space-y-1">
      {order.map((i) => (
        <li key={names[i]} className="grid grid-cols-[6.5rem_1fr_2.75rem] items-center gap-2 text-sm">
          <span className={on[i] ? "font-medium" : "text-mute"}>{names[i]}</span>
          <span className="relative h-3 overflow-hidden rounded-sm bg-line/60">
            <span
              className="absolute inset-y-0 left-0 rounded-sm transition-[width] duration-300"
              style={{ width: `${p[i] * 100}%`, background: color(names[i]), opacity: on[i] ? 1 : 0.35 }}
            />
            <span className="absolute inset-y-0 w-px bg-ink/50" style={{ left: `${th[i] * 100}%` }} />
          </span>
          <span className="text-right font-mono text-xs text-mute">{(p[i] * 100).toFixed(0)}%</span>
        </li>
      ))}
    </ul>
  );
}
