import { active, ekman, groupScores, hue, names } from "@/lib/emotion-classification/meta";

const c = 130;
const r0 = 34;
const R = 104;
// order picked so neighbouring sectors stay distinguishable for colour-blind readers
const ring = ["joy", "surprise", "anger", "sadness", "disgust", "fear"];

const pt = (a: number, r: number) => [c + r * Math.sin(a), c - r * Math.cos(a)];

function wedge(a0: number, a1: number, r: number) {
  const [x0, y0] = pt(a0, r0);
  const [x1, y1] = pt(a0, r);
  const [x2, y2] = pt(a1, r);
  const [x3, y3] = pt(a1, r0);
  return `M${x0},${y0} L${x1},${y1} A${r},${r} 0 0 1 ${x2},${y2} L${x3},${y3} A${r0},${r0} 0 0 0 ${x0},${y0}Z`;
}

// ekman wheel: one sector per group (petal = strongest member), member emotions as spokes, neutral in the middle
export function Wheel({ p }: { p: number[] }) {
  const g = groupScores(p);
  const on = active(p);
  const seg = (2 * Math.PI) / ring.length;
  return (
    <svg viewBox="0 0 260 260" className="mx-auto w-full max-w-[320px]" role="img" aria-label="ekman wheel">
      {ring.map((k, j) => {
        const a0 = j * seg - seg / 2;
        const ms = ekman[k];
        const [lx, ly] = pt(j * seg, R + 16);
        return (
          <g key={k}>
            <path d={wedge(a0 + 0.02, a0 + seg - 0.02, R)} fill="none" stroke="var(--line)" />
            <path
              d={wedge(a0 + 0.02, a0 + seg - 0.02, r0 + (R - r0) * g[k])}
              fill={hue[k]}
              fillOpacity={0.22}
              className="transition-all duration-300"
            />
            {ms.map((n, m) => {
              const i = names.indexOf(n);
              const a = a0 + ((m + 0.5) * seg) / ms.length;
              const [x1, y1] = pt(a, r0);
              const [x2, y2] = pt(a, r0 + (R - r0) * p[i]);
              return (
                <line key={n} x1={x1} y1={y1} x2={x2} y2={y2} stroke={hue[k]} strokeWidth={on[i] ? 3 : 1.5}
                  strokeOpacity={on[i] ? 1 : 0.55} strokeLinecap="round">
                  <title>{`${n} ${(p[i] * 100).toFixed(0)}%`}</title>
                </line>
              );
            })}
            <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" fontSize={11}
              fill={g[k] > 0.3 ? hue[k] : "var(--mute)"} fontWeight={g[k] > 0.3 ? 600 : 400}>
              {k}
            </text>
          </g>
        );
      })}
      <circle cx={c} cy={c} r={r0 - 4} fill={hue.neutral} fillOpacity={0.1 + 0.5 * g.neutral} />
      <text x={c} y={c} textAnchor="middle" dominantBaseline="middle" fontSize={10} fill="var(--ink)">
        neutral
      </text>
    </svg>
  );
}
