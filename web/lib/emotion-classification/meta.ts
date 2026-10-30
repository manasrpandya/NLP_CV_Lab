import d from "./data.json";

export const names = [
  "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
  "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
  "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
  "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
];

export const ekman: Record<string, string[]> = {
  joy: [
    "joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
    "relief", "pride", "admiration", "desire", "caring",
  ],
  surprise: ["surprise", "realization", "confusion", "curiosity"],
  fear: ["fear", "nervousness"],
  sadness: ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
  disgust: ["disgust"],
  anger: ["anger", "annoyance", "disapproval"],
  neutral: ["neutral"],
};
export const groups = Object.keys(ekman);

// one hue per ekman group (css vars so light and dark get their own steps, see huecss)
export const hue: Record<string, string> = Object.fromEntries(
  ["joy", "surprise", "fear", "sadness", "disgust", "anger", "neutral"].map((g) => [g, `var(--e-${g})`]),
);
const lt = { joy: "#eda100", surprise: "#1baf7a", fear: "#6250d6", sadness: "#2a78d6", disgust: "#008300", anger: "#e34948", neutral: "#8a857b" };
const dk = { joy: "#c98500", surprise: "#199e70", fear: "#9085e9", sadness: "#3987e5", disgust: "#008300", anger: "#e66767", neutral: "#9a958a" };
const vars = (m: Record<string, string>) => Object.entries(m).map(([k, v]) => `--e-${k}:${v};`).join("");
export const huecss = `:root{${vars(lt)}}@media (prefers-color-scheme: dark){:root{${vars(dk)}}}`;

export const groupOf: Record<string, string> = Object.fromEntries(
  groups.flatMap((g) => ekman[g].map((n) => [n, g])),
);
export const color = (n: string) => hue[groupOf[n] ?? n];

export const data = d;
export const temp: number = d.temp;
export const th: number[] = d.th;

const sig = (x: number) => 1 / (1 + Math.exp(-x));

// logits for n texts (flat, n*28) -> calibrated probs per text
export function probs(z: Float32Array) {
  const out: number[][] = [];
  for (let i = 0; i < z.length; i += names.length) {
    out.push(Array.from(z.subarray(i, i + names.length), (v) => sig(v / temp)));
  }
  return out;
}

// labels that pass their tuned threshold, with the same top-1 fallback as the python eval
export function active(p: number[]) {
  const on = p.map((v, i) => v >= th[i]);
  if (!on.some(Boolean)) on[p.reduce((b, v, i) => (v / th[i] > p[b] / th[b] ? i : b), 0)] = true;
  return on;
}

// "how sure" score relative to the threshold, used for sorting and the wheel
export const rel = (p: number[]) => p.map((v, i) => Math.min(1, v / th[i] / 2));

export function groupScores(p: number[]) {
  return Object.fromEntries(groups.map((g) => [g, Math.max(...ekman[g].map((n) => p[names.indexOf(n)]))]));
}
