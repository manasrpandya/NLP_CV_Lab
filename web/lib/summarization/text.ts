// sentence splitting, textrank and source matching, all plain js so the quick method needs no download.
// the splitter follows the python one in projects/03-summarization/src/summ/text.py.

const abbr = /\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|u\.s|u\.k)\.$/i;
const cut = /(?<=[.!?])["')\]]*\s+(?=["'(\[]*[A-Z0-9])/;

export function sents(x: string): string[] {
  const t = x.replace(/\s+/g, " ").trim();
  if (!t) return [];
  const out: string[] = [];
  let cur = "";
  for (const part of t.split(cut)) {
    cur = cur ? `${cur} ${part}` : part;
    if (abbr.test(cur) || cur.split(" ").length < 3) continue;
    out.push(cur);
    cur = "";
  }
  if (cur) {
    if (out.length) out[out.length - 1] += ` ${cur}`;
    else out.push(cur);
  }
  return out;
}

const stop = new Set(
  `a about above after again against all also am an and any are as at be because been before being below between both
  but by can could did do does doing down during each few for from further had has have having he her here hers him his
  how i if in into is it its just me more most my no nor not now of off on once only or other our out over own same she
  should so some such than that the their them then there these they this those through to too under until up very was
  we were what when where which while who whom why will with would you your said says say`
    .split(/\s+/)
    .filter(Boolean),
);

export function toks(x: string): string[] {
  return (x.toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?/g) ?? []).filter((w) => !stop.has(w));
}

// sentences as nodes, tf-idf cosine as edge weights, power iteration. returns the indices of the top k, in order.
export function textrank(ss: string[], k: number, d = 0.85, it = 50): number[] {
  const n = ss.length;
  if (n <= k) return ss.map((_, i) => i);
  const ts = ss.map(toks);
  const df = new Map<string, number>();
  for (const t of ts) for (const w of new Set(t)) df.set(w, (df.get(w) ?? 0) + 1);
  const vec = ts.map((t) => {
    const v = new Map<string, number>();
    for (const w of t) v.set(w, (v.get(w) ?? 0) + 1);
    let nn = 0;
    for (const [w, c] of v) {
      const x = c * (Math.log((1 + n) / (1 + (df.get(w) ?? 0))) + 1);
      v.set(w, x);
      nn += x * x;
    }
    nn = Math.sqrt(nn) || 1;
    for (const [w, c] of v) v.set(w, c / nn);
    return v;
  });
  const w = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (const [t, c] of vec[i]) s += c * (vec[j].get(t) ?? 0);
      w[i][j] = w[j][i] = s;
    }
  const rs = w.map((r) => r.reduce((a, b) => a + b, 0));
  let r = new Float64Array(n).fill(1 / n);
  for (let s = 0; s < it; s++) {
    const r2 = new Float64Array(n).fill((1 - d) / n);
    for (let j = 0; j < n; j++) {
      if (rs[j] === 0) {
        for (let i = 0; i < n; i++) r2[i] += (d * r[j]) / n;
        continue;
      }
      for (let i = 0; i < n; i++) r2[i] += (d * w[j][i] * r[j]) / rs[j];
    }
    let diff = 0;
    for (let i = 0; i < n; i++) diff += Math.abs(r2[i] - r[i]);
    r = r2;
    if (diff < 1e-6) break;
  }
  return Array.from(r.keys())
    .sort((a, b) => r[b] - r[a] || a - b)
    .slice(0, k)
    .sort((a, b) => a - b);
}

// how much of a summary sentence's content (words and word pairs) shows up in a source sentence
export function overlap(a: string, b: string): number {
  const x = toks(a);
  if (!x.length) return 0;
  const y = toks(b);
  const ys = new Set(y);
  const uni = x.filter((w) => ys.has(w)).length / x.length;
  const bg = (t: string[]) => new Set(t.slice(1).map((w, i) => `${t[i]} ${w}`));
  const xb = bg(x);
  const yb = bg(y);
  let bi = 0;
  for (const g of xb) if (yb.has(g)) bi++;
  return 0.6 * uni + 0.4 * (xb.size ? bi / xb.size : 0);
}

// the source sentences a summary sentence most likely came from (best one, plus a close second)
export function origin(sum: string, src: string[], score?: number[]): number[] {
  const s = score ?? src.map((t) => overlap(sum, t));
  const ord = s.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]);
  if (!ord.length || ord[0][0] <= 0) return [];
  return ord.filter(([v], k) => k === 0 || (k < 2 && v >= 0.8 * ord[0][0])).map(([, i]) => i);
}
