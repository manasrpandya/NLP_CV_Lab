import res from "./results.json";

// same blind estimate + param mapping as bden/classic.py (method b in the report)

// one channel of an rgba buffer as floats. noise is estimated per channel (the filter is per channel too),
// estimating on luma would shrink independent channel noise by ~0.67
export function chan(px: Uint8Array, w: number, h: number, c: number) {
  const y = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) y[i] = px[4 * i + c] / 255;
  return y;
}

// immerkaer 1996, laplacian-difference mask, returns sigma on the 0-255 scale (calibrated on bsd val)
export function estSigma(y: Float32Array, w: number, h: number) {
  let s = 0;
  for (let r = 1; r < h - 1; r++) {
    for (let c = 1; c < w - 1; c++) {
      const i = r * w + c;
      const v =
        y[i - w - 1] - 2 * y[i - w] + y[i - w + 1] - 2 * y[i - 1] + 4 * y[i] - 2 * y[i + 1] + y[i + w - 1] - 2 * y[i + w] + y[i + w + 1];
      s += Math.abs(v);
    }
  }
  const raw = ((s * Math.sqrt(Math.PI / 2)) / (6 * (w - 2) * (h - 2))) * 255;
  const { a, b } = res.est;
  return Math.max(a * raw + b, 1);
}

const lerp = (x: number, xs: number[], ys: number[]) => {
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) return ys[i - 1] + ((x - xs[i - 1]) / (xs[i] - xs[i - 1])) * (ys[i] - ys[i - 1]);
  }
  return ys[ys.length - 1];
};

// sigma -> cv2 style (d, sig_s, sig_r). the shader radius is floor(d / 2) like cv2
export function autoParams(sig: number) {
  const t = res.auto;
  const sg = t.map((r) => r.sigma);
  const d = 2 * Math.round((lerp(sig, sg, t.map((r) => r.d)) - 1) / 2) + 1;
  return { r: Math.floor(d / 2), ss: lerp(sig, sg, t.map((r) => r.sig_s)), sr: lerp(sig, sg, t.map((r) => r.sig_r)) };
}

export function psnr(a: Uint8Array, b: Uint8Array) {
  let se = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = a[i + c] - b[i + c];
      se += d * d;
    }
    n += 3;
  }
  return 10 * Math.log10((255 * 255) / Math.max(se / n, 1e-10));
}
