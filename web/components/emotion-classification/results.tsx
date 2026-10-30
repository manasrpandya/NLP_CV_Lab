import { data } from "@/lib/emotion-classification/meta";

type row = {
  k: string;
  label: string;
  macro_f1: number;
  micro_f1: number;
  ekman_macro_f1: number;
  ood_acc: number;
  ece_raw: number;
  ece_cal: number;
  mb?: number;
};

const f = (v?: number, d = 3) => (v === undefined ? "" : v.toFixed(d));

export function Results() {
  const rows = data.rows as row[];
  if (!rows.length) return null;
  return (
    <section className="mt-14">
      <h2 className="font-serif text-3xl">results</h2>
      <p className="mt-1 max-w-2xl text-sm text-mute">
        GoEmotions test set (5427 reddit comments, 28 labels). thresholds and temperature tuned on validation.
        ood = zero-shot accuracy on the dair-ai/emotion twitter test set, mapped to ekman groups.
      </p>
      <div className="mt-4 overflow-x-auto rounded-lg border border-line">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="bg-card text-left text-xs text-mute">
            <tr>
              <th className="px-3 py-2 font-normal">model</th>
              <th className="px-3 py-2 text-right font-normal">macro-f1</th>
              <th className="px-3 py-2 text-right font-normal">micro-f1</th>
              <th className="px-3 py-2 text-right font-normal">ekman f1</th>
              <th className="px-3 py-2 text-right font-normal">ood acc</th>
              <th className="px-3 py-2 text-right font-normal">ece raw / cal</th>
              <th className="px-3 py-2 text-right font-normal">size mb</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((r) => (
              <tr key={r.k} className={`border-t border-line ${r.k === "student-int8" ? "text-acc" : ""}`}>
                <td className="px-3 py-1.5 font-sans">{r.label}</td>
                <td className="px-3 py-1.5 text-right">{f(r.macro_f1)}</td>
                <td className="px-3 py-1.5 text-right">{f(r.micro_f1)}</td>
                <td className="px-3 py-1.5 text-right">{f(r.ekman_macro_f1)}</td>
                <td className="px-3 py-1.5 text-right">{f(r.ood_acc)}</td>
                <td className="px-3 py-1.5 text-right">
                  {f(r.ece_raw)} / {f(r.ece_cal)}
                </td>
                <td className="px-3 py-1.5 text-right">{r.mb ? r.mb.toFixed(0) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
