import argparse
import re
from pathlib import Path

from labcore import dump, load

from summ import data, names

# test articles shown as samples on the demo page (row index in the original test split)
picks = {
    863: "space probe",
    872: "smart bullets",
    743: "comedy prank",
    848: "dressage",
}


def samples():
    ds, sc, tc = data.load("cnndm", "test")
    out = []
    for i, name in picks.items():
        r = ds[i]
        x = re.sub(r"[ \t]+", " ", r["article"]).strip()
        out.append({"id": r["id"], "name": name, "text": x, "ref": r["highlights"].replace("\n", " ")})
    return out


def results(m, b, take):
    # rows for the results component on the demo page
    bk = {"ours-medium-b4": "torch-ours-b4", "ours-medium-greedy": "onnx-int8-greedy"}
    out = []
    for k in names.main:
        if k not in m:
            continue
        lab, grp, pm = names.rows[k]
        r = m[k]
        row = {
            "k": k,
            "label": lab if k != "ours-medium-greedy" else "ours, as in the browser",
            "group": grp,
            "rouge1": r["rouge1"],
            "rouge2": r["rouge2"],
            "rougeLsum": r["rougeLsum"],
            "novel_bigram": r["novel_bigram"],
            "words": r["words"],
        }
        for key in ["bertscore", "consistency"]:
            if key in r:
                row[key] = r[key]
        t = b.get(bk.get(k, k), {}).get("mean_ms")
        if t:
            row["ms"] = t
        out.append(row)
    return {"rows": out, "take": take}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../web/lib/summarization")
    ap.add_argument("--samples", action="store_true")
    a = ap.parse_args()
    if a.samples:
        dump(samples(), f"{a.out}/samples.json")
    m = load("results/metrics.json")
    b = load("results/bench.json") if Path("results/bench.json").exists() else {}
    take = load("results/take.json") if Path("results/take.json").exists() else {"rouge": "", "faith": ""}
    dump(results(m, b, take), f"{a.out}/results.json")


if __name__ == "__main__":
    main()
