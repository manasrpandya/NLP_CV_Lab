import argparse
import json
from pathlib import Path

from tqdm import tqdm

from summ import data
from summ.extractive import lead, textrank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cnndm", choices=list(data.sets))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    ds, sc, tc = data.load(a.data, "test")
    sub = ds.select([int(i) for i in data.subset(len(ds), a.n)])
    pre = "" if a.data == "cnndm" else "xsum-"
    fns = {"lead3": lambda x: lead(x, 3), "textrank3": lambda x: textrank(x, 3)}
    if a.data == "xsum":
        fns = {
            "lead1": lambda x: lead(x, 1),
            "lead3": lambda x: lead(x, 3),
            "textrank1": lambda x: textrank(x, 1),
        }
    for k, f in fns.items():
        out = Path(a.out) / f"{pre}{k}"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "gens.jsonl", "w") as fh:
            for r in tqdm(sub, desc=k, mininterval=30):
                fh.write(json.dumps({"id": r["id"], "hyp": f(r[sc]), "ref": r[tc]}) + "\n")


if __name__ == "__main__":
    main()
