import argparse
import re

from labcore import dump

from summ import data

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../web/lib/summarization")
    a = ap.parse_args()
    dump(samples(), f"{a.out}/samples.json")


if __name__ == "__main__":
    main()
