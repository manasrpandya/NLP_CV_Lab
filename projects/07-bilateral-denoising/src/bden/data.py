import argparse
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[2] / "data"
sigmas = (15, 25, 50)
test_sets = ("set12", "bsd68", "kodak24")
n_val = 32  # last 32 of train400 are held out for param search


def _sparse(dst, paths):
    with tempfile.TemporaryDirectory() as td:
        run = lambda *a: subprocess.run(a, cwd=td, check=True, capture_output=True)  # noqa: E731
        run("git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
            "https://github.com/cszn/DnCNN.git", ".")
        run("git", "sparse-checkout", "set", *paths.values())
        for name, sub in paths.items():
            (dst / name).mkdir(parents=True, exist_ok=True)
            for f in sorted((Path(td) / sub).glob("*.png")):
                (dst / name / f.name).write_bytes(f.read_bytes())


def _kodak(dst):
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(1, 25):
        p = dst / f"kodim{i:02d}.png"
        if not p.exists():
            urllib.request.urlretrieve(f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png", p)


def _sidd(dst, n=320):
    import lmdb
    from huggingface_hub import snapshot_download

    p = Path(snapshot_download("talib-sid/sidd-val", repo_type="dataset", allow_patterns=["val/*/data.mdb"]))
    for kind, sub in (("gt", "gt_crops.lmdb"), ("noisy", "input_crops.lmdb")):
        (dst / kind).mkdir(parents=True, exist_ok=True)
        env = lmdb.open(str(p / "val" / sub), readonly=True, lock=False, subdir=True)
        with env.begin() as txn:
            # 1280 crops from 40 scenes, take every 4th so all scenes are covered
            for i in range(0, 1280, 1280 // n):
                buf = txn.get(f"ValidationBlocksSrgb_{i}".encode())
                (dst / kind / f"{i:04d}.png").write_bytes(buf)


def fetch():
    if not (root / "train400").exists():
        _sparse(root, {
            "train400": "TrainingCodes/DnCNN_TrainingCodes_v1.0/data/Train400",
            "set12": "testsets/Set12",
            "bsd68": "testsets/BSD68",
        })
    _kodak(root / "kodak24")
    if not (root / "sidd").exists():
        _sidd(root / "sidd")
    for d in sorted(root.iterdir()):
        print(d.name, len(list(d.rglob("*.png"))))


def read(p):
    # grayscale float32 in [0, 1]
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32) / 255


def files(name):
    if name == "val":
        return sorted((root / "train400").glob("*.png"))[-n_val:]
    if name == "train":
        return sorted((root / "train400").glob("*.png"))[:-n_val]
    return sorted((root / name).glob("*.png"))


def noisy(name, sig, seed=0):
    # list of (clean, noisy) pairs with fixed noise, sig on the 0-255 scale
    if name == "sidd":
        return [(read(p), read(root / "sidd" / "noisy" / p.name)) for p in files("sidd/gt")]
    out = []
    for i, p in enumerate(files(name)):
        x = read(p)
        rng = np.random.default_rng(seed * 1000 + sig * 7 + i)
        out.append((x, (x + rng.normal(0, sig / 255, x.shape)).astype(np.float32)))
    return out


if __name__ == "__main__":
    argparse.ArgumentParser(description="download and cache all datasets under data/").parse_args()
    fetch()
