import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def seed_all(s=0):
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    try:
        import torch

        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def dev():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def tick(out, key="sec"):
    # usage: with tick(d): ...  -> d["sec"] = elapsed
    t0 = time.perf_counter()
    yield
    out[key] = time.perf_counter() - t0


def dump(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def load(p):
    return json.loads(Path(p).read_text())
