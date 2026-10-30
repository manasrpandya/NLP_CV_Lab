import numpy as np
import torch

from bden.export import check, export
from bden.model import build


def test_shapes():
    torch.manual_seed(0)
    for name, nmap in [("nbf", 5), ("nbf_noguide", 4), ("nbf_1stage", 3)]:
        mdl = build(name).eval()
        # odd sizes on purpose, the unet has to handle them
        x = torch.rand(2, 1, 37, 50)
        y, m = mdl(x, maps=True)
        assert y.shape == x.shape and m.shape == (2, nmap, 37, 50)
        assert 30_000 < sum(p.numel() for p in mdl.parameters()) < 100_000


def test_onnx_parity(tmp_path):
    torch.manual_seed(0)
    mdl = build("nbf").eval()
    p = export(mdl, tmp_path / "m.onnx")
    rng = np.random.default_rng(0)
    ok, err = check(mdl, p, [rng.random((45, 61), dtype=np.float32), rng.random((96, 96), dtype=np.float32)])
    assert ok, err
    assert p.stat().st_size < 1024 * 1024
