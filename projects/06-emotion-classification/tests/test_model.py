import numpy as np
import pytest
import torch
from emo import data, losses
from emo.export import feeds, to_onnx, torch_logits
from emo.labels import multihot, nl
from emo.train import groups
from labcore import parity

xs = ["i love this so much", "this is awful :(", "what?? no way", "ok", "thanks [NAME], really kind of you"]


@pytest.fixture(scope="module")
def tok():
    # tiny byte-level bpe trained in place, so nothing is downloaded
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(xs * 20, vocab_size=300, special_tokens=["<s>", "<pad>", "</s>", "<unk>"])
    return PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )


@pytest.fixture(scope="module")
def mdl(tok):
    from transformers import RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(0)
    cfg = RobertaConfig(
        vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=130, pad_token_id=tok.pad_token_id, num_labels=nl,
        problem_type="multi_label_classification",
    )  # fmt: skip
    return RobertaForSequenceClassification(cfg).eval()


def test_collate_shapes(tok):
    y = multihot([[0], [2, 3], [27], [5], [15]])
    ds = data.TextDs(xs, y, tok, max_len=8, soft=y * 3)
    b = data.collate([ds[i] for i in range(len(ds))], pad_id=tok.pad_token_id)
    n = max(len(e) for e in ds.enc)
    assert n <= 8
    assert b["input_ids"].shape == b["attention_mask"].shape == (5, n)
    assert b["y"].shape == b["soft"].shape == (5, nl)
    short = int(np.argmin([len(e) for e in ds.enc]))
    assert b["input_ids"][short, -1] == tok.pad_token_id and b["attention_mask"][short, -1] == 0


def test_forward_and_losses(tok, mdl):
    f = {k: torch.tensor(v) for k, v in feeds(tok, xs).items()}
    z = mdl(**f).logits
    assert z.shape == (len(xs), nl)
    y = torch.zeros_like(z)
    y[:, 0] = 1
    for los in [losses.bce(z, y), losses.asl(z, y), losses.kd(z, z.detach() * 2)]:
        assert torch.isfinite(los) and los > 0


def test_llrd_groups(mdl):
    gs = groups(mdl, 1e-3, 0.5)
    assert sum(len(g["params"]) for g in gs) == len(list(mdl.parameters()))
    lrs = sorted({g["lr"] for g in gs})
    # head, layer 1, layer 0, embeddings
    assert np.allclose(lrs, [1e-3 * 0.5**k for k in [3, 2, 1, 0]])


def test_onnx_parity(tok, mdl, tmp_path):
    mdl.save_pretrained(tmp_path / "m")
    tok.save_pretrained(tmp_path / "m")
    p = to_onnx(tmp_path / "m", tmp_path / "o")
    assert p.exists() and (tmp_path / "o" / "config.json").exists()
    ok, errs = parity([torch_logits(mdl, tok, xs)], p, feeds(tok, xs), atol=1e-4)
    assert ok, errs


def test_kd_soft_targets_partial():
    # distilling towards your own logits at t=1 should be the minimum over a shift of them
    z = torch.randn(4, nl)
    base = losses.kd(z, z, t=1.0)
    assert base <= losses.kd(z + 0.5, z, t=1.0)
