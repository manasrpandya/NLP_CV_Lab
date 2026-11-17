import numpy as np
import pytest
import torch

pytest.importorskip("optimum")
from summ.export import Ort, build  # noqa: E402
from transformers import BartConfig, BartForConditionalGeneration  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    cfg = BartConfig(
        vocab_size=120,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        forced_bos_token_id=0,
    )
    m = BartForConditionalGeneration(cfg).eval()
    d = tmp_path_factory.mktemp("tiny")
    m.save_pretrained(d / "pt")
    return m, build(d / "pt", d / "out")


def test_layout(tiny):
    _, d = tiny
    names = {p.name for p in d.iterdir()}
    assert {"encoder_model.onnx", "decoder_model_merged.onnx"} <= names
    assert {"encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx"} <= names
    assert "decoder_with_past_model.onnx" not in names


def test_fp32_generation_matches_torch(tiny):
    m, d = tiny
    o = Ort(d, q=False)
    rs = np.random.RandomState(1)
    for n in [7, 15, 30]:
        ids = rs.randint(4, 120, (1, n)).astype(np.int64)
        mask = np.ones_like(ids)
        ref = m.generate(
            torch.tensor(ids),
            attention_mask=torch.tensor(mask),
            num_beams=1,
            do_sample=False,
            max_length=20,
            no_repeat_ngram_size=3,
        )[0].tolist()
        got = o.gen(ids, mask, max_len=20)
        assert got[: len(ref) - 1] == ref[: len(ref) - 1]


def test_encoder_states_close(tiny):
    m, d = tiny
    ids = np.random.RandomState(2).randint(4, 120, (1, 12)).astype(np.int64)
    mask = np.ones_like(ids)
    with torch.no_grad():
        ref = m.model.encoder(
            input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask)
        ).last_hidden_state
    got = Ort(d, q=False).encode(ids, mask)
    assert np.abs(ref.numpy() - got).max() < 1e-3


def test_int8_runs_and_stays_close(tiny):
    _, d = tiny
    ids = np.random.RandomState(3).randint(4, 120, (1, 12)).astype(np.int64)
    mask = np.ones_like(ids)
    a, b = Ort(d, q=False).encode(ids, mask), Ort(d, q=True).encode(ids, mask)
    assert np.abs(a - b).max() < 0.5
    assert all(0 <= t < 120 for t in Ort(d, q=True).gen(ids, mask, max_len=15))
