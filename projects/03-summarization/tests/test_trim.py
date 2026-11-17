import json

import pytest

pytest.importorskip("tokenizers")
from summ import trim  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the market rose on monday as investors cheered the news",
    "police said the suspect was arrested on tuesday near the station",
    "the president said the plan would cut taxes for families",
] * 20


def make():
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(
        vocab_size=700,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    t.train_from_iterator(corpus, tr)
    return t


def test_trimmed_tokenizer_matches_on_seen_text():
    t = make()
    tj = json.loads(t.to_str())
    ids = [i for x in corpus[:4] for i in t.encode(x).ids]
    cnt = trim.Counter(ids)
    keep, mer = trim.pick(tj, cnt, 400)
    assert keep[:4] == [0, 1, 2, 3]
    assert len(keep) < len(tj["model"]["vocab"])
    t2 = Tokenizer.from_str(json.dumps(trim.trim_tokenizer(tj, keep, mer)))
    mp = {o: n for n, o in enumerate(keep)}
    for x in corpus[:4]:
        assert [mp[i] for i in t.encode(x).ids] == t2.encode(x).ids
    # text with unseen words still round trips through smaller pieces
    assert t2.decode(t2.encode("zebra quartz jumped").ids) == "zebra quartz jumped"
