import argparse
from pathlib import Path

from labcore import load


def card_onnx(m, o):
    r = m["ours-medium-greedy"]
    return f"""---
license: apache-2.0
datasets: [abisee/cnn_dailymail]
language: [en]
pipeline_tag: summarization
library_name: transformers.js
base_model: facebook/bart-base
tags: [onnx, summarization, bart, cnn-dailymail, length-control]
---

# cnndm-summ-onnx

bart-base fine-tuned on 40k CNN/DailyMail articles with a length prefix (`summarize short: `,
`summarize medium: ` or `summarize long: ` in front of the article), exported to ONNX for transformers.js.
the vocabulary was trimmed from 50,265 to {o["vocab"]:,} tokens (the ones the training set uses, plus
everything bpe needs to build them), which together with int8 quantization brings the model to
{o["onnx_int8"]["mb"]:.0f} mb.

`onnx/encoder_model_quantized.onnx` and `onnx/decoder_model_merged_quantized.onnx` are the int8 files
(`dtype: "q8"` in transformers.js). the pytorch weights (full vocabulary) are in
[cnndm-summ](https://huggingface.co/manasrpandya/cnndm-summ).

results on a fixed random 2,000-article subset of the cnn/dailymail test set, medium prefix: rouge-1
{m["ours-medium-b4"]["rouge1"]:.1f} / rouge-2 {m["ours-medium-b4"]["rouge2"]:.1f} / rouge-lsum
{m["ours-medium-b4"]["rougeLsum"]:.1f} with beam 4, {r["rouge1"]:.1f} / {r["rouge2"]:.1f} /
{r["rougeLsum"]:.1f} with greedy decoding (what the demo uses). on a 300 article check the int8 onnx model
gives rouge-1 {o["onnx_int8"]["rouge1"]:.1f} against {o["torch"]["rouge1"]:.1f} for the pytorch model.

```js
import {{ pipeline }} from "@huggingface/transformers";
const s = await pipeline("summarization", "manasrpandya/cnndm-summ-onnx", {{ dtype: "q8" }});
const out = await s("summarize medium: " + article, {{ max_new_tokens: 140, no_repeat_ngram_size: 3 }});
```

only the first 512 tokens of the article are read. the model was trained on news, so other kinds of text work
less well. live demo: https://nlp-cv-lab.vercel.app/p/summarization, code:
https://github.com/manasrpandya/NLP_CV_Lab/tree/main/projects/03-summarization
"""


def card_pt(m):
    r = m["ours-medium-b4"]
    return f"""---
license: apache-2.0
datasets: [abisee/cnn_dailymail]
language: [en]
pipeline_tag: summarization
base_model: facebook/bart-base
tags: [summarization, bart, cnn-dailymail, length-control]
---

# cnndm-summ

bart-base fine-tuned for one epoch on 40k CNN/DailyMail 3.0.0 articles (fp16, label smoothing 0.1,
input cut at 512 tokens). the input starts with a length prefix: `summarize short: `, `summarize medium: ` or
`summarize long: `, chosen from the tertiles of the reference length (41 and 55 words).

on a fixed random 2,000-article test subset with the medium prefix and beam 4: rouge-1 {r["rouge1"]:.1f},
rouge-2 {r["rouge2"]:.1f}, rouge-lsum {r["rougeLsum"]:.1f}. the browser version (smaller vocabulary, int8)
is in [cnndm-summ-onnx](https://huggingface.co/manasrpandya/cnndm-summ-onnx).

```python
from transformers import pipeline
s = pipeline("summarization", model="manasrpandya/cnndm-summ")
s("summarize medium: " + article, num_beams=4, max_length=142, no_repeat_ngram_size=3)
```

code: https://github.com/manasrpandya/NLP_CV_Lab/tree/main/projects/03-summarization
"""


def main():
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="runs/onnx")
    a = ap.parse_args()
    m, o = load("results/metrics.json"), load("results/onnx.json")
    api = HfApi()
    out = Path(a.onnx)
    (out / "README.md").write_text(card_onnx(m, o))
    repo = "manasrpandya/cnndm-summ-onnx"
    api.create_repo(repo, exist_ok=True)
    keep = [
        "README.md",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "onnx/encoder_model_quantized.onnx",
        "onnx/decoder_model_merged_quantized.onnx",
    ]
    for f in keep:
        api.upload_file(path_or_fileobj=out / f, path_in_repo=f, repo_id=repo)
    api.upload_file(
        path_or_fileobj=card_pt(m).encode(), path_in_repo="README.md", repo_id="manasrpandya/cnndm-summ"
    )
    print("pushed")


if __name__ == "__main__":
    main()
