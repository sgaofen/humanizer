---
license: gemma
base_model: google/gemma-4-E4B
language:
  - en
  - zh
tags:
  - text-rewriting
  - humanizer
  - paraphrase
  - gemma
  - mlx
pipeline_tag: text-generation
library_name: transformers
---

# humanizer-gemma-4-e4b

One repo, every variant: transformers bf16 at the root; llama.cpp GGUF (Q8_0 / bf16; Q5_K_M and Q6_K pending evaluation, Q4_K_M withheld because 4-bit degenerates on this model) under `gguf/`. Samples, code and the evaluation set: [github.com/sgaofen/humanizer](https://github.com/sgaofen/humanizer).

Rewrites AI-written drafts so they read like a person wrote them, keeping every fact, number, name and date. Fine-tune of `google/gemma-4-E4B`.

**Trained on data that teaches the model how people write. No reinforcement learning or optimisation against any AI detector; it passes detectors naturally, and detector numbers below are only an external check.**

## Training

* **SFT, 28.6k pairs** of *(AI draft → real human text)*. The human side is always genuine human writing (scientific abstracts, government reports, student essays, email, forum posts, Chinese prose); the AI side is a draft of the same content written by a current frontier model from the human text.
* **DPO, 2 rounds, 1,898 pairs.** Candidates written by the model itself; a separate LLM judge grades *fidelity only* (facts kept, meaning unchanged, nothing added, greeting/sign-off kept, not a verbatim copy). Chosen = fidelity-clean with the least verbatim overlap; rejected = critical fidelity error or near-copy. No style, length or detector signal enters selection.
* **Inference guard:** if a sample copies > 35 % of the draft's 5-grams it is resampled once with a logits penalty on draft n-grams (digits exempt).

## Evaluation (39-case daily-use set, 2 samples each)

| 62 English samples | this model | baseline 4B rewriter |
|---|---|---|
| samples copying > 35 % of draft | 0 | 33 / 93 |
| critical fidelity errors (judge) | 10 % | ≈ 30 % |
| Chinese cases passing judge | 10 / 16 | 4 / 16 |
| Originality.ai "human" *(external check only)* | 79 % | 57 % |

GGUF files (Q8_0 / bf16; Q5_K_M and Q6_K pending evaluation, Q4_K_M withheld — 53/62 critical errors, gibberish) are in the `gguf/` folder of this repo; their quality on the same set is in `docs/QUALITY.md` of the GitHub repo. MLX 4-/6-bit quantisation of this model is not usable (Gemma 4 PLE layers); use the GGUF quants on a Mac.

Known failure modes: ~1 in 10 outputs has a meaning flip (who did what, ordered vs received, a metric renamed); subject lines/greetings occasionally dropped; Chinese weaker than English; short drafts (< 120 words) less reliable. **Proofread numbers, dates and the direction of every claim.**

## Prompt format (must match exactly)

Base-model continuation, not chat. `prompt_format.json` in this repo holds the instruction and separator:

```
{instr}

{draft}

### Rewritten:

```

Stop at EOS; temperature 0.85, top-p 0.95.

## Usage

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

repo = "jialinyyzz/humanizer-gemma-4-e4b"
pf = json.load(open(hf_hub_download(repo, "prompt_format.json")))
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda")
draft = open("draft.txt").read()
prompt = pf["instr"] + "\n\n" + draft.strip() + pf["sep"]
ids = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=900, do_sample=True, temperature=0.85, top_p=0.95)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

The full inference path with the anti-copy resample, an MLX server, the evaluation set and the training scripts are on GitHub: `sgaofen/humanizer`.

## Licence

Weights derive from Gemma 4 and are provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Code: Apache-2.0.
