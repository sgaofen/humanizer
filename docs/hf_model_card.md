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

One repo, every variant: transformers bf16 at the root; llama.cpp GGUF (Q8_0 / Q6_K / bf16; Q5_K_M and Q4_K_M withheld — fidelity collapses below 6-bit on this model) under `gguf/`. Samples, code and the evaluation set: [github.com/sgaofen/humanizer](https://github.com/sgaofen/humanizer).

Rewrites AI-written drafts so they read like a person wrote them, keeping every fact, number, name and date. Fine-tune of `google/gemma-4-E4B`.

**v2 (2026-09-14): SFT + DPO + GRPO. Trained on data that teaches the model how people write, then reinforced on fidelity only. No detector was used as a reward, filter, signal or selection criterion at any stage; it passes detectors naturally, and detector numbers below are only an external check.**

## Training

* **SFT, 28.6k pairs** of *(AI draft → real human text)*. The human side is always genuine human writing (scientific abstracts, government reports, student essays, email, forum posts, Chinese prose); the AI side is a draft of the same content written by a current frontier model from the human text.
* **DPO, 2 rounds, 1,898 pairs.** Candidates written by the model itself; a separate LLM judge grades *fidelity only* (facts kept, meaning unchanged, nothing added, greeting/sign-off kept, not a verbatim copy). Chosen = fidelity-clean with the least verbatim overlap; rejected = critical fidelity error or near-copy. No style, length or detector signal enters selection.
* **GRPO, 300 steps (v2).** LoRA on the merged SFT + DPO model, 8 samples per draft. Reward = LLM-judged fidelity against an atomic fact list of the draft (critical −3.0, minor −0.15, invented −2.0, reversed −2.0, dropped format element / greeting −1.5), −1.0 if the draft's register is normalised, and a superlinear reuse ramp (reuse = max(verbatim 5-gram copy, content-masked syntactic-skeleton recall); free below 0.31, −4 at full copy). Errors are credited to the sentence that carries them. Release checkpoint picked from a 25-step ladder on the held-out set. No detector anywhere in the reward.
* **Inference guard:** if a sample copies > 35 % of the draft's 5-grams it is resampled once with a logits penalty on draft n-grams (digits exempt).

## Evaluation (39-case daily-use set, 2 samples each)

Two independent judges (GLM-5.3 and gpt-5.6); an error counts only if both report it. v1 re-graded under the same protocol.

| 62 English samples | **v2** | v1 (SFT + DPO) | baseline 4B rewriter |
|---|---|---|---|
| samples copying > 35 % of draft | 0 | 1 | 33 / 93 |
| reuse (verbatim ∨ syntactic skeleton), median | 0.29 | 0.34 | — |
| critical fidelity errors | **0 / 62** | 3 / 62 | ≈ 30 % |
| minor / clean | 20 / 42 | 15 / 44 | — |
| format element dropped | 5 / 62 | 12 / 62 | — |
| Chinese cases passing judge | 13 / 16 | 11 / 16 | 4 / 16 |
| Originality.ai "human" *(external check only)* | **85 %** | 81 % | 57 % |

GGUF files (Q8_0 / Q6_K / bf16; Q5_K_M withheld at 17/62 critical errors, Q4_K_M withheld as gibberish) are in the `gguf/` folder of this repo; their quality on the same set is in `docs/QUALITY.md` of the GitHub repo. MLX 4-/6-bit quantisation of this model is not usable (Gemma 4 PLE layers); use the GGUF quants on a Mac.

Known failure modes (v2): dropped qualifiers in about 1 in 3 outputs ("an estimated 4.2 %" → "4.2 %", "suggest" → "conclude"); meaning flips were v1's main failure (~1 in 10) and did not appear in v2's 62 samples, but the set is small; an occasional garbled sentence — resample; Chinese weaker than English (13/16); short drafts (< 120 words) less reliable. **Proofread numbers, dates and the direction of every claim.** GGUF fidelity numbers in `docs/QUALITY.md` were measured on v1; v2 quants were built with the same recipe.

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

Version history: v2 (2026-09-14, this revision) adds the GRPO stage; v1 (2026-09-10, SFT + DPO) is in the commit history of this repo. The full inference path with the anti-copy resample, an MLX server, the evaluation set and the training scripts are on GitHub: `sgaofen/humanizer`.

## Licence

Weights derive from Gemma 4 and are provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Code: Apache-2.0.
