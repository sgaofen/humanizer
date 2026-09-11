# humanizer

A small model that rewrites AI-written drafts so they read like a person wrote them, while keeping every fact, number, name and date. Base: `google/gemma-4-E4B` (≈4B effective parameters). Runs locally on a Mac (MLX) or on a CUDA GPU (transformers).

**No detector was used as a reward, a filter, a training signal, or a selection criterion at any stage.** The model was trained only on *how people write*; it passes AI detectors as a side effect of writing like a person, and we report detector numbers below purely as an external check.

## What it does

Input: a draft written by a language model (email, essay, report, paper section, forum post, in English or Chinese).
Output: the same content, rewritten. Facts, numbers, units, dates, names and quotations are meant to survive unchanged; structure, wording, rhythm and register change.

It is a rewriter, not a generator: it will not add claims, examples or padding, and it is trained to keep greetings, sign-offs and subject lines.

## How it was trained

1. **Supervised fine-tuning (28.6k pairs).** Each pair is *(AI draft → human text)*. The human side is always real human writing (bioRxiv/PubMed abstracts, government reports, student essays, corporate and mailing-list email, Reddit and Hacker News posts, Zhihu answers, Chinese formal prose, …). The AI side is a draft of the same content written by a current frontier model (Claude Sonnet, GPT-5-class, GLM-5.3) from the human text, so the model learns the direction *machine → person* on identical content. Synthetic "human" text is never used.
2. **Two rounds of preference optimisation (DPO, 1,898 pairs total).** The SFT model writes six candidates per draft (three plain, three under a decoding-time anti-copy penalty). A separate LLM judge (GLM-5.3) grades each candidate for fidelity only — facts kept, meaning unchanged, nothing added, greeting/sign-off kept, not a verbatim copy — with a severity tier. *Chosen* = a candidate with no fidelity issue and the lowest verbatim overlap with the draft; *rejected* = a candidate with a critical fidelity error or a near-verbatim copy. Nothing about style, length, sentence shape or detector score enters the selection.
3. **Decoding-time guard (inference only).** If a first sample copies more than 35 % of the draft's 5-grams, it is resampled once with a logits penalty on tokens that would complete a 5-gram present in the draft (digit tokens exempt). This removes the "lazy identity copy" failure mode without touching fidelity.

Training scripts are in `training/` (LoRA SFT, DPO, candidate generation, the judge prompts). Training data is not released because the human side comes from sources with mixed licences.

## Evaluation

39-case "daily use" set (`eval_daily/`): essays, reports, paper sections, emails, tweets/Reddit/LinkedIn posts, plus 8 Chinese cases; drafts written by Claude Sonnet. Two samples per case.

| metric (62 English samples) | this model | production baseline (Qwen3.5-4B rewriter) |
|---|---|---|
| verbatim 5-gram copy, median | 0.15 | 0.12 |
| samples copying > 35 % of draft | 0 | 33 / 93 |
| **critical fidelity errors** (judge: reversed meaning, changed number/event) | **10 %** (6/62) | ≈ 30 % |
| minor / none | 19 / 37 | — |
| Chinese cases passing the judge | 10 / 16 | 4 / 16 |
| Originality.ai "human" verdicts *(external check only, never optimised)* | **49 / 62 = 79 %** | 53 / 93 = 57 % |

Human-written originals from the training genres score 9/9 "human" on the same detector; the draft inputs score ≈ 0/62. Weakest genres are paper sections (3/6) and reports (5/8).

### Known failure modes (please read before relying on it)

* About 1 in 10 outputs contains a **meaning flip** that a careful reader would catch: who recommends what ("leadership recommends" → "we recommend to leadership"), an event ("received on Aug 30" → "ordered on Aug 30"), a metric ("ridership rose 12 %" → "ride time rose 12 %"), a comparison reversed, or an invented "as requested". Always proofread numbers, dates and the direction of every claim.
* Subject lines and formal greetings are occasionally dropped.
* Chinese is weaker than English (10/16).
* Drafts under ~120 words are rewritten less reliably.

## Usage

### MLX (Apple silicon)

```bash
pip install mlx-lm
# 8-bit MLX weights (~7.4 GB) or bf16 (~16 GB) from the Hugging Face repos below
python humanizer/mlx_nocopy_server.py --model ./humanizer-gemma-4-e4b-mlx-8bit --port 8104
python humanizer/humanize.py --model-dir ./humanizer-gemma-4-e4b-mlx-8bit --port 8104 draft.txt
```

`mlx_nocopy_server.py` is an OpenAI-style `/v1/completions` server with three extra fields (`copy_penalty`, `copy_n`, `draft`) that implement the anti-copy penalty; `humanize.py` adds the adaptive resample.

### transformers (CUDA)

```bash
python humanizer/hf_infer.py --model jialinyyzz/humanizer-gemma-4-e4b draft.txt
```

### Prompt format

The model is a base-model continuation, not a chat model. The exact prompt is shipped in `prompt_format.json` next to the weights and must be reproduced byte for byte:

```
<instruction (6 lines, see prompt_format.json)>

<draft>

### Rewritten:

```

Generation stops at EOS. Sampling: temperature 0.85, top-p 0.95.

## Weights

* `jialinyyzz/humanizer-gemma-4-e4b` — merged bf16, transformers format (SFT + DPO merged into the base)
* `jialinyyzz/humanizer-gemma-4-e4b-mlx-8bit` — MLX, 8-bit, group size 64 (within judge noise of bf16). No 4- or 6-bit build: 4-bit produces gibberish and 6-bit triples critical fidelity errors on this model, see `docs/QUALITY.md`.

Both derive from `google/gemma-4-E4B` and are provided under the Gemma Terms of Use (see `NOTICE`). Code in this repository is Apache-2.0.

## Reproducing

`training/train_sft2.py` (LoRA r=16 on the language tower, 1 epoch, effective batch 16), then `training/gen_candidates.py` → `training/build_prefs.py --tiers --identity-bad` (needs a GLM API key in `~/.config/zai_key`) → `training/train_dpo.py` (β=0.1, lr 5e-6, 1 epoch). Evaluation: `training/gen_cases.py` with `COPY_PENALTY=2 COPY_N=5 ADAPTIVE_COPY=1 ADAPTIVE_THR=0.35`, then `training/glm_eval_en.py` / `glm_eval_zh.py`.
