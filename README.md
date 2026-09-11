# humanizer

[English](README.md) | [中文](README.zh.md)

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

## Samples

One English and one Chinese example; more in [docs/samples.md](docs/samples.md).

### Email to a professor / 给教授的邮件

**Draft (Claude) / 草稿:**

> Dear Professor Lin,
> 
> I'm following up on the dataset access request I submitted on October 14th. I understand these reviews take time, and I appreciate you looking into it.
> 
> To give a bit more context: my analysis only uses the 2019–2021 subset, and I'm interested purely in the temporal patterns, so I won't need any of the personally identifying fields. If it would help, I'm happy to send a more detailed methodology document, or to walk through the scope of the request in person at whatever time suits you.
> 
> Please let me know if there's anything else you need from me to move the approval along. Thank you again for your time.
> 
> Best regards,
> Wei

**Humanizer output / 改写:** *(5-gram copy 0.14)*

> Dear Professor Lin,
> 
> Just following up on the access request for the data set that I submitted on October 14th. I know that the review process takes time and I appreciate the time you're taking to look into this.
> 
> To provide a little more context, I am only using the 2019-2021 subset for my analysis and I am only interested in the temporal patterns so I won't need any personally identifiable fields. If you would like, I can provide a more detailed methodology document or we can go through the scope of the request in person at your convenience.
> 
> Please let me know if there is anything else that I need to provide to help with the approval process. Thanks again for your time.
> 
> Best regards,
> Wei

### 中文邮件 / Chinese email

**Draft (Claude) / 草稿:**

> 李老师您好,
> 
> 关于上周提到的数据访问申请,我想再补充几点说明。
> 
> 首先,本次分析只涉及 2019 至 2021 年的时间序列子集,不需要任何身份识别字段,所以不会触及隐私数据。其次,考虑到审批通常需要两到三周,我已经把实验计划相应顺延,整体进度不会受影响。另外,如果您对数据的使用方式还有疑问,我可以提供一份更详细的方法说明,或者找个您方便的时间当面沟通。
> 
> 希望这件事能尽快推进。再次感谢您的时间和帮助。
> 
> 祝好
> 小王

**Humanizer output / 改写:** *(5-gram copy 0.31)*

> 李老师您好，
> 
> 上周关于数据访问申请的事情补充说明一下。
> 
> 1、本次只使用2019-2021的时间序列子集，不涉及任何身份识别字段，所以不存在涉及隐私数据的问题；
> 
> 2、考虑到审批周期大概需要两到三周，所以已经将实验计划顺延，不影响整体进度；
> 
> 3、如果对数据的使用方法还有疑问，我可以再提供一个更详细的方法说明，或者找个时间见面沟通。
> 
> 希望能尽快推进。再次感谢您的时间和帮助！
> 
> 祝好
> 小王

## Usage

### GGUF (llama.cpp, Ollama, LM Studio, llama-cpp-python) — recommended

Files in the `gguf/` folder of `jialinyyzz/humanizer-gemma-4-e4b`: `Q8_0` (8.0 GB), `Q6_K` (6.2 GB) and `bf16` (14.9 GB). Q8_0 and Q6_K are within judge noise of bf16 on fidelity. **Q5_K_M and Q4_K_M are not published**: Q5_K_M triples critical fidelity errors (17/62 vs 6) and Q4_K_M degenerates into gibberish on this model, like the MLX 4-bit builds (see `docs/QUALITY.md`).

```bash
pip install llama-cpp-python        # CMAKE_ARGS="-DGGML_METAL=on" (Mac) or "-DGGML_CUDA=on"
python humanizer/gguf_infer.py --gguf humanizer-gemma-4-e4b-Q8_0.gguf --format prompt_format.json draft.txt
```

`gguf_infer.py` applies the anti-copy guard through a logits processor. Plain `llama-cli`, Ollama and LM Studio run the same file but cannot apply the penalty; with them, resample if the output still copies more than about a third of the draft.

### transformers (CUDA), merged bf16

```bash
python humanizer/hf_infer.py --model jialinyyzz/humanizer-gemma-4-e4b draft.txt
```

### MLX (Apple silicon), bf16 only

`humanizer/mlx_nocopy_server.py` + `humanizer/humanize.py` serve the merged bf16 weights with the guard. Note: mlx_lm's Gemma 4 loader rejects the 54 unused k/v tensors of the 18 shared-KV layers in the HF checkpoint; strip `layers.24–41.self_attn.(k_proj|v_proj|k_norm)` before loading. MLX 4-/6-bit quantisation of this model is not usable (see `docs/QUALITY.md`); use the GGUF quants instead.

### Prompt format

The model is a base-model continuation, not a chat model. The exact prompt is shipped in `prompt_format.json` next to the weights and must be reproduced byte for byte:

```
<instruction (6 lines, see prompt_format.json)>

<draft>

### Rewritten:

```

Generation stops at EOS. Sampling: temperature 0.85, top-p 0.95.

## Weights

* `jialinyyzz/humanizer-gemma-4-e4b` — one repo holds every variant: merged bf16 in transformers format at the root (SFT + DPO merged into the base), and llama.cpp GGUF files (Q8_0, Q6_K, bf16; Q5_K_M and Q4_K_M withheld — fidelity collapses below 6-bit) plus `prompt_format.json` under `gguf/`.

Both derive from `google/gemma-4-E4B` and are provided under the Gemma Terms of Use (see `NOTICE`). Code in this repository is Apache-2.0.

## Reproducing

`training/train_sft2.py` (LoRA r=16 on the language tower, 1 epoch, effective batch 16), then `training/gen_candidates.py` → `training/build_prefs.py --tiers --identity-bad` (needs a GLM API key in `~/.config/zai_key`) → `training/train_dpo.py` (β=0.1, lr 5e-6, 1 epoch). Evaluation: `training/gen_cases.py` with `COPY_PENALTY=2 COPY_N=5 ADAPTIVE_COPY=1 ADAPTIVE_THR=0.35`, then `training/glm_eval_en.py` / `glm_eval_zh.py`.
