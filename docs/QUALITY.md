# Quality check of the released model

## v2 (2026-09-14): SFT + DPO + GRPO

Same 39-case daily-use set, two samples per case. Fidelity now graded by **two independent judges** (GLM-5.3 and gpt-5.6-luna); an error counts only if both report it, and severity is the milder of the two. Every row below, including v1, is graded under this protocol. `reuse` = max(verbatim 5-gram copy, syntactic-skeleton 5-gram recall with content words masked) — it catches sentence-by-sentence synonym substitution that verbatim copy misses. Decoding identical for every row (temperature 0.85, top-p 0.95, adaptive anti-copy resample).

| run | critical | minor | clean | format element dropped | reuse (median) | Chinese pass | Originality.ai "human" *(external check only)* |
|---|---|---|---|---|---|---|---|
| v1: SFT + DPO (previous release) | 3 | 15 | 44 | 12 | 0.34 | 11 / 16 | 50 / 62 = 81 % |
| RL on fidelity only (R3) | 2 | 7 | 53 | 3 | 0.39 | — | 32 / 62 = 52 % |
| + register term (R4) | 2 | 9 | 51 | 3 | 0.37 | 12 / 16 | 45 / 62 = 73 % |
| + reuse ramp, weight 8, step 175 (R6-s175) | 1 | 25 | 36 | 3 | 0.28 | 15 / 16 | 53 / 62 = 85 % |
| + reuse ramp, weight 8, step 200 (R6-s200) | 3 | 19 | 40 | 5 | 0.30 | — | 51 / 62 = 82 % |
| + reuse ramp, weight 8, step 300 (R6 final) | 7 | 17 | 38 | 7 | 0.28 | — | not scanned |
| **+ reuse ramp, weight 4, step 300 (R7 final) — released as v2** | **0** | 20 | 42 | 5 | 0.29 | 13 / 16 | **53 / 62 = 85 %** |
| reuse ramp weight 4, step 175 (R7-s175) | 3 | 14 | 45 | 1 | 0.34 | — | not scanned |

What the ladder shows:

* **RL on fidelity alone makes the model write more "properly"** — contractions expanded, slang formalised, hedges regularised — and the detector pass rate collapses (81 % → 52 %) even though fact errors improve. Policy entropy falls from 0.68 to 0.49 over training; within one run the higher-entropy half of the outputs passes the detector 32 points more often than the lower-entropy half. A single reward term that penalises normalising the draft's register (R4) restores most of it (73 %) and keeps the fidelity gain.
* **The reuse ramp lowers reuse from 0.37 to 0.28–0.30 within ~150 steps and then stops falling**; training beyond that point only adds fact errors (R6: 1 critical at step 175, 7 at step 300). Halving the ramp weight (R7) keeps the reuse gain and removes the late-stage damage (0 critical at step 300). The trade-off that remains is *minor* errors — dropped qualifiers — which rise from 15 to 20.
* Detector pass rate is a plateau, not a point: R6-s175 85 %, R6-s200 82 %, R7 85 %, all above v1's 81 % (paired McNemar tests vs v1 are not individually significant at n = 62).
* Pure-SFT variants with different data recipes, generated under the same decoding, all have far worse fidelity (5–12 critical, 30–40 minor): the fact preservation comes from DPO/RL, not from the SFT recipe.

GGUF quants of v2 (Q8_0, Q6_K, bf16) were built with the same recipe as v1; the per-quant table below was measured on v1 and has not been re-run on v2.

## v1 (2026-09-10): SFT + DPO

Model: `humanizer-gemma-4-e4b` = google/gemma-4-E4B + LoRA SFT (28.6k pairs) + DPO ×2 (1,898 pairs), merged. Decoding: temperature 0.85, top-p 0.95, adaptive anti-copy (resample once with the 5-gram penalty if the first sample copies > 35 %).

Evaluation set: `eval_daily/` — 39 cases (31 English, 8 Chinese), drafts written by Claude Sonnet across essays, reports, paper sections, emails, tweets/Reddit/LinkedIn. Two samples per case. Fidelity is graded by GLM-5.3 with a severity tier (critical / minor / none); the judge re-run on identical outputs moves counts by ±3–5, so treat differences below that as noise.

## Same weights, three runtimes

| run | copy (median 5-gram) | > 35 % copy | length ratio | critical | minor | none | Chinese pass |
|---|---|---|---|---|---|---|---|
| cluster, HF transformers bf16 (A100/RTX6000) | 0.15 | 0 / 62 | 1.02 | 6 | 19 | 37 | 10 / 16 |
| Mac, MLX bf16 (judge pass 1 / 2) | 0.13 | 0 / 62 | 1.01 | 8 / 7 | 29 / 27 | 25 / 28 | 11 / 16 |
| Mac, MLX 8-bit (group 64, 8.5 bpw, 7.4 GB) | 0.19 | 1 / 62 | 1.03 | 9 | 25 | 28 | 11 / 16 |
| Mac, MLX 4-bit (4.5 bpw) — **not released** | 0.00 | 0 | 1.35 | 62 | 0 | 0 | 0 / 16 |
| Mac, MLX mixed 4/6-bit — **not released** | 0.00 | 0 | 1.18 | 62 | 0 | 0 | 0 / 16 |
| Mac, MLX 4-bit linear layers only, embeddings kept bf16 — **not released** | 0.00 | 0 | 1.17 | 62 | 0 | 0 | 0 / 16 |
| Mac, MLX 6-bit (6.5 bpw, 5.7 GB) — **not released** | 0.13 | 0 | 1.02 | 17 | 24 | 21 | 9 / 16 |
| cluster, llama.cpp GGUF Q8_0 via `llama-server` (no anti-copy guard; plain resample when copy > 0.35 — what Ollama / LM Studio users get) | 0.24 | 7 / 62 | 1.00 | 7 | 20 | 25 | 12 / 16 |
| cluster, llama.cpp GGUF Q4_K_M — **not released** | 0.00 | 0 | ≈2.4 (runaway) | 53 | 0 | 0 | 0 / 16 |
| cluster, llama.cpp GGUF Q8_0, llama-cpp-python with the anti-copy guard (`gguf_infer.py` path) | 0.16 | 0 / 62 | 1.00 | 7 | 18 | 37 | 12 / 16 |
| cluster, llama.cpp GGUF Q6_K, with guard | 0.16 | 0 / 62 | 1.00 | 7 | 24 | 31 | 13 / 16 |
| cluster, llama.cpp GGUF Q6_K via `llama-server` (no guard) | 0.20 | 4 / 62 | 1.00 | 11 | 27 | 24 | 14 / 16 |
| cluster, llama.cpp GGUF Q5_K_M, with guard — **not released** | 0.17 | 1 / 62 | 1.00 | 17 | 29 | 16 | 9 / 16 |
| cluster, llama.cpp GGUF Q5_K_M via `llama-server` — **not released** | 0.17 | 2 / 62 | 1.00 | 11 | 24 | 27 | 10 / 16 |
| cluster, GGUF Q4_K_M with per-layer embeddings / altup / embedding / output tensors kept at Q8_0 — **not released** | 0.17 | 3 / 62 | 1.00 | 24 | 20 | 18 | 9 / 16 |

* MLX bf16 on the Mac reproduces the cluster result within judge noise on critical errors; it shows a few more "minor" verdicts (hedging shifts) — likely sampling variance plus numerics, not a deployment bug.
* 8-bit is the smallest usable quantisation: within noise of bf16 on every metric.
* Plain 4-bit, mixed 4/6-bit, and 4-bit with embeddings left in bf16 all produce gibberish, so the damage is in the 4-bit linear layers themselves, not the embeddings. 6-bit is coherent but nearly triples critical fidelity errors (17 vs 6–9). For this model on MLX, 8-bit is the floor; no 4- or 6-bit build is published.
* **GGUF Q8_0 is fine**: critical errors and Chinese pass rate are within judge noise of bf16. Its higher copy rate (0.24 median, 7 samples above 0.35) is the missing decoder-side guard, not the quantisation — `llama-server` / Ollama / LM Studio cannot apply the logits penalty, so the eval only resampled plainly. `humanizer/gguf_infer.py` (llama-cpp-python) restores the guard.
* **GGUF Q4_K_M is broken the same way MLX 4-bit is**: outputs drift into digit runs and repeated fragments (median 442 words for ~180-word drafts, copy ≈ 0, 53/62 critical, 0/16 Chinese). Confirmed on two backends (llama-server and llama-cpp-python 0.3.35). It is not published. Keeping Gemma 4's per-layer embedding / altup / embedding / output tensors at 8-bit and quantising only the attention and MLP linears to 4-bit removes the gibberish (outputs are coherent, normal length) but still quadruples critical errors (24 vs 6), so 4-bit linears alone already damage fidelity.
* **Q6_K is the smallest release**: 7/62 critical and 13/16 Chinese with the guard — within noise of bf16 and Q8_0 — at 6.2 GB. **Q5_K_M is withheld**: 17/62 critical with the guard (11 without), the same collapse pattern as MLX 6-bit. The published GGUF set is therefore Q8_0, Q6_K and bf16.

## Reading the six critical errors (cluster run)

All six are *direction/event flips* that a careful reader would catch; numbers themselves were preserved.

| case | draft | output |
|---|---|---|
| corporate memo | "leadership recommends moving 15 % of the Q4 budget" | "we therefore recommend that leadership move 15 %" (who recommends, reversed) |
| essay on AI art | "permitting AI submissions could favor students with greater technological resources" | "may disadvantage students with more technical resources" (comparison reversed) |
| email to landlord | "Could we schedule a repair visit sometime this week?" | adds "as requested" (invented) |
| handoff email | "as my internship wraps up this week" | "before leaving for my internship this week" (start/end reversed) |
| complaint email | "a standing desk I received on August 30th" | "I ordered a standing desk … on August 30" (event changed) |
| policy memo | "ridership on the route rose 12 %" | "ride time on Route 22 increased 12 %" (metric renamed) |

Minor verdicts are mostly hedging shifts ("suggest" → "indicate", "significantly" dropped) and dropped subject lines.

## External check (not used in training or selection)

Originality.ai on the 62 English samples: 49 pass as human (79 %); the production 4B rewriter it replaces: 53 / 93 (57 %); human-written originals from the training genres: 9 / 9; the Claude drafts themselves: ≈ 0. Weakest genres: paper sections 3/6, reports 5/8.
