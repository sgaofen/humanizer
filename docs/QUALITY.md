# Quality check of the released model (2026-09-10)

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

* MLX bf16 on the Mac reproduces the cluster result within judge noise on critical errors; it shows a few more "minor" verdicts (hedging shifts) — likely sampling variance plus numerics, not a deployment bug.
* 8-bit is the smallest usable quantisation: within noise of bf16 on every metric.
* Plain 4-bit, mixed 4/6-bit, and 4-bit with embeddings left in bf16 all produce gibberish, so the damage is in the 4-bit linear layers themselves, not the embeddings. 6-bit is coherent but nearly triples critical fidelity errors (17 vs 6–9). For this model on MLX, 8-bit is the floor; no 4- or 6-bit build is published.

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
