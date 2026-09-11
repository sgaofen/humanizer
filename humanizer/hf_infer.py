#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hf_infer.py — reference inference with 🤗 transformers (CUDA), same guards as the MLX path.

    python hf_infer.py --model jialinyyzz/humanizer-gemma-4-e4b draft.txt

Requires transformers >= 5.x with Gemma 4 support. The model card explains the prompt format;
this script reads it from prompt_format.json shipped with the weights.
"""
import argparse, json, os, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from huggingface_hub import hf_hub_download
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copy_rate import copy_rate


class NoCopyProcessor(LogitsProcessor):
    """Decoding-time anti-copy: any token that would complete an n-gram present in the draft gets
    logit -= penalty. N-grams containing a digit token are exempt so numbers, dates and amounts survive.
    This is not a detector and never looks at any detector score; it only limits verbatim runs."""
    def __init__(self, draft_ids, n=5, penalty=2.0, tok=None):
        self.n = n; self.p = penalty; ids = draft_ids.tolist()
        digit = {i for i in set(ids) if tok is not None and any(ch.isdigit() for ch in tok.decode([i]))}
        self.prefix = {}
        for i in range(len(ids) - n + 1):
            g = tuple(ids[i:i + n])
            if not (set(g) & digit):
                self.prefix.setdefault(g[:-1], set()).add(g[-1])
        self.plen = None

    def __call__(self, input_ids, scores):
        if self.plen is None:
            self.plen = input_ids.shape[1]
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.plen:].tolist()
            if len(gen) < self.n - 1:
                continue
            nxt = self.prefix.get(tuple(gen[-(self.n - 1):]))
            if nxt:
                scores[b, torch.tensor(sorted(nxt), device=scores.device)] -= self.p
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('draft', nargs='?'); ap.add_argument('--model', default='jialinyyzz/humanizer-gemma-4-e4b')
    ap.add_argument('--thr', type=float, default=0.35); ap.add_argument('--penalty', type=float, default=2.0)
    a = ap.parse_args()
    draft = open(a.draft).read() if a.draft else sys.stdin.read()
    pf_path = os.path.join(a.model, 'prompt_format.json') if os.path.isdir(a.model) else hf_hub_download(a.model, 'prompt_format.json')
    pf = json.load(open(pf_path)); prompt = pf['instr'] + '\n\n' + draft.strip() + pf['sep']
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map='cuda').eval()

    def gen(penalty=False):
        ids = tok(prompt, return_tensors='pt').to('cuda')
        lp = LogitsProcessorList([NoCopyProcessor(tok(draft, return_tensors='pt')['input_ids'][0], n=5, penalty=a.penalty, tok=tok)]) if penalty else None
        out = model.generate(**ids, max_new_tokens=max(700, int(len(draft.split()) * 2.2) + 200), do_sample=True,
                             temperature=0.85, top_p=0.95, logits_processor=lp)
        return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    txt = gen(); c = copy_rate(draft, txt)['copy_5gram']
    if c > a.thr:
        txt = gen(penalty=True); c = copy_rate(draft, txt)['copy_5gram']
    print(txt); print(f'\n[copy_5gram={c:.3f}]', file=sys.stderr)


if __name__ == '__main__':
    main()
