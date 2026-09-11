#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_infer.py — run the humanizer from a GGUF file with llama-cpp-python, with the same guards as the
transformers/MLX paths (adaptive anti-copy resample using a logits processor; digits exempt).

    pip install llama-cpp-python            # add CMAKE_ARGS="-DGGML_CUDA=on" / "-DGGML_METAL=on" for GPU
    python gguf_infer.py --gguf humanizer-gemma-4-e4b-Q8_0.gguf --format prompt_format.json draft.txt

Plain llama.cpp / Ollama / LM Studio can run the GGUF too, but they cannot apply the anti-copy penalty; there
the fallback is: if the output copies > 35 % of the draft's 5-grams, simply sample again.
"""
import argparse, json, os, sys
import numpy as np
from llama_cpp import Llama, LogitsProcessorList
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copy_rate import copy_rate


class NoCopy:
    def __init__(self, llm, draft, n=5, penalty=2.0):
        ids = llm.tokenize(draft.encode('utf-8'), add_bos=False); self.n = n; self.p = penalty
        digit = {i for i in set(ids) if any(ch.isdigit() for ch in llm.detokenize([i]).decode('utf-8', 'ignore'))}
        self.prefix = {}
        for i in range(len(ids) - n + 1):
            g = tuple(ids[i:i + n])
            if not (set(g) & digit):
                self.prefix.setdefault(g[:-1], set()).add(g[-1])
        self.plen = None

    def __call__(self, input_ids, scores):
        ids = list(input_ids)
        if self.plen is None:
            self.plen = len(ids)
        gen = ids[self.plen:]
        if len(gen) >= self.n - 1:
            nxt = self.prefix.get(tuple(gen[-(self.n - 1):]))
            if nxt:
                scores = np.array(scores, copy=True); scores[list(nxt)] -= self.p
        return scores


def humanize(llm, pf, draft, thr=0.35, penalty=2.0, temperature=0.85):
    prompt = pf['instr'] + '\n\n' + draft.strip() + pf['sep']
    kw = dict(max_tokens=max(700, int(len(draft.split()) * 2.2) + 200), temperature=temperature, top_p=0.95, stop=['\n\n\n\n'])
    txt = llm(prompt, **kw)['choices'][0]['text'].strip()
    c = copy_rate(draft, txt)['copy_5gram']; retried = False
    if c > thr:
        txt = llm(prompt, logits_processor=LogitsProcessorList([NoCopy(llm, draft, 5, penalty)]), **kw)['choices'][0]['text'].strip()
        c = copy_rate(draft, txt)['copy_5gram']; retried = True
    return txt, {'copy_5gram': round(c, 3), 'resampled_with_penalty': retried}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('draft', nargs='?'); ap.add_argument('--gguf', required=True); ap.add_argument('--format', required=True, help='prompt_format.json')
    ap.add_argument('--n-gpu-layers', type=int, default=-1); ap.add_argument('--ctx', type=int, default=8192); ap.add_argument('--threads', type=int, default=None)
    a = ap.parse_args()
    llm = Llama(model_path=a.gguf, n_ctx=a.ctx, n_gpu_layers=a.n_gpu_layers, n_threads=a.threads, verbose=False)
    pf = json.load(open(a.format)); draft = open(a.draft).read() if a.draft else sys.stdin.read()
    txt, meta = humanize(llm, pf, draft)
    print(txt); print(f"\n[{meta}]", file=sys.stderr)
