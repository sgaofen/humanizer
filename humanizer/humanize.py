#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""humanize.py — rewrite an AI-written draft so it reads like a person wrote it, keeping every fact.

Talks to a local completion server (see mlx_nocopy_server.py) and applies the same decoding-time
guards the model was evaluated with:

  * prompt format is read from <model-dir>/prompt_format.json (must match training, byte for byte)
  * adaptive anti-copy: if the first sample copies > THR of the draft's 5-grams, resample once with
    the NoCopy logits penalty (digits are exempt so numbers/dates survive)

Usage:
    python humanize.py --model-dir ./humanizer-gemma-4-e4b-mlx-4bit --port 8104 draft.txt
    cat draft.txt | python humanize.py --model-dir ... --port 8104
"""
import argparse, json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copy_rate import copy_rate


def load_format(model_dir):
    pf = json.load(open(os.path.join(model_dir, 'prompt_format.json')))
    return lambda draft: pf['instr'] + '\n\n' + draft.strip() + pf['sep']


def complete(port, prompt, draft, temperature=0.85, penalty=None, copy_n=5, timeout=900):
    body = {'prompt': prompt, 'max_tokens': max(700, int(len(draft.split()) * 2.2) + 200),
            'stop': ['\n\n\n\n'], 'temperature': temperature, 'top_p': 0.95}
    if penalty:
        body.update({'copy_penalty': penalty, 'copy_n': copy_n, 'draft': draft})
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)['choices'][0]['text'].strip()


def humanize(draft, model_dir, port, thr=0.35, penalty=2.0, copy_n=5, temperature=0.85):
    build = load_format(model_dir); prompt = build(draft)
    txt = complete(port, prompt, draft, temperature)
    c = copy_rate(draft, txt)['copy_5gram']; retried = False
    if c > thr:
        txt = complete(port, prompt, draft, temperature, penalty=penalty, copy_n=copy_n)
        c = copy_rate(draft, txt)['copy_5gram']; retried = True
    return txt, {'copy_5gram': round(c, 3), 'resampled_with_penalty': retried,
                 'words_in': len(draft.split()), 'words_out': len(txt.split())}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('draft', nargs='?', help='text file; omit to read stdin')
    ap.add_argument('--model-dir', required=True); ap.add_argument('--port', type=int, default=8104)
    ap.add_argument('--thr', type=float, default=0.35); ap.add_argument('--penalty', type=float, default=2.0)
    ap.add_argument('--json', action='store_true', help='print JSON with stats')
    a = ap.parse_args()
    draft = open(a.draft).read() if a.draft else sys.stdin.read()
    txt, meta = humanize(draft, a.model_dir, a.port, a.thr, a.penalty)
    if a.json:
        print(json.dumps({'text': txt, **meta}, ensure_ascii=False, indent=1))
    else:
        print(txt); print(f"\n[copy_5gram={meta['copy_5gram']} resampled={meta['resampled_with_penalty']}]", file=sys.stderr)
