#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本机(MLX)日常集生成(2026-09-10):对 eval/ood_cases_daily/*.txt 每篇采样 N 次,走本机 mlx_nocopy_server,
自适应反照抄与集群 hpc3/gen_cases.py 同一逻辑(首发照抄 > ADAPTIVE_THR 才带惩罚重采一发),
输出 eval/cases_<tag>.json 与集群同格式(text/copy/retried),之后照常用 glm_eval_en.py / glm_eval_zh.py 判定。

用途:① 验证 Mac 部署与集群输出同质;② bf16 vs 4bit 量化对比。
用法: python scripts/local_eval_daily.py --port 8104 --model-dir models/m_gemma_gd4 --tag E20gd4ad_mac [--n 2] [--thr 0.35]
"""
import argparse, glob, json, os, sys, time, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, f'{ROOT}/scripts')
from promptfmt import load_for_model
from copy_rate import copy_rate

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, required=True); ap.add_argument('--model-dir', required=True); ap.add_argument('--tag', required=True)
ap.add_argument('--n', type=int, default=2); ap.add_argument('--thr', type=float, default=0.35)
ap.add_argument('--penalty', type=float, default=2.0); ap.add_argument('--copy-n', type=int, default=5)
ap.add_argument('--cases', default=f'{ROOT}/eval/ood_cases_daily'); ap.add_argument('--temp', type=float, default=0.85)
a = ap.parse_args()
build, style = load_for_model(a.model_dir); print('格式', style, flush=True)

def call(prompt, draft, penalty=False):
    body = {'prompt': prompt, 'max_tokens': max(700, int(len(draft.split()) * 2.2) + 200), 'stop': ['\n\n\n\n'],
            'temperature': a.temp, 'top_p': 0.95}
    if penalty:
        body.update({'copy_penalty': a.penalty, 'copy_n': a.copy_n, 'draft': draft})
    r = urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/completions', json.dumps(body).encode(),
                                                      {'Content-Type': 'application/json'}), timeout=900)
    return json.load(r)['choices'][0]['text'].strip()

out = {}; t0 = time.time()
for fpath in sorted(glob.glob(f'{a.cases}/*.txt')):
    case = os.path.basename(fpath)[:-4]; draft = open(fpath).read(); prompt = build(draft); rows = []
    for i in range(a.n):
        txt = call(prompt, draft); retried = None
        c = copy_rate(draft, txt)['copy_5gram']
        if c > a.thr:
            txt = call(prompt, draft, penalty=True); retried = 'adaptive_copy'; c = copy_rate(draft, txt)['copy_5gram']
        rows.append({'text': txt, 'copy': round(c, 3), 'inv': 0, 'retried': retried, 'words': len(txt.split())})
        print(f'{case:28s} #{i+1} 抄={c:.2f} 词={len(txt.split())}{" 重采" if retried else ""}', flush=True)
    out[case] = rows
json.dump(out, open(f'{ROOT}/eval/cases_{a.tag}.json', 'w'), ensure_ascii=False, indent=1)
print(f'LOCAL_GEN_DONE {a.tag} {len(out)} 例 {time.time()-t0:.0f}s', flush=True)
