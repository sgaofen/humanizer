#!/usr/bin/env python3
"""集群侧推理生成(2026-09-02 用户规矩:推理测试也在集群做,做完再转 MLX 放本机)。
加载 Qwen3.5-4B-Base + LoRA adapter,对 eval/ood_cases/*.txt 每篇采样 n 次,
守卫与本机 eval/ab_n3.py 完全一致(trim_tail 落款裁剪 / 编数字≥3 降温重打 / 照抄>0.35 重采),
输出 JSON {case: [{text, inv, copy}]},Originality 打分回本机做。
用法: python hpc3/gen_cases.py --adapter runs/small_E1/adapter --out eval/cases_E1.json [--n 3]
"""
import argparse, glob, json, os, sys, torch
HT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HT); sys.path.insert(0, f'{HT}/hpc3')
from promptfmt import load_for_model, legacy_prompt
from gen4b_pure import trim_tail, _invented
from copy_rate import copy_rate
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import LogitsProcessor, LogitsProcessorList

class NoCopyProcessor(LogitsProcessor):
    """解码期反照抄(2026-09-07):凡是会把「已生成的最后 n-1 个 token + 候选 token」凑成输入里出现过的 n-gram 的候选,logit 减 penalty。
    豁免:n-gram 里含数字 token(数字/日期/金额必须原样)。不是检测器,不看输出分数,只限制逐字复制的长度。COPY_PENALTY=0 关闭。"""
    def __init__(self, input_ids, n=4, penalty=3.0, tok=None):
        self.n=n; self.p=penalty; self.grams=set(); self.tok=tok
        ids=input_ids.tolist()
        digit=set(i for i in set(ids) if tok is not None and any(c.isdigit() for c in tok.decode([i])))
        for i in range(len(ids)-n+1):
            g=tuple(ids[i:i+n])
            if not (set(g)&digit): self.grams.add(g)
        self.prefix={}
        for g in self.grams: self.prefix.setdefault(g[:-1],set()).add(g[-1])
        self.plen=input_ids.shape[0]
    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            gen=input_ids[b, self.plen:].tolist()
            if len(gen)<self.n-1: continue
            nxt=self.prefix.get(tuple(gen[-(self.n-1):]))
            if nxt:
                idx=torch.tensor(sorted(nxt),device=scores.device); scores[b, idx]-=self.p
        return scores

ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=3); ap.add_argument('--base', default='google/gemma-4-E4B')
ap.add_argument('--cases', default=f'{HT}/eval/ood_cases')
ap.add_argument('--pre_adapter', default='', help='DPO 产物用:先把 SFT 适配器合并进底座,再挂 --adapter(与 train_dpo.py 的 merge_and_unload 对应)')
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map='cuda')
if a.pre_adapter:
    print('先合并 SFT 适配器', a.pre_adapter, flush=True)
    model = PeftModel.from_pretrained(model, a.pre_adapter).merge_and_unload()
model = PeftModel.from_pretrained(model, a.adapter); model.eval()
try:
    build_fn, style = load_for_model(a.adapter)
except Exception:
    build_fn, style = legacy_prompt, 'legacy(默认:adapter 目录无 prompt_format.json)'
print('格式', style, flush=True)

COPY_PENALTY = float(os.environ.get('COPY_PENALTY', '0')); COPY_N = int(os.environ.get('COPY_N', '4'))
def gen(prompt, temp, draft_ids=None):
    ids = tok(prompt, return_tensors='pt').to('cuda')
    lp = LogitsProcessorList([NoCopyProcessor(draft_ids if draft_ids is not None else ids['input_ids'][0], n=COPY_N, penalty=COPY_PENALTY, tok=tok)]) if (COPY_PENALTY > 0 and not (ADAPTIVE_COPY and draft_ids is None)) else None
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=1200, do_sample=True, temperature=temp, top_p=0.95,
                             pad_token_id=tok.eos_token_id, logits_processor=lp)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()

RESAMPLE = os.environ.get('RESAMPLE', '0') == '1'   # 审计 2026-09-06:copy/invented best-of-2 重采样是隐藏的尺子门,默认关闭;设 RESAMPLE=1 复现旧行为
ADAPTIVE_COPY = os.environ.get('ADAPTIVE_COPY', '0') == '1'   # 自适应:先不加惩罚;第一发照抄超过 ADAPTIVE_THR(默认 0.5)才用 COPY_PENALTY 重采一发
ADAPTIVE_THR = float(os.environ.get('ADAPTIVE_THR', '0.5'))   # 09-09:Gemma 栈阈值 .5 只触发 6/62,试 .35
def gen_one(prompt, draft):
    dids = tok(draft, return_tensors='pt')['input_ids'][0].to('cuda') if COPY_PENALTY > 0 else None
    retried = None
    if ADAPTIVE_COPY and COPY_PENALTY > 0:
        txt = trim_tail(gen(prompt, 0.85, None), draft)
        if copy_rate(draft, txt)['copy_5gram'] > ADAPTIVE_THR:
            txt = trim_tail(gen(prompt, 0.85, dids), draft); retried = 'adaptive_copy'
        return txt, retried
    txt = trim_tail(gen(prompt, 0.85, dids), draft)
    if RESAMPLE:
        if _invented(txt, draft) >= 3:
            t2 = trim_tail(gen(prompt, 0.5), draft)
            if _invented(t2, draft) < _invented(txt, draft): txt = t2; retried = 'inv'
        c = copy_rate(draft, txt)['copy_5gram']
        if c > 0.35:
            t3 = trim_tail(gen(prompt, 0.85), draft)
            if copy_rate(draft, t3)['copy_5gram'] < c: txt = t3; retried = 'copy'
    return txt, retried

out = {}
for f in sorted(glob.glob(f'{a.cases}/*.txt')):
    case = os.path.basename(f)[:-4]; draft = open(f, encoding='utf-8').read().strip()
    prompt = build_fn(draft); rows = []
    for j in range(a.n):
        txt, retried = gen_one(prompt, draft)
        rows.append({'text': txt, 'inv': _invented(txt, draft), 'copy': round(copy_rate(draft, txt)['copy_5gram'], 3), 'retried': retried})
        print(f'{case:14s} #{j+1} 编={rows[-1]["inv"]} 抄={rows[-1]["copy"]} 词={len(txt.split())}', flush=True)
    out[case] = rows
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
print('GEN_CASES_DONE', a.out)
