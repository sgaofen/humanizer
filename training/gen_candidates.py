#!/usr/bin/env python3
"""best-of-N 候选生成(post-SFT 阶段,2026-09-06):对 inputs jsonl 里每条 draft 用 base+adapter 采 N 个候选。
批量:B 条 prompt × N 个 return 一起 generate;分片:--shard k/M;断点续跑:已写 key 跳过。输出 jsonl {key,src,cands:[...]}。
用法: python hpc3/gen_candidates.py --adapter runs/small_E18f/adapter --inputs data/claude_inputs_dpo.jsonl --out data/cands_E18f_s0.jsonl --shard 0/4 --n 4 --bs 4"""
import argparse, json, os, sys, torch, time
HT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HT); sys.path.insert(0, f'{HT}/hpc3')
from promptfmt import load_for_model, legacy_prompt
from gen4b_pure import trim_tail
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import LogitsProcessor, LogitsProcessorList
COPY_PENALTY=float(os.environ.get('COPY_PENALTY','0')); COPY_N=int(os.environ.get('COPY_N','5'))
class NoCopyProcessor(LogitsProcessor):
    def __init__(self, input_ids, n, penalty, tok):
        self.n=n; self.p=penalty; ids=input_ids.tolist()
        digit=set(i for i in set(ids) if any(c.isdigit() for c in tok.decode([i])))
        self.prefix={}
        for i in range(len(ids)-n+1):
            g=tuple(ids[i:i+n])
            if not (set(g)&digit): self.prefix.setdefault(g[:-1],set()).add(g[-1])
        self.plen=None
    def __call__(self, input_ids, scores):
        if self.plen is None: self.plen=input_ids.shape[1]
        for b in range(input_ids.shape[0]):
            gen=input_ids[b,self.plen:].tolist()
            if len(gen)<self.n-1: continue
            nxt=self.prefix.get(tuple(gen[-(self.n-1):]))
            if nxt: scores[b, torch.tensor(sorted(nxt),device=scores.device)]-=self.p
        return scores
ap = argparse.ArgumentParser()
ap.add_argument('--adapter', required=True); ap.add_argument('--inputs', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=4); ap.add_argument('--bs', type=int, default=4); ap.add_argument('--shard', default='0/1')
ap.add_argument('--pre_adapter', default='', help='DPO 产物用:先把 SFT 适配器合并进底座再挂 --adapter'); ap.add_argument('--base', default='google/gemma-4-E4B'); ap.add_argument('--temp', type=float, default=0.85); ap.add_argument('--max_new', type=int, default=1100)
a = ap.parse_args()
if COPY_PENALTY > 0: a.bs = 1   # 惩罚按每条草稿建 n-gram 表,只能单条批
k, m = map(int, a.shard.split('/'))
rows = [json.loads(l) for l in open(a.inputs) if l.strip()]
rows = [r for i, r in enumerate(rows) if i % m == k]
done = set()
# 09-10:守护取消失败时两个探针会同时往一个输出文件追加(E20gd4c_s1 多出 484 行);用锁文件互斥,后来者直接退出
_lock=a.out+'.lock'; _me=os.environ.get('SLURM_JOB_ID','')
if os.path.exists(_lock):
    _other=open(_lock).read().strip()
    if _other and _other!=_me and os.system(f'squeue -j {_other} -h -o %T 2>/dev/null | grep -qE "RUNNING|PENDING"')==0:
        print(f'cands_lock: {a.out} 由作业 {_other} 持有且仍在跑,本作业 {_me} 退出', flush=True); raise SystemExit(0)
open(_lock,'w').write(_me)
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)['key'])
        except Exception: pass
todo = [r for r in rows if r['key'] not in done]
print(f'shard {k}/{m}: {len(rows)} 条, 已做 {len(done)}, 待做 {len(todo)}', flush=True)
tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = 'left'
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map='cuda')
if a.pre_adapter:
    print('先合并 SFT 适配器', a.pre_adapter, flush=True)
    model = PeftModel.from_pretrained(model, a.pre_adapter).merge_and_unload()
model = PeftModel.from_pretrained(model, a.adapter); model.eval()
try: build_fn, style = load_for_model(a.adapter)
except Exception: build_fn, style = legacy_prompt, 'legacy'
print('格式', style, flush=True)
t0 = time.time(); ng = 0
with open(a.out, 'a') as f:
    for s in range(0, len(todo), a.bs):
        batch = todo[s:s + a.bs]
        prompts = [build_fn(r['draft']) for r in batch]
        ids = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2600).to('cuda')
        lp = None
        if COPY_PENALTY > 0:
            dids = tok(batch[0]['draft'], return_tensors='pt')['input_ids'][0]
            lp = LogitsProcessorList([NoCopyProcessor(dids, COPY_N, COPY_PENALTY, tok)])
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new, do_sample=True, temperature=a.temp, top_p=0.95,
                                 num_return_sequences=a.n, pad_token_id=tok.pad_token_id, logits_processor=lp)
        L = ids['input_ids'].shape[1]
        texts = tok.batch_decode(out[:, L:], skip_special_tokens=True)
        for i, r in enumerate(batch):
            cands = [trim_tail(t.strip(), r['draft']) for t in texts[i * a.n:(i + 1) * a.n]]
            f.write(json.dumps({'key': r['key'], 'src': r['src'], 'cands': cands, 'mode': 'penalty' if COPY_PENALTY > 0 else 'plain'}, ensure_ascii=False) + '\n'); f.flush()
        ng += len(batch)
        print(f'{ng}/{len(todo)} {time.time()-t0:.0f}s', flush=True)
print('GEN_CANDS_DONE', a.out, flush=True)
