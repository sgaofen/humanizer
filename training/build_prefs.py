#!/usr/bin/env python3
"""post-SFT 偏好对构建(2026-09-06):候选 → GLM 逐条事实判(与 glm_eval_en 同一标准)→ chosen/rejected。
铁律:检测器不参与;判断全靠 GLM;照抄率只用来在「已干净」的候选里排序(偏好,不是门)。
用法: python3 scripts/build_prefs.py --cands 'data/cands_E18f_s*.jsonl' --inputs data/claude_inputs_dpo.jsonl --fmt experiments/dpo/adapter_fmt --out data/prefs_E18f.jsonl [--workers 8]
中间产物 experiments/dpo/cand_verdicts_<tag>.jsonl 可续跑。"""
import argparse,json,os,re,sys,glob,hashlib,threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'hpc3'))
from llm_http import zai
from promptfmt import load_for_model
ap=argparse.ArgumentParser(); ap.add_argument('--cands',required=True); ap.add_argument('--inputs',required=True); ap.add_argument('--fmt',required=True); ap.add_argument('--out',required=True)
ap.add_argument('--workers',type=int,default=8); ap.add_argument('--tag',default='E18f'); ap.add_argument('--tiers',action='store_true',help='两级判定:rejected 只用 critical,chosen 必须 none'); ap.add_argument('--style-profile',default='',help='09-09:真人体裁句长档案 json;chosen 先限定在带内(msl 在 Q1-Q3、超长/短句占比≤Q3),带内再取最低照抄;带内没有则退回最低照抄'); ap.add_argument('--identity-bad',action='store_true',help='GLM 判 identity(几乎照抄)的候选也当 rejected(结构信号,专治照抄模式)'); a=ap.parse_args()
P='''You are a strict reviewer. DRAFT is the original text; OUTPUT is a rewrite that must keep every fact. Output ONLY JSON:
{"facts_all_kept": true/false, "meaning_changed": true/false, "added_content": true/false, "greeting_signoff_kept": true/false, "identity": true/false, "reads_like_person": 1-5, "severity": "critical" | "minor" | "none", "evidence": "<=40 words naming the worst problem, or 'none'"}
severity (judge only what matters to a reader who relies on the text): "critical" = a fact reversed or contradicted; a number, name, date, amount, or deadline changed or dropped; an invented fact, claim, quote, promise, or reason; a request, decision, action item, or who-did-what changed or lost; a completed action turned into an intention (or the reverse); language switched; text truncated. "minor" = only hedging or emphasis shifted (suggests -> shows, mostly -> all, most trusted -> the one trusted), a boundary softened (at least 200 -> more than 200), synonyms with slightly different flavor, added connectives or framing sentences that carry no new facts, reordering. "none" = faithful. Dropping or changing a subject line, greeting, sign-off, signature or contact line is NOT critical and NOT minor for severity purposes (report it only through greeting_signoff_kept).
facts_all_kept: every fact, number, name, date, claim and request of DRAFT appears in OUTPUT (rewording is fine). meaning_changed: OUTPUT states something DRAFT does not (a reversed condition, a claim made stronger/weaker, an opinion turned into a fact, a lost qualifier such as 'significant'). added_content: OUTPUT contains facts, examples, numbers or sentences with new information not in DRAFT (pure connectives do not count). greeting_signoff_kept: if DRAFT has a greeting/sign-off, OUTPUT keeps them (true if DRAFT has none). identity: OUTPUT is essentially a verbatim copy of DRAFT. reads_like_person: 5 = a competent person clearly wrote it, 1 = obviously machine text. Same language as DRAFT is required; if OUTPUT switches language or is truncated/garbled, set facts_all_kept false.
DRAFT:
"""{d}"""
OUTPUT:
"""{o}"""'''
def grams(t,n=5):
    if re.search(r'[一-鿿]',t[:200]):
        s=re.sub(r'\s+','',t); return set(s[i:i+8] for i in range(len(s)-7))
    w=re.findall(r"[A-Za-z0-9']+",t.lower()); return set(tuple(w[i:i+n]) for i in range(len(w)-n+1))
def copy_rate(d,o):
    A,B=grams(d),grams(o); return len(A&B)/max(1,len(B))
inputs={}
for l in open(a.inputs):
    r=json.loads(l); inputs[r['key']]=r
cands={}
for f in glob.glob(a.cands):
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        if r['key'] in cands: cands[r['key']]['cands'].extend(r['cands'])   # 原样 + 反照抄两份候选合并
        else: cands[r['key']]=r
print('输入',len(inputs),'有候选',len(cands),flush=True)
VF=f'experiments/dpo/cand_verdicts_{a.tag}.jsonl'; os.makedirs('experiments/dpo',exist_ok=True)
done={}
if os.path.exists(VF):
    for l in open(VF):
        try: r=json.loads(l); done[(r['key'],r['i'])]=r['verdict']
        except Exception: pass
jobs=[(k,i,c) for k,r in cands.items() for i,c in enumerate(r['cands']) if (k,i) not in done and k in inputs]
print('待判候选',len(jobs),'已判',len(done),flush=True)
lock=threading.Lock()
def judge(j):
    k,i,c=j
    if len(c.strip())<40: v={'facts_all_kept':False,'meaning_changed':True,'added_content':False,'reads_like_person':1,'evidence':'empty/short'}
    else:
        t=zai(P.replace('{d}',inputs[k]['draft']).replace('{o}',c),model='glm-5.3',max_tokens=400,timeout=240) or ''
        m=re.search(r'\{.*\}',t,re.S)
        try: v=json.loads(m.group(0))
        except Exception: v=None
    with lock:
        if v is not None:
            open(VF,'a').write(json.dumps({'key':k,'i':i,'verdict':v},ensure_ascii=False)+'\n'); done[(k,i)]=v
        n=len(done)
        if n%200==0: print(f'已判 {n}',flush=True)
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(judge,jobs))
build_fn,style=load_for_model(a.fmt); print('prompt 格式',style)
def clean(v):
    ok=bool(v and v.get('facts_all_kept') and not v.get('meaning_changed') and not v.get('added_content') and v.get('greeting_signoff_kept',True) and not v.get('identity') and (v.get('reads_like_person') or 0)>=4)
    if a.tiers: ok=ok and v.get('severity','none')=='none'
    return ok
def bad(v):
    if a.identity_bad and v and v.get('identity'): return True
    if a.tiers: return bool(v and v.get('severity')=='critical')
    return bool(v and (v.get('meaning_changed') or v.get('added_content') or not v.get('facts_all_kept')))
PROF=json.load(open(a.style_profile)) if a.style_profile else {}
import re as _re, statistics as _st
def _sl(t):
    sents=[x for x in _re.split(r'(?<=[.!?])\s+|\n+', t) if len(x.split())>=3]; L=[len(x.split()) for x in sents] or [0]
    return _st.mean(L), sum(1 for x in L if x>30)/len(L), sum(1 for x in L if x<9)/len(L)
ALIAS={'daily_email':'email_short','daily_paper':'paper_bio','academic_en':'paper_clean','daily_report':'report_crs','daily_essay':'essay_student','daily_reddit':'reddit_post','daily_linkedin':'blog_personal','daily_tweet':None}
def in_band(src,t):
    pr=PROF.get(src) or (PROF.get(ALIAS[src]) if ALIAS.get(src) else None)
    if not pr: return True
    m,lg,sh=_sl(t); return pr['msl_q1']<=m<=pr['msl_q3'] and lg<=max(pr['long_q3'],0.05) and sh<=max(pr['short_q3'],0.05)
stats={'pairs':0,'no_clean':0,'no_bad':0,'all_clean':0,'style_band':0}; n=0
with open(a.out,'w') as f:
    for k,r in cands.items():
        if k not in inputs: continue
        d=inputs[k]['draft']
        rows=[(i,c,done.get((k,i))) for i,c in enumerate(r['cands']) if (k,i) in done]
        good=[(i,c,v) for i,c,v in rows if clean(v)]; worse=[(i,c,v) for i,c,v in rows if bad(v)]
        if not good: stats['no_clean']+=1; continue
        if not worse: stats['no_bad']+=1; stats['all_clean']+=1; continue
        band=[g for g in good if in_band(r['src'],g[1])] if PROF else []
        if band: stats['style_band']+=1
        ci,ct,cv=min(band or good,key=lambda x:copy_rate(d,x[1]))          # 干净里(优先句长带内)改得最狠的
        ri,rt,rv=max(worse,key=lambda x:(x[2].get('severity')=='critical',bool(x[2].get('identity')),x[2].get('meaning_changed',False),not x[2].get('facts_all_kept',True),x[2].get('added_content',False)))
        f.write(json.dumps({'prompt':build_fn(d),'chosen':ct,'rejected':rt,'key':k,'src':r['src'],'chosen_copy':round(copy_rate(d,ct),3),'rejected_reason':rv.get('evidence','')[:120],'provider':f'dpo:{a.tag}:glm-5.3'},ensure_ascii=False)+'\n'); stats['pairs']+=1
print('偏好对',stats,'→',a.out)
