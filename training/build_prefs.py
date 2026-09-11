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
ap.add_argument('--workers',type=int,default=8); ap.add_argument('--tag',default='E18f'); ap.add_argument('--tiers',action='store_true',help='两级判定:rejected 只用 critical,chosen 必须 none'); ap.add_argument('--style-profile',default='',help='09-09:真人体裁句长档案 json;chosen 先限定在带内(msl 在 Q1-Q3、超长/短句占比≤Q3),带内再取最低照抄;带内没有则退回最低照抄'); ap.add_argument('--structure',action='store_true',help='09-11:chosen 取干净候选里 句序对齐度+照抄率 最低的(结构打散优先),不看任何检测器'); ap.add_argument('--identity-bad',action='store_true',help='GLM 判 identity(几乎照抄)的候选也当 rejected(结构信号,专治照抄模式)'); ap.add_argument('--verdicts',default='',help='显式指定判定缓存文件(默认按提示词哈希分版本:experiments/dpo/cand_verdicts_<tag>_<hash8>.jsonl;提示词一改旧缓存自动不复用)'); a=ap.parse_args()
P='''You are a strict reviewer. DRAFT is the original text; OUTPUT is a rewrite that must keep every fact. Output ONLY JSON:
{"facts_all_kept": true/false, "meaning_changed": true/false, "added_content": true/false, "padding": true/false, "greeting_signoff_kept": true/false, "format_kept": true/false, "identity": true/false, "reads_like_person": 1-5, "severity": "critical" | "minor" | "none", "evidence": "<=40 words naming the worst problem, or 'none'"}
severity (judge only what matters to a reader who relies on the text): "critical" = a fact reversed or contradicted; a number, name, date, amount, or deadline changed or dropped (re-expressing the same value is fine: 'ten weeks' -> '10 weeks', 'from 82% to 87%' -> 'up 5 percentage points', '1,200' -> '1200'); an invented fact, claim, quote, promise, or reason; a request, decision, action item, or who-did-what changed or lost; a completed action turned into an intention (or the reverse); steps of a procedure, a timeline, or a numbered sequence put in a different order; language switched; text truncated or garbled. "minor" = only hedging or emphasis shifted (suggests -> shows, mostly -> all, most trusted -> the one trusted), a boundary softened (at least 200 -> more than 200), synonyms with slightly different flavor, a connective added between two facts that DRAFT already states together. "none" = faithful. Changing voice (passive <-> active), reordering sentences or paragraphs whose order carries no meaning, merging or splitting sentences, and changing list/prose form are NOT problems when every fact and the document format survive. Subject line, title, greeting, sign-off, signature and contact lines are judged ONLY through format_kept / greeting_signoff_kept, never through severity.
facts_all_kept: every fact, number, name, date, claim and request of DRAFT appears in OUTPUT (rewording is fine). meaning_changed: OUTPUT states something DRAFT does not (a reversed condition, a claim made stronger/weaker, an opinion turned into a fact, a lost qualifier such as 'significant'). added_content: OUTPUT contains facts, examples, numbers or claims with new information not in DRAFT (pure connectives do not count). padding: OUTPUT contains more than one sentence that corresponds to nothing in DRAFT — generic framing, filler, commentary, restating what was just said — even if it adds no facts. greeting_signoff_kept: if DRAFT has a greeting/sign-off/signature/contact line, OUTPUT keeps them with the same names and details (true if DRAFT has none). format_kept: OUTPUT keeps DRAFT's document format — subject line and title present, numbered/bulleted lists stay lists, paragraph breaks are not collapsed into one block, section headers kept, no stray markup such as <p> (true if DRAFT has none of these); reordering is fine. identity: after ignoring punctuation, filler words and sentence order, OUTPUT reproduces DRAFT's wording nearly word for word (roughly 85% or more of DRAFT's phrases reappear unchanged). reads_like_person: 5 = a competent person clearly wrote it, 1 = obviously machine text. Same language as DRAFT is required; if OUTPUT switches language or is truncated/garbled, set facts_all_kept false.
DRAFT:
"""{d}"""
OUTPUT:
"""{o}"""'''
from textmetrics import copy as tm_copy, structure as tm_structure, all_metrics   # 09-11 审计:统一量尺,见 scripts/textmetrics.py(v2:两向照抄/覆盖率/事实召回/退化)
def copy_rate(d,o): return tm_copy(d,o)          # = max(输出侧 5-gram 照抄, 草稿侧被抄, 两向整句复现)
P_HASH=hashlib.md5(P.encode()).hexdigest()[:8]
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
VF=a.verdicts or f'experiments/dpo/cand_verdicts_{a.tag}_{P_HASH}.jsonl'; os.makedirs('experiments/dpo',exist_ok=True); print('判定缓存',VF,'提示词',P_HASH,flush=True)
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
    if not c.strip(): v={'facts_all_kept':False,'meaning_changed':True,'added_content':False,'padding':False,'greeting_signoff_kept':False,'format_kept':False,'identity':False,'reads_like_person':1,'severity':'critical','evidence':'empty'}   # 只拦空输出;长短/截断由 GLM 判
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
REQ=('facts_all_kept','meaning_changed','added_content','greeting_signoff_kept','format_kept','identity','reads_like_person')
def clean(v):
    if not v or any(v.get(k) is None for k in REQ): return False        # 09-11 审计 S6:缺任何必需字段 = 不够干净(旧缓存/残缺 JSON 不再默认放行)
    if a.tiers and v.get('severity') is None: return False
    ok=bool(v.get('facts_all_kept') and not v.get('meaning_changed') and not v.get('added_content') and not v.get('padding') and v.get('greeting_signoff_kept') and v.get('format_kept') and not v.get('identity') and (v.get('reads_like_person') or 0)>=4)
    ok=ok and not v.get('_degenerate') and v.get('_copy',0)<0.85 and v.get('_lcs',0)<0.9
    if a.tiers: ok=ok and v.get('severity')=='none'
    return ok
def bad(v):
    if not v: return False
    if v.get('_degenerate'): return True                                    # 输出非空却切不出句子(乱码/碎片/标签)
    if a.identity_bad and v.get('identity'): return True
    if a.identity_bad and (v.get('_copy',0)>=0.85 or v.get('_lcs',0)>=0.9): return True   # GLM 没标 identity 但两向照抄 ≥.85 / 内容 token 有序公共子序列 ≥.9(抄全文+插填充词)也算原样
    if a.tiers: return bool(v.get('severity')=='critical' or v.get('format_kept') is False)   # 09-11 用户要求:格式丢了算坏样本
    return bool(v.get('meaning_changed') or v.get('added_content') or not v.get('facts_all_kept'))
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
def align(d,o): return tm_structure(d,o)['align_sim']      # 每句找最像草稿句的相似度均值(逐句改写深度;不看顺序)
def struct_score(m):
    return m['align_sim']+m['copy']+0.5*m['one_to_one']+0.5*(1-m['coverage'])+1.0*(1-m['format_recall'])   # 越低=改得越深、拆并越自由、抄得越少;覆盖率项让"删句"不再是"改得深"(审计 S3/S4)
stats={'pairs':0,'no_clean':0,'no_bad':0,'all_clean':0,'style_band':0}; n=0
with open(a.out,'w') as f:
    for k,r in cands.items():
        if k not in inputs: continue
        d=inputs[k]['draft']
        MM={}; rows=[]
        for i,c in enumerate(r['cands']):
            if (k,i) not in done: continue
            m=all_metrics(d,c); MM[i]=m; rows.append((i,c,dict(done.get((k,i)) or {}, _copy=m['copy'], _lcs=m['lcs_content'], _degenerate=m['degenerate'])))
        good=[(i,c,v) for i,c,v in rows if clean(v)]; worse=[(i,c,v) for i,c,v in rows if bad(v)]
        if not good: stats['no_clean']+=1; continue
        if not worse: stats['no_bad']+=1; stats['all_clean']+=1; continue
        band=[g for g in good if in_band(r['src'],g[1])] if PROF else []
        if band: stats['style_band']+=1
        pool=band or good; fr_max=max(MM[x[0]]['fact_recall'] for x in pool); pool=[x for x in pool if MM[x[0]]['fact_recall']>=fr_max-0.05]   # 审计 S4:先守住硬事实召回,再比结构
        ci,ct,cv=min(pool,key=(lambda x:struct_score(MM[x[0]])) if a.structure else (lambda x:MM[x[0]]['copy']))          # 干净里改得最狠的(--structure:逐句相似+照抄+拆并+覆盖)
        ri,rt,rv=max(worse,key=lambda x:(x[2].get('severity')=='critical',bool(x[2].get('identity')),x[2].get('meaning_changed',False),not x[2].get('facts_all_kept',True),x[2].get('added_content',False)))
        f.write(json.dumps({'prompt':build_fn(d),'chosen':ct,'rejected':rt,'key':k,'src':r['src'],'chosen_copy':MM[ci]['copy'],'chosen_align':MM[ci]['align_sim'],'chosen_1to1':MM[ci]['one_to_one'],'chosen_cov':MM[ci]['coverage'],'chosen_facts':MM[ci]['fact_recall'],'chosen_fmt':MM[ci]['format_recall'],'rejected_reason':rv.get('evidence','')[:120],'provider':f'dpo:{a.tag}:glm-5.3:{P_HASH}'},ensure_ascii=False)+'\n'); stats['pairs']+=1
print('偏好对',stats,'→',a.out)
