#!/usr/bin/env python3
"""英文评测的事实判定(不碰 Originality):GLM-5.3 逐条比较 DRAFT 与 OUTPUT。
写 eval/fidelity_<tag>.json(不改 cases 文件)。用法:python3 scripts/glm_eval_en.py --tag E18c --cases eval/ood_cases_claude [--workers 6]"""
import argparse,json,os,re,sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from llm_http import zai
ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True); ap.add_argument('--cases',required=True); ap.add_argument('--workers',type=int,default=6); a=ap.parse_args()
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P='''You are a strict reviewer. DRAFT is the original text; OUTPUT is a rewrite that must keep every fact. Output ONLY JSON:
{"facts_all_kept": true/false, "meaning_changed": true/false, "added_content": true/false, "greeting_signoff_kept": true/false, "identity": true/false, "reads_like_person": 1-5, "severity": "critical" | "minor" | "none", "evidence": "<=40 words naming the worst problem, or 'none'"}
severity (judge only what matters to a reader who relies on the text): "critical" = a fact reversed or contradicted; a number, name, date, amount, or deadline changed or dropped; an invented fact, claim, quote, promise, or reason; a request, decision, action item, or who-did-what changed or lost; a completed action turned into an intention (or the reverse); language switched; text truncated. "minor" = only hedging or emphasis shifted (suggests -> shows, mostly -> all, most trusted -> the one trusted), a boundary softened (at least 200 -> more than 200), synonyms with slightly different flavor, added connectives or framing sentences that carry no new facts, reordering. "none" = faithful. Dropping or changing a subject line, greeting, sign-off, signature or contact line is NOT critical and NOT minor for severity purposes (report it only through greeting_signoff_kept).
facts_all_kept: every fact, number, name, date, claim and request of DRAFT appears in OUTPUT (rewording is fine). meaning_changed: OUTPUT states something DRAFT does not (a reversed condition, a claim made stronger/weaker, an opinion turned into a fact, a lost statistical qualifier such as 'significant'). added_content: OUTPUT contains facts, examples, numbers or sentences with new information not in DRAFT (pure connectives do not count). greeting_signoff_kept: if DRAFT has a greeting/sign-off, OUTPUT keeps them. identity: OUTPUT is essentially a copy of DRAFT. reads_like_person: 5 = a competent person clearly wrote it, 1 = obviously machine text.
DRAFT:
"""{d}"""
OUTPUT:
"""{o}"""'''
d=json.load(open(f'{ROOT}/eval/cases_{a.tag}.json'))
en=[c for c in d if not (c.startswith(('g_','h_','j_')) or '_zh' in c)]
jobs=[]
for c in en:
    draft=open(f'{ROOT}/{a.cases}/{c}.txt').read().strip()
    for i,r in enumerate(d[c]): jobs.append((c,i,draft,r['text']))
def judge(j):
    c,i,draft,out=j
    t=zai(P.replace('{d}',draft).replace('{o}',out),model='glm-5.3',max_tokens=400,timeout=240) or ''
    m=re.search(r'\{.*\}',t,re.S)
    try: v=json.loads(m.group(0))
    except Exception: v=None
    return dict(case=c,i=i,verdict=v)
with ThreadPoolExecutor(a.workers) as ex: res=list(ex.map(judge,jobs))
ok=[r for r in res if r['verdict']]
fk=sum(1 for r in ok if r['verdict'].get('facts_all_kept')); mc=sum(1 for r in ok if r['verdict'].get('meaning_changed')); ad=sum(1 for r in ok if r['verdict'].get('added_content'))
gs=sum(1 for r in ok if r['verdict'].get('greeting_signoff_kept') is False); idn=sum(1 for r in ok if r['verdict'].get('identity'))
ppl=[r['verdict'].get('reads_like_person') or 0 for r in ok]
clean=sum(1 for r in ok if r['verdict'].get('facts_all_kept') and not r['verdict'].get('meaning_changed') and not r['verdict'].get('added_content'))
json.dump(res,open(f'{ROOT}/eval/fidelity_{a.tag}.json','w'),ensure_ascii=False,indent=1)
crit=sum(1 for r in ok if r['verdict'].get('severity')=='critical'); mino=sum(1 for r in ok if r['verdict'].get('severity')=='minor')
print(f'{a.tag} 严重错 {crit} | 轻微 {mino} | 无 {len(ok)-crit-mino}')
print(f'{a.tag} 英文 {len(ok)}/{len(res)} 判成功 | 事实全保 {fk} | 意思被改 {mc} | 添加内容 {ad} | 丢问候落款 {gs} | 恒等 {idn} | 三项全干净 {clean} | 像人均分 {sum(ppl)/max(1,len(ppl)):.2f}')
for r in ok:
    v=r['verdict']
    if v.get('meaning_changed') or not v.get('facts_all_kept') or v.get('added_content'): print(f"  {r['case']}#{r['i']}: {v.get('evidence','')[:110]}")
