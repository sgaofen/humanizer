#!/usr/bin/env python3
"""中文评测判定(审计 G0):Originality 不打中文分,用 GLM-5.3 判每个中文 case 的输出。
读 eval/cases_<tag>.json 里名字含 _zh 或 g_/h_/j_ 前缀的 case,对 draft(eval/<cases_dir>/<case>.txt)与每个输出问:
事实全保?有无添加?问候落款保留?是否恒等?像人 1–5。通过 = 事实全保 ∧ 无添加 ∧ 问候落款保 ∧ 非恒等 ∧ ≥4。结果写回 cases 文件的 'zh' 字段并打印 x/N。
用法:python3 scripts/glm_eval_zh.py --tag E17c --cases eval/ood_cases_claude
"""
import argparse,json,os,re,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from llm_http import zai
ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True); ap.add_argument('--cases',required=True); a=ap.parse_args()
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P=("你是评审。DRAFT 是 AI 写的原稿,OUTPUT 是改写模型的输出。只输出 JSON:{\"facts_all_kept\":true/false,\"added_content\":true/false,\"greeting_signoff_kept\":true/false,\"identity\":true/false,\"reads_like_person\":1-5,\"evidence\":\"≤120字,引用证据\"}。"
   "facts_all_kept:DRAFT 的每个事实、数字、人名、要点都在 OUTPUT 里;added_content:OUTPUT 有 DRAFT 没有的事实、说法或段落;greeting_signoff_kept:DRAFT 的问候与落款(若有)在 OUTPUT 中保留;identity:OUTPUT 与 DRAFT 几乎逐字相同;reads_like_person:像真人自然写出来的程度。\n\nDRAFT:\n{d}\n\nOUTPUT:\n{o}")
d=json.load(open(f'{ROOT}/eval/cases_{a.tag}.json'))
zh=[c for c in d if c.startswith(('g_','h_','j_')) or '_zh' in c]
passed=0; total=0
for c in zh:
    draft=open(f'{ROOT}/{a.cases}/{c}.txt').read().strip()
    for r in d[c]:
        t=zai(P.replace('{d}',draft).replace('{o}',r['text']),model='glm-5.3',max_tokens=400,timeout=240) or ''
        m=re.search(r'\{.*\}',t,re.S)
        try: j=json.loads(m.group(0))
        except Exception: j=None
        r['zh']=j; total+=1
        ok=bool(j and j.get('facts_all_kept') and not j.get('added_content') and j.get('greeting_signoff_kept') and not j.get('identity') and (j.get('reads_like_person') or 0)>=4)
        r['zh_pass']=ok; passed+=ok
        print(c,'通过' if ok else '未过',j and j.get('evidence','')[:100])
json.dump(d,open(f'{ROOT}/eval/cases_{a.tag}.json','w'),ensure_ascii=False,indent=1)
print(f'{a.tag} 中文 {passed}/{total} 通过(GLM 判)')
