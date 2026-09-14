#!/usr/bin/env python3
"""中文评测判定(审计 G0):Originality 不打中文分,用 GLM-5.3 判每个中文 case 的输出。
读 eval/cases_<tag>.json 里名字含 _zh 或 g_/h_/j_ 前缀的 case,对 draft(eval/<cases_dir>/<case>.txt)与每个输出问:
事实全保?有无添加?问候落款保留?是否恒等?像人 1–5。通过 = 事实全保 ∧ 无添加 ∧ 问候落款保 ∧ 非恒等 ∧ ≥4。结果写回 cases 文件的 'zh' 字段并打印 x/N。
用法:python3 scripts/glm_eval_zh.py --tag E17c --cases eval/ood_cases_claude
"""
import argparse,json,os,re,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from llm_http import zai, luna
from judge_consensus import merge
ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True); ap.add_argument('--cases',required=True); ap.add_argument('--votes',type=int,default=2); ap.add_argument('--second',default='luna',choices=['luna','glm','none'],help='第二票用哪个模型。默认 luna(gpt-5.6-luna):同一个模型判两遍共享同样的系统偏差,跨模型才是真的两票;而且 RL 在线时不抢 GLM 配额')
ap.add_argument('--limit',type=int,default=0,help='只判前 N 条,冒烟用')
a=ap.parse_args()
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P=("你是评审。DRAFT 是 AI 写的原稿,OUTPUT 是改写模型的输出。只输出 JSON:{\"facts_all_kept\":true/false,\"added_content\":true/false,\"greeting_signoff_kept\":true/false,\"format_kept\":true/false,\"identity\":true/false,\"reads_like_person\":1-5,\"evidence\":\"≤120字,引用证据\"}。"
   "facts_all_kept:DRAFT 的每个事实、数字、人名、要点都在 OUTPUT 里;added_content:OUTPUT 有 DRAFT 没有的事实、说法或段落;greeting_signoff_kept:DRAFT 的问候与落款(若有)在 OUTPUT 中保留;format_kept:DRAFT 的文档格式在 OUTPUT 中保留 —— 主题行/标题还在、条列仍是条列、段落没有被压成一整块、小标题还在(DRAFT 没有这些则为 true);重排顺序可以,主题行或落款换了措辞但要素还在也算保留;identity:OUTPUT 与 DRAFT 几乎逐字相同;【以下一律不算错,不要报】:同义替换、礼貌或情绪强度的变化、句式重组拆并重排、口语或书面语域的选择、被动改主动、数值的等价表达;只有当改动影响了【什么是真的、有多确定、有多少】时才算错;reads_like_person:像真人自然写出来的程度。\n\nDRAFT:\n{d}\n\nOUTPUT:\n{o}")
d=json.load(open(f'{ROOT}/eval/cases_{a.tag}.json'))
zh=[c for c in d if c.startswith(('g_','h_','j_')) or '_zh' in c]
passed=0; total=0; res=[]
def _once(draft,out,second=False):
    q=P.replace('{d}',draft).replace('{o}',out)
    t=(luna(q) if (second and a.second=='luna') else zai(q,model='glm-5.3',max_tokens=400,timeout=240)) or ''
    m=re.search(r'\{.*\}',t,re.S)
    try:
        v=json.loads(m.group(0))
        if second and a.second=='luna': v['_by']='luna'
        return v
    except Exception: return None
for c in zh:
    draft=open(f'{ROOT}/{a.cases}/{c}.txt').read().strip()
    for i,r in enumerate(d[c]):
        j=_once(draft,r['text'])
        if a.votes>1 and a.second!='none': j=merge(j,_once(draft,r['text'],second=True))
        r['zh']=j; total+=1; res.append({'case':c,'i':i,'verdict':j})
        # 09-12:通过判据里原本含 reads_like_person>=4,那是 AI 自评"像不像人写",不作数,已删;
        # 判定失败(j=None)不再算作"未通过",单独计数,否则分母把判官自己的故障算到模型头上
        ok=bool(j and j.get('facts_all_kept') and not j.get('added_content') and j.get('greeting_signoff_kept') and j.get('format_kept', True) and not j.get('identity'))
        r['zh_pass']=ok; passed+=ok
        print(c,'通过' if ok else '未过',j and j.get('evidence','')[:100])
json.dump(d,open(f'{ROOT}/eval/cases_{a.tag}.json','w'),ensure_ascii=False,indent=1)
json.dump(res,open(f'{ROOT}/eval/fidelity_zh_{a.tag}.json','w'),ensure_ascii=False,indent=1)   # 09-11:中文判定明细以前只写回 cases,事后没法分析,单独落盘
import collections as _c
_f=sum(1 for x in res if x['verdict'] and x['verdict'].get('format_kept') is False)
_j=sum(1 for x in res if not x['verdict'])
_n=total-_j
print(f'{a.tag} 中文 {passed}/{_n} 通过(GLM 判,分母已剔除判定失败)| 丢格式 {_f} | 判定失败 {_j} | 单票判定 {sum(1 for x in res if x["verdict"] and x["verdict"].get("_votes")==1)}')
