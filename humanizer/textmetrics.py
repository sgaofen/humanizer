#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""textmetrics.py — 改写任务的硬量尺,唯一事实源(2026-09-11 审计后重写)。

以后配对(build_prefs)、生成端守卫(gen_cases / local_eval / 发布推理)、RL 奖励的硬约束都只从这里 import。
审计发现的四个缺陷及对策:
  1. 整句照搬但打乱顺序,旧 5-gram 集合照抄率只有 ~0.5(跨句 5-gram 散了)→ 加 verbatim_sent(模糊整句匹配)并取两者的 max 作 copy。
  2. 分句把 "Dr." "Oct." "$3.5" 当句号 → 缩写/小数/序号感知的分句器;中文按 。!?;换行 切。
  3. 位置对齐分不清"重排"与"拆并" → 改为"每句找最像的草稿句",分别给出 align_sim(逐句改写深度)、order_agree(顺序保持)、one_to_one(拆并程度)。
  4. 中文:ASCII 正则对中文全盲 → 中文用字 n-gram(词级 5-gram 对应 8 字),内容 token 用字二元组。

所有指标 0~1,越高越"像草稿"。没有任何检测器分数参与。
"""
import re
from difflib import SequenceMatcher

_ABBR = {'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 'st', 'vs', 'etc', 'e.g', 'i.e', 'fig', 'no', 'vol', 'inc', 'ltd', 'co', 'corp',
         'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec', 'u.s', 'u.k', 'ph.d', 'approx', 'dept', 'univ', 'al'}
_STOP = set('''a an the and or but if then than so of to in on at by for from with without into onto over under about as is are was were be been being am
do does did done have has had having i you he she it we they me him her us them my your his its our their this that these those there here who whom which what when where why how
not no nor very just also too only own same such can could will would shall should may might must need dare up down out off again further once more most all any both each few
because while until although though whether either neither'''.split())
_CJK = re.compile(r'[一-鿿]')


def is_cjk(t):
    """整体是否中文文本:前 400 字里中文字符多于英文词。"""
    head = t[:400]; return len(_CJK.findall(head)) > len(re.findall(r'[A-Za-z]{2,}', head))


def sentences(t):
    """缩写/小数/序号感知的分句;段落换行也是边界。中文按 。!?;! ? 切。返回去掉首尾空白的句子。"""
    t = (t or '').strip()
    if not t:
        return []
    out = []
    for para in re.split(r'\n\s*\n|\n', t):
        para = para.strip()
        if not para:
            continue
        if is_cjk(para):
            parts = re.split(r'(?<=[。！？；!?;])', para)
        else:
            # 在 [.!?]+空白 处切;不切的情况:缩写(Dr. Oct. e.g.)、单字母首字母(J. Smith)、句首列表序号("1. " "a. ")、小数点
            parts = []; buf = ''
            for tok in re.split(r'(?<=[.!?])(\s+)', para):
                if tok.strip() == '':
                    prev = buf.rstrip()
                    last = prev.split()[-1] if prev.split() else ''
                    core = last.rstrip('.!?').lower()
                    is_abbr = prev.endswith('.') and (core in _ABBR or re.fullmatch(r'[a-z]', core) or core.endswith('.') and len(core) <= 4)
                    is_marker = prev.endswith('.') and re.fullmatch(r'[0-9]{1,2}|[a-z]|[ivx]{1,4}', core) and len(prev.split()) == 1
                    if is_abbr or is_marker:
                        buf += tok; continue
                    parts.append(buf); buf = ''
                else:
                    buf += tok
            if buf.strip():
                parts.append(buf)
        out += [p.strip() for p in parts if p.strip()]
    return [s for s in out if (len(_CJK.findall(s)) >= 4 if _CJK.search(s) else len(s.split()) >= 3)]


def tokens(t):
    """内容 token:英文小写词(去停用词,保留数字);中文用字二元组(数字/字母串整体保留)。"""
    if is_cjk(t):
        s = re.sub(r'\s+', '', t)
        toks = re.findall(r'[A-Za-z0-9]+', s)
        chars = _CJK.findall(s)
        return toks + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return [w for w in re.findall(r"[a-z0-9']+", t.lower()) if w not in _STOP]


def _grams(t, n=5):
    if is_cjk(t):
        s = re.sub(r'\s+', '', t); k = 8
        return [s[i:i + k] for i in range(len(s) - k + 1)]
    w = re.findall(r"[a-z0-9']+", t.lower())
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def copy_5gram(draft, out):
    """输出的 5-gram(中文 8 字)里有多少在草稿里出现过;按 token 计数而非集合,重复照搬也算。"""
    g = _grams(out)
    if not g:
        return 0.0
    dg = set(_grams(draft))
    return sum(1 for x in g if x in dg) / len(g)


def verbatim_sent(draft, out, thr=0.9):
    """输出里有多大比例的句子和草稿某句几乎一字不差(SequenceMatcher ratio > thr)。抓"整句照搬只换顺序"。"""
    ds = sentences(draft); os_ = sentences(out)
    if not os_ or not ds:
        return 0.0
    hit = 0
    for s in os_:
        sl = s.lower()
        if any(SequenceMatcher(None, sl, x.lower()).ratio() > thr for x in ds):
            hit += 1
    return hit / len(os_)


def copy(draft, out):
    """综合照抄率 = max(5-gram 照抄, 整句照搬比例)。任一路径抄了都算。"""
    return max(copy_5gram(draft, out), verbatim_sent(draft, out))


def _jac(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))


def structure(draft, out):
    """结构指标。每个输出句在草稿里找最像的句子(内容 token Jaccard):
       align_sim  最像句相似度均值(0=每句都和草稿任何一句不像,1=逐句照搬/紧贴改写)
       order_agree 相邻输出句所匹配的草稿句序号是否单调不减的比例(1=顺序完全保持,0=完全打乱)
       one_to_one 草稿句里被恰好一个输出句匹配的比例(1=一句对一句;拆并越多越低)
       n_out/n_draft 句数"""
    ds = sentences(draft); os_ = sentences(out)
    if not ds or not os_:
        return {'align_sim': 0.0, 'order_agree': 0.0, 'one_to_one': 0.0, 'n_out': len(os_), 'n_draft': len(ds)}
    dt = [tokens(s) for s in ds]
    idx, sims = [], []
    for s in os_:
        ot = tokens(s)
        best = max(range(len(ds)), key=lambda j: _jac(ot, dt[j]))
        sims.append(_jac(ot, dt[best])); idx.append(best if _jac(ot, dt[best]) > 0 else -1)
    pairs = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1) if idx[i] >= 0 and idx[i + 1] >= 0]
    order = sum(1 for a, b in pairs if b >= a) / len(pairs) if pairs else 1.0
    counts = {}
    for j in idx:
        if j >= 0: counts[j] = counts.get(j, 0) + 1
    one = sum(1 for j in range(len(ds)) if counts.get(j, 0) == 1) / len(ds)
    return {'align_sim': round(sum(sims) / len(sims), 3), 'order_agree': round(order, 3), 'one_to_one': round(one, 3),
            'n_out': len(os_), 'n_draft': len(ds)}


def all_metrics(draft, out):
    m = structure(draft, out)
    m.update({'copy_5gram': round(copy_5gram(draft, out), 3), 'verbatim_sent': round(verbatim_sent(draft, out), 3)})
    m['copy'] = round(max(m['copy_5gram'], m['verbatim_sent']), 3)
    return m


if __name__ == '__main__':
    D = "Dr. Lin approved the request on Oct. 14. We measured density at three sites in June. No difference was found between groups. The budget is $3.5 million for 2019-2021. Please send the report by Friday."
    cases = {
        '原样照抄': D,
        '整句打乱顺序(零改写)': "Please send the report by Friday. The budget is $3.5 million for 2019-2021. No difference was found between groups. We measured density at three sites in June. Dr. Lin approved the request on Oct. 14.",
        '顺序不变·逐句换词': "Dr. Lin signed off on the request on Oct. 14. In June we sampled density at three locations. The groups did not differ. For 2019-2021 the budget comes to $3.5 million. Get the report to me by Friday.",
        '顺序不变·拆并句子': "Dr. Lin approved the request on Oct. 14, and in June we measured density at three sites. No difference was found between groups. The budget is $3.5 million for 2019-2021, so please send the report by Friday.",
        '重排+换词(真重组)': "Get the report to me by Friday. For 2019-2021 the budget comes to $3.5 million. In June we sampled density at three locations and the groups did not differ. Dr. Lin signed off on the request on Oct. 14.",
        '只保留数字的胡写': "Random words here about 2019-2021 and $3.5 million and Oct. 14 with three sites and Friday but nothing else matters at all today.",
    }
    print('草稿分句:', sentences(D))
    print(f"{'情形':22s} copy  5gram verb  align order 1to1  句数")
    for k, o in cases.items():
        m = all_metrics(D, o); print(f"{k:22s} {m['copy']:.2f}  {m['copy_5gram']:.2f}  {m['verbatim_sent']:.2f}  {m['align_sim']:.2f}  {m['order_agree']:.2f}  {m['one_to_one']:.2f}  {m['n_out']}/{m['n_draft']}")
    Z = "李老师您好,关于上周说的数据访问申请,还有一点补充说明。首先,这次分析仅需2019-2021年时间序列子集。其次,审批大概两到三周。祝好,小王"
    Z2 = "祝好,小王。其次,审批大概两到三周。首先,这次分析仅需2019-2021年时间序列子集。关于上周说的数据访问申请,还有一点补充说明。李老师您好。"
    Z3 = "李老师好,上周提到的数据访问申请我再补充一下:分析只用2019到2021年那段时间序列,审批大概要两三周。谢谢您,小王"
    for k, o in (('中文原样', Z), ('中文整句打乱', Z2), ('中文真改写', Z3)):
        m = all_metrics(Z, o); print(f"{k:22s} {m['copy']:.2f}  {m['copy_5gram']:.2f}  {m['verbatim_sent']:.2f}  {m['align_sim']:.2f}  {m['order_agree']:.2f}  {m['one_to_one']:.2f}  {m['n_out']}/{m['n_draft']}")
