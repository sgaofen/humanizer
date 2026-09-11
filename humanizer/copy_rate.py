#!/usr/bin/env python3
"""量【改写时照抄了多少原文措辞】—— 改写任务真正的病灶指标。

2026-08-24 用户纠正了我的方向:改写的本义是【保持论述顺序、换掉行文】,
把信息顺序打乱那不叫改写叫重写。所以 order_div 不该当把关线。
真正该量的是"照抄率":模型有没有把原文的句子原样搬过来 ——
记忆里的实测早有线索:"它根本没执行改写,只是搬句子,整句照抄"。

  copy_5gram  输出里有多少 5 元词组是从输入原样搬来的(0~1,越高=抄得越多)
  verbatim_sent 整句照抄的比例(归一化编辑距离 <0.1 视为照抄)
  novel_ratio  1 - copy_5gram,即"自己写的比例"
"""
import re
from difflib import SequenceMatcher

RE_WORD = re.compile(r"[A-Za-z0-9']+")
RE_SENT = re.compile(r'(?<=[.!?])\s+')

def _w(t): return RE_WORD.findall((t or '').lower())

def _ngrams(ws, n=5):
    return set(tuple(ws[i:i+n]) for i in range(len(ws)-n+1))

def _sents(t):
    return [s.strip() for s in RE_SENT.split(t or '') if len(s.split()) >= 5]

def copy_rate(src, out, n=5):
    ws_s, ws_o = _w(src), _w(out)
    gs, go = _ngrams(ws_s, n), _ngrams(ws_o, n)
    c5 = len(gs & go) / len(go) if go else 0.0
    ss, so = _sents(src), _sents(out)
    verb = 0
    for s in so:
        if any(SequenceMatcher(None, s.lower(), x.lower()).ratio() > 0.9 for x in ss):
            verb += 1
    return {'copy_5gram': round(c5, 3),
            'verbatim_sent': round(verb / len(so), 3) if so else 0.0,
            'novel_ratio': round(1 - c5, 3),
            'n_sent_out': len(so)}

if __name__ == '__main__':
    a = "The samples were collected in June. We measured density at three sites. No difference was found."
    b_copy = "The samples were collected in June. We measured density at three sites. Nothing differed."
    b_new  = "Density came from three sites, sampled that June. Nothing separated the groups."
    print('照抄版:', copy_rate(a, b_copy))
    print('真改写:', copy_rate(a, b_new))
