#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测判官的两票一致合并(2026-09-12)。

起因:09-12 的审计发现「判两次取一致」只进了 RL 奖励(scripts/rl_reward.py:82),评测侧
glm_eval_en/zh 一直是单发单判。仓库里实测同一批输出连判两次只有 61–77% 一致、严重错数摆动 ±1–3,
而版本对比表里的臂间差正好也是 1–3 —— 也就是说那张表的差值和它自己的噪声同量级。

合并口径与 rl_reward.judge 完全一致:
  - 布尔「还在不在」类(facts_all_kept / greeting_signoff_kept / format_kept):两次都说丢了才算丢
  - 布尔「出问题了」类(meaning_changed / added_content / identity):两次都报才算
  - severity:取较轻的一档
"""
RANK = {'none': 0, 'minor': 1, 'critical': 2}
KEPT = ('facts_all_kept', 'greeting_signoff_kept', 'format_kept')
BAD = ('meaning_changed', 'added_content', 'identity')


def merge(a, b):
    """两份判定合并。任一为 None 时退回单判并打标记(_votes=1),便于事后把这些条目挑出来。"""
    if a is None or b is None:
        v = dict(a or b) if (a or b) else None
        if v is not None: v['_votes'] = 1
        return v
    m = dict(a)
    m['severity'] = min(a.get('severity', 'none'), b.get('severity', 'none'), key=lambda s: RANK.get(str(s).lower(), 0))
    for k in KEPT:
        m[k] = not (a.get(k) is False and b.get(k) is False)
    for k in BAD:
        m[k] = bool(a.get(k)) and bool(b.get(k))
    ev = [x.get('evidence') for x in (a, b) if x.get('evidence') and str(x.get('evidence')).lower() != 'none']
    m['evidence'] = ev[0] if ev else 'none'
    m['_votes'] = 2
    m['_raw'] = [a, b]
    return m
