#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""薄包装(2026-09-11 审计后):所有照抄/结构量尺以 textmetrics.py 为唯一事实源。
接口保持 copy_rate(src, out) -> dict;注意 'copy_5gram' 现在 = max(5-gram 照抄, 整句照搬比例),中文按 8 字 gram,不再对中文全盲。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textmetrics import all_metrics

def copy_rate(src, out, n=5):
    m = all_metrics(src or '', out or '')
    return {'copy_5gram': m['copy'], 'copy_5gram_raw': m['copy_5gram'], 'verbatim_sent': m['verbatim_sent'], 'novel_ratio': round(1 - m['copy'], 3),
            'n_sent_out': m['n_out'], 'align_sim': m['align_sim'], 'order_agree': m['order_agree'], 'one_to_one': m['one_to_one']}
