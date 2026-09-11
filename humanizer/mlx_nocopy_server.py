#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gemma 臂的本地推理服务(2026-09-10):和 `python -m mlx_lm server` 一样吃 /v1/completions,
多两个可选字段,用来做解码期反照抄(集群上 hpc3/gen_cases.py 的 NoCopyProcessor 原样移植):

    copy_penalty  >0 才开;凡是会把「已生成的最后 n-1 个 token + 候选 token」凑成草稿里出现过的
                  n-gram 的候选 token,logit 减 penalty。含数字的 n-gram 豁免(数字必须原样)。
    copy_n        n-gram 长度,默认 5。
    draft         草稿原文;不传就用整个 prompt 当"不许照抄"的来源。

自适应(第一发照抄 >0.35 才带惩罚重采)放在客户端 gen4b.py 里,服务端只管按参数生成。
不是检测器,不看输出分数,只限制逐字复制的长度。

用法: python scripts/mlx_nocopy_server.py --model models/m_gemma --port 8104
"""
import argparse, json, sys, time, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--port', type=int, default=8104)
ap.add_argument('--host', default='127.0.0.1')
a = ap.parse_args()

MODEL, TOK = load(a.model)
LOCK = threading.Lock()   # 一次只跑一个请求,和 mlx_lm server 行为一致
EOS = set(TOK.eos_token_ids) if hasattr(TOK, 'eos_token_ids') and TOK.eos_token_ids else {TOK.eos_token_id}


class NoCopy:
    def __init__(self, src_ids, n=5, penalty=2.0):
        self.n = n; self.p = float(penalty)
        digit = {i for i in set(src_ids) if any(c.isdigit() for c in TOK.decode([i]))}
        self.prefix = {}
        for i in range(len(src_ids) - n + 1):
            g = tuple(src_ids[i:i + n])
            if not (set(g) & digit):
                self.prefix.setdefault(g[:-1], set()).add(g[-1])
        self.plen = None   # 由第一次调用时的 tokens 长度确定(prompt 长度)

    def __call__(self, tokens, logits):
        # mlx_lm 约定:tokens = 到目前为止的全部 token(prompt + 已生成),logits 形状 (1, vocab)
        if self.plen is None:
            self.plen = tokens.shape[0]
        gen = tokens[self.plen:].tolist() if tokens.shape[0] > self.plen else []
        if len(gen) < self.n - 1:
            return logits
        nxt = self.prefix.get(tuple(gen[-(self.n - 1):]))
        if not nxt:
            return logits
        idx = mx.array(sorted(nxt))
        return logits.at[:, idx].add(-self.p) if hasattr(logits, 'at') else _sub(logits, idx, self.p)


def _sub(logits, idx, p):
    mask = mx.zeros(logits.shape[-1], dtype=logits.dtype)
    mask[idx] = -p
    return logits + mask


def complete(prompt, max_tokens, temperature, top_p, stop, copy_penalty, copy_n, draft):
    ids = TOK.encode(prompt)
    procs = []
    if copy_penalty and copy_penalty > 0:
        src = TOK.encode(draft) if draft else ids
        procs.append(NoCopy(src, n=int(copy_n or 5), penalty=copy_penalty))
    sampler = make_sampler(temp=temperature, top_p=top_p)
    out_ids = []
    text = ''
    for tok, _ in generate_step(mx.array(ids), MODEL, max_tokens=max_tokens, sampler=sampler,
                                logits_processors=procs or None):
        t = int(tok)
        if t in EOS:
            break
        out_ids.append(t)
        if stop and len(out_ids) % 8 == 0:
            text = TOK.decode(out_ids)
            if any(s in text for s in stop):
                break
    text = TOK.decode(out_ids)
    for s in (stop or []):
        k = text.find(s)
        if k >= 0:
            text = text[:k]
    return text, len(ids), len(out_ids)


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):   # 安静
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith('/v1/models') or self.path == '/health':
            return self._json(200, {'object': 'list', 'data': [{'id': a.model}], 'nocopy': True})
        self._json(404, {'error': 'not found'})

    def do_POST(self):
        if not self.path.startswith('/v1/completions'):
            return self._json(404, {'error': 'only /v1/completions'})
        n = int(self.headers.get('Content-Length', 0)); body = json.loads(self.rfile.read(n) or b'{}')
        prompt = body.get('prompt', '')
        if isinstance(prompt, list): prompt = prompt[0]
        t0 = time.time()
        with LOCK:
            text, np_, ng = complete(prompt, int(body.get('max_tokens', 700)), float(body.get('temperature', 0.85)),
                                     float(body.get('top_p', 0.95)), body.get('stop') or [],
                                     float(body.get('copy_penalty', 0) or 0), body.get('copy_n', 5), body.get('draft'))
        self._json(200, {'id': f'cmpl-{int(t0)}', 'object': 'text_completion', 'model': a.model,
                         'choices': [{'text': text, 'index': 0, 'finish_reason': 'stop'}],
                         'usage': {'prompt_tokens': np_, 'completion_tokens': ng, 'total_tokens': np_ + ng},
                         'nocopy': bool(body.get('copy_penalty')), 'seconds': round(time.time() - t0, 1)})


print(f'[mlx_nocopy_server] {a.model} @ {a.host}:{a.port}', flush=True)
HTTPServer((a.host, a.port), H).serve_forever()   # 单线程:MLX 数组不能跨线程(Stream(gpu) 错误),和 mlx_lm server 一样串行
