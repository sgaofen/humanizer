#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直连 HTTP 调模型 —— 替掉 CLI 子进程,快 4-5 倍,支持高并发。

为什么换(2026-08-28 实测):
  走 pi/agy 的 CLI 子进程,每次都要起进程;agy 还会带 13,736 token 的系统上下文,
  单次 7-9 秒。直连 coding 端点是 1.7 秒。
关键:key 是 coding 套餐的,必须走 /api/coding/paas/v4,
  打 /api/paas/v4 一律 429(和并发无关,空载也 429)。踩过。
"""
import json, os, time, urllib.request, urllib.error, threading, collections

ZAI_KEY = None
_lock = threading.Lock()
ZAI_URL = 'https://api.z.ai/api/coding/paas/v4/chat/completions'


def _key():
    global ZAI_KEY
    if ZAI_KEY is None:
        with _lock:
            ZAI_KEY = open(os.path.expanduser('~/.config/zai_key')).read().strip()
    return ZAI_KEY


FAILS = collections.Counter()          # 失败原因计数(调并发用,不影响返回值)


def zai(prompt, model='glm-5.2', max_tokens=4096, timeout=180, retries=3):
    """返回文本;失败返回 None。thinking 关掉,否则 token 全被思考吃掉。

    2026-08-29:并发 14 时约 21% 返回空,而单发一律正常 —— 空是并发打出来的,
    不是内容问题。原来这里失败就无声吞掉,根本看不出是 429 还是空内容,
    所以加了 FAILS 计数,好照着实测选并发数。"""
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0.8,
        'thinking': {'type': 'disabled'},
    }).encode()
    for attempt in range(retries):
        req = urllib.request.Request(ZAI_URL, body, {
            'Authorization': f'Bearer {_key()}', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            m = d['choices'][0]['message']
            t = (m.get('content') or '').strip()
            if t:
                return t
            FAILS[f'空内容#{attempt}'] += 1
            time.sleep(1 + attempt * 2)
        except urllib.error.HTTPError as e:
            FAILS[f'HTTP{e.code}'] += 1
            if e.code in (429, 500, 502, 503):
                time.sleep(2 + attempt * 4); continue
            return None
        except Exception as ex:
            FAILS[type(ex).__name__] += 1
            time.sleep(1 + attempt * 2)
    FAILS['最终失败'] += 1
    return None


if __name__ == '__main__':
    import sys
    t0 = time.time()
    print(repr(zai(sys.argv[1] if len(sys.argv) > 1 else 'Reply with exactly: alpha beta gamma',
                   max_tokens=200)), f'({time.time()-t0:.1f}s)')


# ── Luna(gpt-5.6-luna,走 codex CLI)────────────────────────────────────────────
# 2026-09-12:判定的第二票改用另一个模型。原来 votes=2 是 GLM 自己判两遍,同一个模型的两次采样
# 共享同样的系统偏差 —— 它系统性看错的地方,判几次都一样看错。换成跨模型才是真的两票。
# 附带好处:RL 在线训练期间集群在狂打 GLM(R1 曾因此判定失败率 30%),评测走 Luna 就不抢配额。
# 代价:codex 是子进程,约 7 秒一发(GLM 直连 1.7 秒),所以只用在评测侧,不进 RL 奖励
#(集群节点上也没有 codex 与它的登录态)。
import subprocess, tempfile

LUNA_FAILS = collections.Counter()


def luna(prompt, model='gpt-5.6-luna', timeout=300, retries=2):
    """返回文本;失败返回 None。接口与 zai() 对齐,方便两边互换。"""
    for attempt in range(retries):
        fd, path = tempfile.mkstemp(suffix='.txt'); os.close(fd)
        try:
            subprocess.run(['codex', 'exec', '--ephemeral', '-s', 'read-only', '--skip-git-repo-check',
                            '-m', model, '-o', path, '-'],
                           input=prompt, capture_output=True, text=True, timeout=timeout, cwd='/tmp')
            t = open(path).read().strip()
            if t:
                return t
            LUNA_FAILS[f'空内容#{attempt}'] += 1
        except subprocess.TimeoutExpired:
            LUNA_FAILS['超时'] += 1
        except Exception as ex:
            LUNA_FAILS[type(ex).__name__] += 1
        finally:
            try: os.remove(path)
            except Exception: pass
        time.sleep(1 + attempt * 2)
    LUNA_FAILS['最终失败'] += 1
    return None
