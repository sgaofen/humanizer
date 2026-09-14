#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RL 奖励(2026-09-11,用户批准的设计):针对事实错误的奖励,没有任何风格项、没有任何检测器项。
    reward = −(严重事实错 ×3 + 轻微事实错 ×0.5 + 编造 ×2) − 2·意思反转 − 1.5·格式丢失 − 1.5·问候落款丢失
    硬约束(不判定直接给底分 HARD_FAIL):语言切换、输出退化(切不出句子/乱码)、空输出、照抄>0.85(纯省判定调用)。
    照抄本身 09-13 起改成超线性斜坡,见 copy_penalty。
    事实逐条核对用 fact ledger(build_ledger.py 抽好的清单),GLM-5.3 对每条给 kept / changed / dropped,再列 invented。
用法(离线批量):python3 scripts/rl_reward.py --ledger data/rl_ledger.jsonl --samples data/rl_samples_it1.jsonl --out data/rl_rewards_it1.jsonl --workers 8
samples 每行 {key, cands:[text,...]};输出每行 {key, rewards:[...], details:[...]}。"""
import argparse, json, os, re, sys, threading, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_http import zai
from textmetrics import all_metrics, is_cjk

HARD_FAIL = -8.0
COPY_THR = 0.35          # 只剩记录用途;09-13 起照抄不再有悬崖
# ── 照抄罚:悬崖换斜坡(2026-09-13)──────────────────────────────────────────────
# 旧形状:copy>0.35 直接 -8、以下一律 0 分。于是"抄 0.34 是免费的",模型学会贴着门槛走
# (三版 RL 的最大值精确停在 0.340/0.333/0.333)。训练日志实测:15-17% 的采样被这个门砸中,
# 且 100% 是因为照抄 —— 审计说它贡献了奖励绝对值的 67-80%,把事实信号压扁。
# 新形状:超线性,低位几乎不罚、高位很凶。免罚线锚在【训练对两侧之间的照抄率】上
# (新尺子量 n=300:中位 0.096、P75 0.170、P90 0.257)。
# ★ 这个数【不是】"人改写时会抄多少" —— 没有任何人改写过任何东西。训练对里人写文本是原始素材,
#   草稿是 GLM 照着它造的,所以这个重合度量的是"GLM 造草稿时复用了多少原话",是我们数据管线的性质。
#   它能当锚的理由是别的:训练方向是 草稿→人写,模型学的就是这个映射,映射本身的照抄率就是 0.096,
#   模型输出本来就该收敛到这里。也就是说这是个【自指】的锚 —— 当初造草稿时改得越狠,这个锚就越低。
#   别再把它叫成"人类基线"。
#   0.10→0    0.15→0.02  0.20→0.11  0.25→0.27  0.30→0.51  0.35→0.83
#   0.50→2.33 0.65→4.70  0.80→8.00(封顶)
REUSE_FREE = 0.31        # 免罚线 = 训练对目标映射的 reuse 中位数(0.311),见下
W_COPY = float(__import__('os').environ.get('W_COPY', 8.0))   # 09-13 R7:可由 --export W_COPY=4 覆盖。R6(8.0)reuse .365→.28 但 final 严重错 2→7,斜坡与严重错(3.0)同量级,模型选择深改
K_COPY = 1.61            # 指数 >1:0.7 的罚是 0.5 的 3.2 倍,线性只会是 1.4 倍
CAP_COPY = W_COPY
COPY_SKIP = 0.85         # 【原词照抄】到这个程度不必再花判定调用,直接给满罚


def copy_penalty(c):
    """复用率 reuse → 扣分。超线性,0.31 以下免罚,1.0 封顶 8.0。

    输入 09-13 起从 copy 换成 reuse = max(词面照抄, 句法骨架复用)。原因:逐句同义替换时
    copy 只有 0.28,而骨架复用是 0.90 —— 模型最主要的抄法(句序分句全不动、只换同义词)
    在 copy 上几乎看不见。骨架的分母是草稿,灌废话稀释不了。
    免罚线 0.31 标在【训练对目标映射】的 reuse 中位数上(0.311),它是自指的锚:
    模型学的就是这个映射,输出本来就该收敛到这里。不是"人类基线"——没人改写过任何东西。
    实测落点:发布版 0.379、R1 0.405、R2 0.408、R3 0.426、手写真重构 0.11–0.15。
    """
    return min(CAP_COPY, W_COPY * max(0.0, (float(c) - REUSE_FREE) / (1.0 - REUSE_FREE)) ** K_COPY)

P = '''You are checking whether a REWRITE preserved the facts of the original. You are given the original's FACT LIST (extracted beforehand) and the REWRITE. Judge each fact against the REWRITE only:
- "kept": the fact is stated in the REWRITE with the same meaning (rewording, reordering, merging, splitting, passive/active, and re-expressing a number in an equivalent form such as 'ten' for '10' are all fine);
- "changed": the REWRITE states it differently in substance;
- "dropped": the REWRITE does not state it at all.
For every fact that is NOT "kept", also give:
- "severity": judge by whether a reader acting on the text would be misled.
  "critical" = a fact reversed or contradicted; a number, name, date, amount or deadline changed or dropped; who-did-what or who-asks-whom changed; a completed action turned into an intention or the reverse; a request, decision or action item lost; an invented fact or quote.
  "minor" = the output loses a concrete detail or shifts a quantified boundary without breaking the main fact — a dropped specific ("timestamps display in UTC" becomes "no timezone conversion", "prior pilot data" becomes "pilot data"), a changed boundary or frequency ("at least 200" -> "more than 200", "doubles in 4 to 6 hours" -> "doubles every 4-6 hours"), a dropped statistical or evidential qualifier ("significantly improves" -> "improves"), or an open question stated as an established finding ("examined whether X impairs Y" -> "X impairs Y").
  **NOT an error at all — use "kept" and do not report these**: synonym substitution that preserves the meaning ("largest" -> "most significant", "provides evidence" -> "offers evidence", "biggest shock" -> "scariest thing", "interact rather than contribute independently" -> "do not work in isolation"); shifts in politeness, warmth or emotional emphasis ("would prefer" -> "would much prefer", "impressed" -> "extremely impressed", "happy to" -> "prepared to"); sentence restructuring, merging, splitting, reordering; register choices such as contractions, slang, informal punctuation or their absence; changing voice. Only flag a wording change when it alters what is true, how certain it is, or how much there is.
- "span": copy the FIRST EIGHT WORDS, verbatim, of the sentence in the REWRITE that carries this error (for a "dropped" fact, the sentence where it should have been); use "" if no single sentence carries it.
Then list "invented": facts, claims, numbers, reasons, promises or examples the REWRITE states that are not in the fact list or the original (pure connectives/framing do not count), each as {"text": "...", "span": "<first eight words of the sentence>"}. "register_kept": true unless the REWRITE normalises the register of the ORIGINAL — expanding its contractions (isnt -> is not, we're -> we are), replacing its slang or colloquial idioms with formal wording (has a clue -> knows, kinda -> somewhat), adding punctuation it deliberately omits (question marks, apostrophes, sentence-final periods), capitalising what it leaves lowercase, or deleting its verbal tics (wait, like, honestly, lol). Mirroring the original's informality is correct; tidying it up is not. If the ORIGINAL is already formal, this is true. "reversed_meaning": true if any claim's direction is flipped or an opinion became a fact. "format_kept": subject line / title / list structure / paragraph breaks / headers of the original survive (true if it has none; rewording a subject line or sign-off is fine as long as the element is still there). "greeting_signoff_kept": greeting, sign-off, signature and contact lines survive with the same names (true if none).
Output ONLY JSON: {"facts": [{"id": 1, "status": "kept|changed|dropped", "severity": "critical|minor", "span": "...", "note": "<=12 words"}, ...], "invented": [{"text": "...", "span": "..."}], "reversed_meaning": true/false, "register_kept": true/false, "format_kept": true/false, "greeting_signoff_kept": true/false}

FACT LIST:
{facts}

ORIGINAL (for format/greeting reference only):
"""{d}"""

REWRITE:
"""{o}"""'''

# 09-11 R1 的教训:每条事实错都按 -1 计,严重错与轻微错同价 → 模型去消灭便宜的轻微错(评测上 27→17),
# 而严重错没被特别惩罚反升(5→9)。R2 按严重度分级,并与评测的 severity 同轴。
W_CRITICAL = 3.0
W_MINOR = 0.15   # 09-12:原 0.5。R2 实测 18 条轻微 vs 1 条严重,轻微总扣分是严重的 3 倍,而人工逐条读发现近半是同义替换/语气微调的误报 → 模型学到"别换词最安全"从而规整化(过检 81%→58%)
W_INVENTED = 2.0
W_REVERSED = 2.0
W_REGISTER = 1.0   # 09-12:RL 三版一致把过检率压到 52–58%(判定误报已归零,所以不是尺子的问题),
                   # 说明只罚事实错就会让模型往保守完整规范收敛,而规范正是 AI 味。人味必须显式进奖励:
                   # 罚的是"偏离草稿的语域",不是检测器分数。
W_FORMAT = 1.5
W_GREETING = 1.5


def hard_check(draft, out):
    """硬约束。返回 (是否硬失败, 原因, 量尺)。"""
    if not (out or '').strip():
        return True, 'empty', None
    m = all_metrics(draft, out)
    if m['degenerate']:
        return True, 'degenerate', m
    if m['copy'] > COPY_SKIP:
        return True, f'copy {m["copy"]:.2f}', m      # 只有抄到 0.85 以上才跳过判定(省 GLM 调用),其余走斜坡
    if is_cjk(draft) != is_cjk(out):
        return True, 'language_switch', m
    return False, '', m


def _judge_once(draft, out, facts):
    fl = '\n'.join(f'{x["id"]}. {x["text"]}' for x in facts)
    p = P.replace('{facts}', fl).replace('{d}', draft).replace('{o}', out)
    for attempt in range(2):
        t = zai(p, model='glm-5.3', max_tokens=1200, timeout=240) or ''
        mm = re.search(r'\{.*\}', t, re.S)
        try:
            v = json.loads(mm.group(0))
            if isinstance(v.get('facts'), list):
                return v
        except Exception:
            pass
    return None


def judge(draft, out, facts, votes=2):
    """判两次取一致(09-12):实测同一批输出判两次只有 73–77% 的条目判定一致,
       四分之一会跳档,而且"至少一次判严重"的条目里只有一半两次都判严重。
       噪声一半指向同义替换/语气微调,会把模型逼向"别换词最安全"的规整化。
       所以:只有【两次都判为非 kept】的事实才算错,severity 取两次里较轻的一档(保守);
       invented / reversed / format / greeting 同样要两次都报才算。votes=1 时退回单判。"""
    a = _judge_once(draft, out, facts)
    if a is None or votes < 2:
        return a
    b = _judge_once(draft, out, facts)
    if b is None:
        return a
    sb = {x.get('id'): x for x in b.get('facts', []) if str(x.get('status', '')).lower() in ('changed', 'dropped')}
    merged = []
    for x in a.get('facts', []):
        st = str(x.get('status', '')).lower()
        y = sb.get(x.get('id'))
        if st in ('changed', 'dropped') and y is not None:
            sev_a = str(x.get('severity', 'critical')).lower(); sev_b = str(y.get('severity', 'critical')).lower()
            merged.append(dict(x, severity=('minor' if 'minor' in (sev_a, sev_b) else 'critical')))
        else:
            merged.append(dict(x, status='kept'))
    inv_a = a.get('invented') or []; inv_b = b.get('invented') or []
    return {'facts': merged,
            'invented': inv_a if (inv_a and inv_b) else [],
            'reversed_meaning': bool(a.get('reversed_meaning') and b.get('reversed_meaning')),
            'register_kept': not (a.get('register_kept') is False and b.get('register_kept') is False),
            'format_kept': not (a.get('format_kept') is False and b.get('format_kept') is False),
            'greeting_signoff_kept': not (a.get('greeting_signoff_kept') is False and b.get('greeting_signoff_kept') is False)}


def score(v, n_facts, reuse=0.0):
    """把判定折成标量 + 句级扣分表。全部保留、无编造、格式在 → 0(最高)。
       返回 (总分, 明细, {span 前八词: 扣分})。span 用于把惩罚定位到承载错误的那句话(句级 credit)。"""
    spans = {}
    def _add(sp, w):
        sp = (sp or '').strip()
        if sp:
            spans[sp] = spans.get(sp, 0.0) + w
    r = 0.0; crit = minor = dropped = changed = 0
    for x in v.get('facts', []):
        st = str(x.get('status', '')).lower()
        if st not in ('changed', 'dropped'):
            continue
        changed += st == 'changed'; dropped += st == 'dropped'
        w = W_CRITICAL if str(x.get('severity', 'critical')).lower() == 'critical' else W_MINOR
        crit += w == W_CRITICAL; minor += w == W_MINOR
        r -= w; _add(x.get('span'), w)
    inv = v.get('invented') or []
    for x in inv:
        if isinstance(x, dict):
            r -= W_INVENTED; _add(x.get('span'), W_INVENTED)
        else:
            r -= W_INVENTED
    if v.get('reversed_meaning'): r -= W_REVERSED
    if v.get('register_kept') is False: r -= W_REGISTER
    if v.get('format_kept') is False: r -= W_FORMAT
    if v.get('greeting_signoff_kept') is False: r -= W_GREETING
    cp = copy_penalty(reuse); r -= cp
    return r, {'copy_pen': round(cp, 3), 'changed': changed, 'dropped': dropped, 'invented': len(inv), 'critical': crit, 'minor': minor,
               'reversed': bool(v.get('reversed_meaning')), 'register_kept': v.get('register_kept'), 'format_kept': v.get('format_kept'),
               'greeting_kept': v.get('greeting_signoff_kept'), 'n_facts': n_facts}, spans


def reward(draft, out, facts, votes=2):
    """单条奖励。返回 (reward, detail)。"""
    hard, why, m = hard_check(draft, out)
    if hard:
        return HARD_FAIL, {'hard_fail': why, 'copy': (m or {}).get('copy')}
    v = judge(draft, out, facts, votes=votes)
    if v is None:
        return None, {'judge_failed': True, 'copy': m['copy']}
    r, d, spans = score(v, len(facts), m.get('reuse', m['copy'])); d['copy'] = m['copy']; d['align_sim'] = m['align_sim']; d['format_recall'] = m['format_recall']
    d['spans'] = spans
    d['facts'] = [(x.get('id'), x.get('status'), x.get('note', '')) for x in v.get('facts', []) if x.get('status') != 'kept']
    d['invented_list'] = v.get('invented') or []
    return r, d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger', required=True); ap.add_argument('--samples', required=True); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    L = {}
    for l in open(a.ledger):
        r = json.loads(l); L[r['key']] = r
    S = [json.loads(l) for l in open(a.samples)]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)['key'])
            except Exception: pass
    S = [s for s in S if s['key'] in L and s['key'] not in done]
    print(f'待评 {len(S)} 组(已有 {len(done)})', flush=True)
    lock = threading.Lock(); n = 0
    def work(s):
        led = L[s['key']]; out = []
        for c in s['cands']:
            r, d = reward(led['draft'], c, led['facts']); out.append((r, d))
        return {'key': s['key'], 'src': led.get('src'), 'rewards': [x[0] for x in out], 'details': [x[1] for x in out]}
    with open(a.out, 'a') as f, ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(work, S):
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush(); n += 1
                if n % 50 == 0: print(f'已评 {n}', flush=True)
    print(f'REWARD_DONE {n} 组 → {a.out}', flush=True)
