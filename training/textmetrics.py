#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""textmetrics.py — 改写任务的硬量尺,唯一事实源(2026-09-11 审计后重写;同日对抗审计后二次修订)。

以后配对(build_prefs)、生成端守卫(gen_cases / local_eval / 发布推理)、RL 奖励的硬约束都只从这里 import。

v1(09-11 上午)修的四点:整句照搬只换顺序(verbatim_sent)、缩写/小数分句、位置对齐分不清重排与拆并、中文全盲。
v2(09-11 对抗审计,子代理构造样本复现后修):
  S1 语言判定翻转:不再"整篇判语言二选一"。词 5-gram 与中文字 gram、英文词 token 与中文字二元组【同时】算,拼在一起,
     草稿或输出任一含中文就带上字 gram。英文前言 + 中文原文照抄 → copy 仍 ≈1。
  S2 分母在输出侧被灌水稀释:加 draft_coverage(草稿 gram 被输出命中的比例)与 verbatim_draft(草稿句被整句复现的比例),
     copy = max(输出侧照抄, 草稿侧被抄, 两向整句复现)。抄全文 + 灌五段废话 → copy 仍 ≈1。
     填充词插入("I hope in fact this email finds as such you well")打断 n-gram → 整句匹配改为词级有序包含(matching blocks / 草稿句长),
     插词不降低包含率;另给 lcs_content(内容 token 有序公共子序列 / 草稿内容 token 数)作 identity 兜底。
  S3/S4 结构分奖励删内容:one_to_one 只在"被匹配到的草稿句"里算(拆并),另给 coverage(草稿句被覆盖比例)与 fact_recall
     (数字/邮箱/网址/全大写缩写 召回),删句、删落款、丢数字不再是"改得深"。
  S5 否定/量词在停用词表里:no not nor all any most only same such few more none 移出停用词。
  S7 整句照搬 0.9 字符阈值实为句长探测器:改为词级(中文字级)有序包含 ≥.9,且草稿句 ≥5 词 / ≥8 字才参与。
  S9 分句被引号/HTML 标签/碎片破坏:句末标点后允许闭引号/括号;先剥 HTML 标签;输出非空但切不出句子 → degenerate=True 且结构读数取最差。
  S10 退化输出:order_agree 只数 b>a(b==a 记为合并);重复同一句/只写一句 → coverage 极低。
  S11 数字密度顶高 align_sim:align 的 Jaccard 剔除含数字的 token 与 fig/s1 之类引用标记(它们必须保留,不是"没改"的证据)。
  S14 硬换行草稿被切碎:段内单换行只在 行末无终止标点、下一行不是列表项、且本行 ≥30 字符 时视为软换行拼接。
  S8 中文 8 字 gram 与英文 5 词 gram 灵敏度不一致:字 gram 改 6 字(真实模型输出上中英中位数持平)。
  S3 补:主题行/称呼/落款/签名块成对剥离(split_format),照抄与结构只在正文上量;格式要素另给 format_recall(审计实证:旧结构分的赢家 84% 丢主题行)。
  C1 弯撇号归一(don’t 与 don't 曾切成不同 token)。

所有指标 0~1,越高越"像草稿"(coverage/fact_recall 例外:越高越"内容没丢")。没有任何检测器分数参与。
"""
import re
from difflib import SequenceMatcher

_ABBR = {'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 'st', 'vs', 'etc', 'e.g', 'i.e', 'fig', 'figs', 'no', 'vol', 'inc', 'ltd', 'co', 'corp',
         'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec', 'u.s', 'u.k', 'ph.d', 'approx', 'dept', 'univ', 'al', 'eq', 'ref', 'refs', 'cf', 'ca', 'pp', 'p'}
# 09-11 v2:否定词与量词(no not nor all any most only same such few more none)不再是停用词——它们是语义极性与量化范围的载体
_STOP = set('''a an the and or but if then than so of to in on at by for from with without into onto over under about as is are was were be been being am
do does did done have has had having i you he she it we they me him her us them my your his its our their this that these those there here who whom which what when where why how
very just also too own can could will would shall should may might must need dare up down out off again further once
because while until although though whether either neither'''.split())
_CJK = re.compile(r'[㐀-䶿一-鿿豈-﫿]')
_CJK_K = 6          # 中文字 gram 长度(v2:8→6,随机编辑率标定下与英文 5 词 gram 灵敏度最接近)
_LIST = re.compile(r'^\s*(?:[-*•·▪]|\d{1,3}[.)、:：]|[a-zA-Z][.)]|[①-⑩]|\(\d{1,3}\)|#+\s)')
_CLOSE = '"\'”’)]』」》'


def has_cjk(t):
    return bool(_CJK.search(t or ''))


def is_cjk(t):
    """整体是否中文文本:前 400 字里中文字符多于英文词。只用于报表分组,量尺本身不再按它分支。"""
    head = (t or '')[:400]; return len(_CJK.findall(head)) > len(re.findall(r'[A-Za-z]{2,}', head))


def strip_markup(t):
    """剥掉 HTML 标签:块级标签变换行,其余变空格。"""
    t = re.sub(r'(?i)</?(?:p|br|div|li|ul|ol|h[1-6]|tr|table|blockquote)\b[^>]*>', '\n', t or '')
    return re.sub(r'<[^<>]{1,60}>', ' ', t)


def _join_soft_wraps(para):
    """段内单换行:行末没有终止标点、下一行不是列表项、且本行够长(≥30 字符)→ 视为软换行,拼成一句。"""
    lines = para.split('\n'); out = []
    for i, ln in enumerate(lines):
        s = ln.rstrip()
        if not s:
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ''
        soft = (out and not re.search(r'[.!?:;。！？；：…]["\'”’)\]]*\s*$', out[-1]) and not _LIST.match(s) and len(out[-1].strip()) >= 30)
        if soft:
            out[-1] = out[-1].rstrip() + ' ' + s.lstrip()
        else:
            out.append(s)
    return out


def _split_en(seg):
    """英文分句:在 [.!?] + 可选闭引号/括号 + 空白 处切;不切:缩写(Dr. Oct. e.g. U.S.)、单字母首字母、行首列表序号("1." "a.")。"""
    parts = re.split(r'(?<=[.!?])([' + re.escape(_CLOSE) + r']*)(\s+)', seg)
    out = []; buf = ''
    i = 0
    while i < len(parts):
        text = parts[i]; q = parts[i + 1] if i + 1 < len(parts) else ''; ws = parts[i + 2] if i + 2 < len(parts) else ''
        buf += text + q
        if ws:
            prev = text.rstrip(); toks = prev.split(); last = toks[-1] if toks else ''
            core = last.rstrip('.!?').lower()
            is_abbr = prev.endswith('.') and (core in _ABBR or re.fullmatch(r'[a-z]', core) is not None or re.fullmatch(r'(?:[a-z]\.)+[a-z]?', core) is not None)
            is_marker = prev.endswith('.') and len(buf.split()) == 1 and re.fullmatch(r'[0-9]{1,3}|[a-z]|[ivx]{1,4}', core) is not None
            if is_abbr or is_marker:
                buf += ws
            else:
                out.append(buf); buf = ''
        i += 3
    if buf.strip():
        out.append(buf)
    return out


def sentences(t):
    """分句。段落(空行)是边界;段内软换行拼接;中文按 。!?; 切,英文按缩写感知规则切,混排段两套都切。
       过滤掉碎片(英文 <3 词、中文 <4 字)。"""
    t = strip_markup(t).strip()
    if not t:
        return []
    out = []
    for para in re.split(r'\n\s*\n', t):
        for line in _join_soft_wraps(para):
            pieces = re.split(r'(?<=[。！？；])', line) if has_cjk(line) else [line]
            for pc in pieces:
                pc = pc.strip()
                if not pc:
                    continue
                out += _split_en(pc) if re.search(r'[.!?]', pc) else [pc]
    res = []
    for s in out:
        s = s.strip()
        if not s:
            continue
        if _CJK.search(s):
            if len(_CJK.findall(s)) >= 4 or len(s.split()) >= 3: res.append(s)
        elif len(s.split()) >= 3:
            res.append(s)
    return res


_QUOTE = str.maketrans({'’': "'", '‘': "'", '`': "'", '´': "'"})


def _words(t):
    """小写英文词/数字串;弯撇号(don’t)先归一成直撇号,否则同一个词在两边切成不同 token(审计 C1)。"""
    return re.findall(r"[a-z0-9']+", (t or '').lower().translate(_QUOTE))


def tokens(t):
    """内容 token:英文小写词(去停用词,保留数字)+ 中文字二元组(数字/字母串整体保留)。不按语言分支,两套同时出。"""
    toks = [w for w in _words(t) if w not in _STOP]
    if _CJK.search(t):
        s = re.sub(r'\s+', '', t); chars = _CJK.findall(s)
        toks += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return toks


_NUMLIKE = re.compile(r'.*\d.*|fig|figs|eq|ref|refs|table|tbl|s\d*|[ivx]{1,4}')


def _align_tokens(t):
    """align_sim 用的 token:剔除含数字的 token 与 fig/table/s2 之类引用标记(必须保留的东西不算"没改")。"""
    return [w for w in tokens(t) if not _NUMLIKE.fullmatch(w)]


def _grams(t, cjk=None, n=5):
    if cjk is None:
        cjk = has_cjk(t)
    w = _words(t)
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if cjk:
        s = re.sub(r'\s+', '', t); k = _CJK_K
        g += [s[i:i + k] for i in range(len(s) - k + 1) if _CJK.search(s[i:i + k])]
    return g


def copy_5gram(draft, out):
    """输出的 5-gram(中文 6 字)里有多少在草稿里出现过(输出侧分母);按 token 计数,重复照搬也算。"""
    cjk = has_cjk(draft) or has_cjk(out)
    g = _grams(out, cjk)
    if not g:
        return 0.0
    dg = set(_grams(draft, cjk))
    return sum(1 for x in g if x in dg) / len(g)


def draft_coverage(draft, out):
    """草稿的 5-gram(中文 6 字)有多少在输出里出现过(草稿侧分母)。抄全文再灌废话,这个不会被稀释。"""
    cjk = has_cjk(draft) or has_cjk(out)
    dg = _grams(draft, cjk)
    if not dg:
        return 0.0
    og = set(_grams(out, cjk))
    return sum(1 for x in dg if x in og) / len(dg)


def _units(s):
    """整句匹配的单位:中文句用字(去空白、去标点),英文句用小写词。"""
    if _CJK.search(s):
        return [ch for ch in re.sub(r'\s+', '', s.lower()) if ch.isalnum()]
    return _words(s)


def _contain(a, b):
    """有序包含率:a 的单位里有多少能按顺序在 b 里对上(matching blocks 总长 / len(a))。插词不降低它。"""
    if not a or not b:
        return 0.0
    sm = SequenceMatcher(None, a, b, autojunk=False)
    return sum(bl.size for bl in sm.get_matching_blocks()) / len(a)


def _long_enough(u, s):
    return len(u) >= (8 if _CJK.search(s) else 5)


def verbatim_pair(draft, out, thr=0.9):
    """返回 (verbatim_out, verbatim_draft):
       verbatim_out   输出句里有多大比例几乎整句来自草稿(该输出句 ≥.9 落在某草稿句里,或某草稿句 ≥.9 落在它里面)
       verbatim_draft 草稿句里有多大比例被输出几乎整句复现(≥.9 有序包含在某输出句里)
       只有够长的句子参与(英文 ≥5 词、中文 ≥8 字),碎片不算。
       09-13:两个比例都改成【按词/字数】计权。原来按句数等权,于是改不动的样板行(`## v2.3.2 (2026-09-12)`、
       `- Bump x from A to B (#N).`、带 @负责人 和截止日期的待办行)被超额计权,一篇正文全改写过的
       变更日志/会议纪要照样冲过 0.35 吃 −8。"""
    ds = [(s, _units(s)) for s in sentences(draft)]; os_ = [(s, _units(s)) for s in sentences(out)]
    ds = [(s, u) for s, u in ds if _long_enough(u, s)]; os_ = [(s, u) for s, u in os_ if _long_enough(u, s)]
    if not ds or not os_:
        return 0.0, 0.0
    hit_o = 0; hit_d = [False] * len(ds)
    for so, uo in os_:
        this = False
        for j, (sd, ud) in enumerate(ds):
            c_d = _contain(ud, uo)          # 草稿句落在输出句里的比例
            if c_d >= thr:
                hit_d[j] = True; this = True
            elif not this and _contain(uo, ud) >= thr:   # 输出句是草稿句的一块
                this = True
        if this:
            hit_o += len(uo)                # 09-13:按词/字数计权,不按句数
    tot_o = sum(len(u) for _, u in os_) or 1
    tot_d = sum(len(u) for _, u in ds) or 1
    return hit_o / tot_o, sum(len(u) for (s_, u), h in zip(ds, hit_d) if h) / tot_d


def verbatim_sent(draft, out, thr=0.9):
    """兼容旧名:= max(两个方向)。"""
    a, b = verbatim_pair(draft, out, thr); return max(a, b)


def lcs_content(draft, out):
    """草稿内容 token 的有序公共子序列占比(matching blocks / 草稿内容 token 数)。抄全文+插填充词 → ≈1;真改写通常 <.5。
       用作 identity 兜底,不进 copy 的 max(见 all_metrics 注释)。"""
    a = tokens(draft); b = tokens(out)
    return _contain(a, b)


def skeleton(t):
    """抹掉实词,只留虚词 + 标点 + 数字位。同义替换时骨架一字不变,所以它抓得住 copy 漏掉的那类抄。
       中文返回 None:虚词骨架在中文失效(没有英文那样干净的虚词表),退回纯 copy —— 已知缺口,别假装填上。"""
    if _CJK.search(t or ''):
        return None
    out = []
    for w in re.findall(r"[A-Za-z']+|[0-9][0-9,.%$]*|[^\sA-Za-z0-9]", t or ''):
        if re.match(r"[A-Za-z']", w):
            out.append(w.lower() if w.lower() in _STOP else 'W')
        elif re.match(r'[0-9]', w):
            out.append('N')
        else:
            out.append(w)
    return out


def skeleton_recall(draft, out, n=5):
    """草稿骨架的 5-gram 有多少在输出里出现。分母是【草稿】,所以灌废话稀释不了。"""
    a, b = skeleton(draft), skeleton(out)
    if a is None or b is None:
        return None
    bs = set(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
    g = [tuple(a[i:i + n]) for i in range(len(a) - n + 1)]
    return sum(1 for x in g if x in bs) / len(g) if g else 0.0


def reuse(draft, out):
    """复用率 = max(词面照抄, 句法骨架复用)。两条互补,实测(09-13):

         攻击/对照              copy   骨架
         原样照抄               1.00   1.00
         逐句同义替换(句序不动)   0.28   0.90   ← copy 漏,骨架抓
         照抄+每句插填充词        1.00   0.29   ← 骨架漏,copy 抓
         照抄+灌废话            1.00   1.00
         真改写                 0.00   0.00   ← 都放过
         真重构                 0.00   0.00   ← 都放过

       为什么用 max 不是两项:同题材内 Spearman 0.77,高度相关,两项等于把照抄权重乘二。
       骨架的分母是草稿,与 draft_coverage 同理 —— 源长度当分母有 Broder containment、JPlag
       similarityOfFirst、ROUGE recall、SARI、MOSS、GPLAG、PAN recall 七条先例。
    """
    c = copy(draft, out)
    sk = skeleton_recall(draft, out)
    return c if sk is None else max(c, sk)


def extractive_fragments(draft, out):
    """Grusky et al. (NAACL 2018) 的抽取式片段 → (coverage, density)。

    coverage = 输出里有多少词属于"从草稿整段搬来"的片段;density = 每个词所属片段的平均长度(平方加权)。
    人名、数字、术语必须逐字保留,它们抬高 coverage 但不抬高 density —— 所以 density 才是
    "有没有抄措辞"的公平量,而 copy_5gram 把两者混在一起还硬卡在 5 词窗口。

    ★ 只作【诊断】,不许进奖励:09-13 实测 density 按输出长度归一,"照抄+每句插填充词"能把它从 37 压到 2.67,
      而真改写是 1.90 —— 几乎分不开。copy 的 max-of-four 里有 draft_coverage(分母是草稿),灌废话稀释不了,
      所以奖励仍然用 copy。
    """
    A = _words(draft); S = _words(out)
    if not S:
        return 0.0, 0.0
    F = []; i = 0
    while i < len(S):
        f = []; j = 0
        while j < len(A):
            if S[i] == A[j]:
                i_, j_ = i, j
                while i_ < len(S) and j_ < len(A) and S[i_] == A[j_]:
                    i_ += 1; j_ += 1
                if len(f) < i_ - i:
                    f = S[i:i_]
                j = j_
            else:
                j += 1
        i += max(len(f), 1)
        if f: F.append(f)
    return sum(len(x) for x in F) / len(S), sum(len(x) ** 2 for x in F) / len(S)


def copy(draft, out):
    """综合照抄率 = max(输出侧 5-gram 照抄, 草稿侧 gram 被抄比例, 两向整句复现)。任一路径抄了都算。"""
    vo, vd = verbatim_pair(draft, out)
    return max(copy_5gram(draft, out), draft_coverage(draft, out), vo, vd)


def _jac(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))


_FACT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|https?://\S+|www\.\S+|\d+(?:[.,:/-]\d+)*%?|\b[A-Z][A-Z0-9]{1,}\b")


def facts(t):
    """硬事实 token:邮箱、网址、数字(含 1,200 / 3.5 / 2019-2021 / 12%)、全大写缩写(DOD、CCL2)。数字去千分位逗号。"""
    out = set()
    for m in _FACT.findall(t or ''):
        x = m.rstrip('.,;:)')
        if re.fullmatch(r'\d+(?:[.,:/-]\d+)*%?', x):
            x = x.replace(',', '')
            for part in re.split(r'[/:-]', x.rstrip('%')):      # 2019-2021 / 14:30 / 6/22 → 各成分也算一个事实,改写成"2019到2021"仍算保留
                if part and part != x: out.add(part)
        out.add(x.lower())
    return out


def fact_recall(draft, out):
    """草稿的硬事实 token 有多少出现在输出里(草稿没有则 1.0)。不是判错(判错归 GLM),只用来在干净候选里排序/守底。"""
    fd = facts(draft)
    if not fd:
        return 1.0
    fo = facts(out); on = re.sub(r'\s+', '', (out or '').lower())
    return sum(1 for x in fd if x in fo or x.replace(',', '') in on) / len(fd)


_FMT = re.compile(r'^\s*(?:(?:subject|re|fw|fwd|to|from|cc|date|title)\s*[:：]|(?:dear|hi|hello|hey|good (?:morning|afternoon|evening))\b|(?:best|best regards|best wishes|regards|kind regards|warm regards|warmly|sincerely|sincerely yours|cheers|thanks|thank you|many thanks|thanks again|yours|yours truly|respectfully|take care|talk soon|all the best)\s*[,!.]?\s*$|#{1,6}\s|(?:尊敬的|亲爱的|各位|敬爱的)|[^\s,，:：]{1,12}(?:老师|教授|同学|先生|女士|经理|总|博士|主任|导师)[:：,，]?\s*$|(?:此致|敬礼|祝好|祝安|祝|顺祝|顺颂|谢谢|感谢|敬上|谨上|学生|申请人|您好)[:：!！。,，]?\s*$)', re.I)
_SIG = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+|\d{4}[-/年.]\d{1,2}([-/月.]\d{1,2}日?)?|^\s*[—–]{1,2}\s*\S')


_FENCE = re.compile(r'^\s*(?:```|~~~)')
_RULE_LN = re.compile(r'^\s*(?:[-*_]\s*){3,}$')
_FIELD = re.compile(r'^\s*[A-Za-z][A-Za-z /_-]{0,20}\s*[:：]\s*\S')
_ENDPUNCT = re.compile(r'[.!?。!?…]["\'”’)\]]?\s*$')
_DATAISH = re.compile(r'\d|@|https?://|www\.|\.(?:com|org|net|io|edu|gov|cn)\b|[/\\]|[—–|·]\s*\S+\s*[—–|·]')
_SIGNOFF = re.compile(r'^\s*(?:best|best regards|best wishes|regards|kind regards|warm regards|warmly|sincerely|'
                      r'sincerely yours|cheers|thanks|thanks again|thank you|many thanks|yours|yours truly|'
                      r'respectfully|take care|talk soon|all the best|thanks so much|thanks a lot|warm wishes|gratefully|with appreciation|much appreciated|appreciate it|with thanks|best of luck)\s*[,!.]?\s*$'
                      r'|^\s*(?:此致|敬礼|祝好|祝安|顺祝|顺颂|敬上|谨上)[:：!！。,,]?\s*$', re.I)
_GREET = re.compile(r'^\s*(?:dear|hi|hello|hey|good (?:morning|afternoon|evening))\b'
                    r'|^\s*(?:尊敬的|亲爱的|各位|敬爱的)'
                    r'|^[^\s,,::]{1,12}(?:老师|教授|同学|先生|女士|经理|总|博士|主任|导师)[:：,,]?\s*$', re.I)


def _dataish(ln):
    """这行像不像"数据"而不是"话":含数字/邮箱/链接/路径,或 Title Case 密集(公司名、职位、地名)。"""
    if _DATAISH.search(ln):
        return True
    w = [x for x in ln.split() if x[:1].isalpha()]
    return len(w) >= 2 and sum(1 for x in w if x[:1].isupper()) / len(w) >= 0.6


def split_prose(t):
    """把一段文本切成 (散文行, 结构行)。结构行 = 设计上就要求一字不改的东西(2026-09-13)。

    为什么换掉按行枚举格式要素的老路:09-13 的误报筛查(106 个 agent、16 个格式家族、逐条独立复现)
    确认 70 条误报,其中 18 条是照抄硬失败 −8,全是同一个病 —— 主题行、签名块、Markdown 表格、
    代码块、引用块、联系方式、版本标题、待办勾选行、依赖升级行这些必须逐字保留的东西进了照抄率的分母,
    于是保留得越忠实照抄率越高。老办法 split_format 逐条枚举"什么算格式",枚举不完(光 line 291 一处就吃了 8 条)。
    这里反过来只认散文,认不出来的一律归结构:宁可少算照抄,不可冤枉合法改写。
    真照抄仍然抓得住 —— 整篇照抄时散文行同样被逐字抄了(tests/test_fp.py 有负样本守着)。
    """
    prose, struct = [], []
    fence = False; after_signoff = False
    lines = (strip_markup(t) or '').split('\n')   # 先剥 HTML:块级标签变换行,否则 <p> 会被当成一整行长散文,
                                                #   标签本身还会污染 5-gram(09-13 自测 S9 那行抓到的)
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    head = set(nonempty[:3])
    for i, raw in enumerate(lines):
        ln = raw.strip()
        if _FENCE.match(raw):
            fence = not fence; struct.append(ln); continue
        if not ln:
            continue
        if fence:
            struct.append(ln); continue
        if _SIGNOFF.match(ln):
            after_signoff = True; struct.append(ln); continue     # 落款之后全是签名块
        cjk = bool(_CJK.search(ln)); nw = len(tokens(ln))
        drop = (after_signoff
                or raw.startswith('    ') or raw.startswith('\t')          # 缩进代码块
                or ln.startswith('>') or ln.startswith('#')                # 引用 / 标题
                or _RULE_LN.match(ln) or raw.count('|') >= 2               # 分隔线 / 表格行
                or (i in head and _FIELD.match(ln) and (len(ln) <= 30 if cjk else nw <= 12))
                #   ↑ 抬头字段 Subject:/To:/Date: —— 必须带长度护栏。09-13 实测:没护栏时
                #     'What I love: the heat retention is unmatched…' 这种 500 字的正文段会被整条剥掉,
                #     跟当天早些时候中文 split_format 吃掉 157 字正文是同一个病
                or (_GREET.match(ln) and (len(ln) <= 30 if cjk else nw <= 8))   # 称呼行(带长度护栏)
                or (_LIST.match(ln) and (len(ln) < 16 if cjk else nw < 8))      # 短列表项 = 数据行
                or (not _ENDPUNCT.search(ln) and (len(ln) <= 16 if cjk else nw <= 8) and _dataish(ln)))
        (struct if drop else prose).append(ln)
    return '\n'.join(prose), struct


def prose_lines(t):
    return split_prose(t)[0]


def split_format(t):
    """把文档格式要素(主题行/标题/称呼/落款/签名块/联系方式)和正文分开。返回 (格式行列表, 正文)。
       规则:匹配 _FMT 的行;开头 2 行里 ≤6 词/≤12 字的短行;结尾 4 行里 ≤5 词/≤12 字的短行或含邮箱/日期的行。
       09-12 修:中文三条判据全部失效过(称呼分支无行尾锚、长度护栏被 or _CJK 短路、nw<=12 对无空格中文恒真),
       导致 157 字的整段正文被当成格式行剥掉,copy 从 0.25 虚高到 1.00,合法中文改写在 RL 里吃 -8 硬失败。"""
    lines = [ln for ln in (strip_markup(t) or '').split('\n')]
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    fmt = set()
    for pos, i in enumerate(nonempty):
        ln = lines[i].strip(); nw = len(ln.split()); nc = len(ln)
        short = (nw <= 6 and nc <= 60) if not _CJK.search(ln) else nc <= 12
        if _FMT.match(ln) and ((nc <= 30) if _CJK.search(ln) else nw <= 14):
            fmt.add(i)
        elif pos < 2 and short:
            fmt.add(i)
        elif pos >= len(nonempty) - 4 and ((nw <= 4 and nc <= 40 and nw <= 12) if not _CJK.search(ln) else (nc <= 8 or (_SIG.search(ln) and nc <= 40))):
            fmt.add(i)                      # 结尾短行(署名/职务/日期/邮箱);"Thank you for considering this." 这种整句不算
    body = '\n'.join(ln for i, ln in enumerate(lines) if i not in fmt)
    return [lines[i].strip() for i in sorted(fmt)], body


def format_recall(draft, out):
    """草稿的格式行有多少在输出里还在(逐行模糊匹配:去标点后的单位有序包含 ≥.8,或输出某行包含它)。草稿没有格式行则 1.0。
       不是判错(格式判断归 GLM 的 format_kept),只用于选样时不让"删主题行/删落款"当成"改得深"(审计 S3)。"""
    fd, _ = split_format(draft)
    fd = [x for x in fd if not re.fullmatch(r'#+', x.strip())]
    if not fd:
        return 1.0
    ol = [ln.strip() for ln in (strip_markup(out) or '').split('\n') if ln.strip()]
    ou = [_units(ln) for ln in ol]
    hit = 0
    for x in fd:
        ux = _units(x)
        if not ux:
            hit += 1; continue
        if any(_contain(ux, u) >= 0.8 for u in ou):
            hit += 1
    return hit / len(fd)


_KIND = (
    ('subject', re.compile(r'^\s*(?:subject|re|fw|fwd|to|from|cc|date|主题|收件人|发件人)\s*[:：]', re.I)),
    ('title', re.compile(r'^\s*#{1,6}\s')),
    ('greeting', re.compile(r'^\s*(?:dear|hi|hello|hey|good (?:morning|afternoon|evening)|尊敬的|亲爱的|各位|敬爱的)\b', re.I)),
    ('greeting', re.compile(r'^[^\s,，:：]{1,12}(?:老师|教授|同学|先生|女士|经理|总|博士|主任|导师|您好)[:：,，]?\s*$')),
    ('signoff', re.compile(r'^\s*(?:best regards|best wishes|best|sincerely(?: yours)?|kind regards|warm regards|warmly|regards|cheers|many thanks|thanks again|thanks|thank you|yours(?: truly)?|respectfully|take care|talk soon|all the best)\s*[,!.]?\s*$', re.I)),
    ('signoff', re.compile(r'^\s*(?:此致|敬礼|祝好|祝安|顺祝|顺颂|谢谢|感谢|敬上|谨上|学生|申请人)[:：!！。,，]?\s*$')),
)


def fmt_kind(line):
    """格式行的类别:subject / title / greeting / signoff / signature(含邮箱日期电话的落款行)/ other。"""
    for k, rx in _KIND:
        if rx.search(line or ''):
            return k
    if _SIG.search(line or ''):
        return 'signature'
    return 'other'


def paragraphs(t):
    return [x for x in re.split(r'\n\s*\n', (strip_markup(t) or '').strip()) if x.strip()]


def para_kept(draft, out):
    """草稿的分段是否被压平(GLM 判 format_kept 时抓得最多的一类:paragraph breaks collapsed into one block)。
       草稿不足 2 段则 1.0;输出段数少于草稿则按比例下降。"""
    nd = len(paragraphs(draft))
    if nd < 2:
        return 1.0
    return round(min(1.0, len(paragraphs(out)) / nd), 3)


def _shape(ln):
    """结构行的形状。判"要素还在不在"用形状,不用措辞 —— 改写一个标题和删掉一个标题不是一回事。"""
    if _SIG.search(ln or ''):
        return 'signature'                       # 含邮箱/电话/日期:是事实,仍要求逐字
    if _SIGNOFF.match(ln or ''):
        return 'signoff'
    if _GREET.match(ln or ''):
        return 'greeting'
    if (ln or '').startswith('#') or (_FIELD.match(ln or '') and len(ln or '') <= 80):
        return 'header'
    if _LIST.match(ln or ''):
        return 'list'
    if _RULE_LN.match(ln or '') or (ln or '').count('|') >= 2:
        return 'table'
    return 'other'


def format_presence(draft, out):
    """格式【要素】还在不在(不是逐字保没保住)。

    09-11 立这个指标,是因为 format_recall 用词级包含 >=0.8 判命中,于是"主题行改写"和"主题行整行删除"同分。
    09-13 的误报筛查发现它自己也留了同一个尾巴:类别落到 other 的要素仍走逐字包含 >=0.8,于是中文小标题、
    Q:/A: 行、社交帖标题、tl;dr、列表项这些【GLM 明文允许改写】的东西,一改就判 0.00,和删掉同分 —— 20 条误报出在这里。
    现在改成:除 signature(含邮箱/电话/日期,是事实)之外,一律按【形状存在性】判 ——
    输出里还有同形状的行就算保住,措辞随便改。草稿没有结构行则 1.0。
    """
    _, fd = split_prose(draft)
    fd = [x for x in fd if x.strip() and not re.fullmatch(r'#+|[-*_\s]{3,}|```|~~~', x.strip())]
    if not fd:
        return 1.0
    ol = [ln.strip() for ln in (strip_markup(out) or '').split('\n') if ln.strip()]
    _, so = split_prose(out)
    shapes_out = set(_shape(x) for x in ol) | set(_shape(x) for x in so)
    ou = [_units(ln) for ln in ol]
    hit = 0
    for x in fd:
        k = _shape(x); ux = _units(x)
        if k == 'signature':
            if not ux or any(_contain(ux, u) >= 0.8 for u in ou):
                hit += 1
        elif k in shapes_out or (ux and any(_contain(ux, u) >= 0.5 for u in ou)):
            hit += 1
    return hit / len(fd)


def structure(draft, out):
    """结构指标。每个输出句在草稿里找最像的句子(去数字的内容 token Jaccard):
       align_sim   最像句相似度均值(0=每句都和草稿任何一句不像,1=逐句照搬/紧贴改写)
       order_agree 相邻输出句所匹配的草稿句序号严格递增的比例(b==a 记为合并,不算保序;1=顺序完全保持)
       one_to_one  【被匹配到的】草稿句里恰好被一个输出句匹配的比例(1=一句对一句;拆并越多越低;删句不影响它)
       coverage    草稿句里至少被一个输出句匹配到(Jaccard>0)的比例(删内容/退化重复 → 低)
       degenerate  输出非空却切不出任何句子(碎片/乱码/HTML)→ True,并把结构读数取最差
       n_out/n_draft 句数"""
    ds = sentences(draft); os_ = sentences(out)
    if not ds:
        return {'align_sim': 0.0, 'order_agree': 1.0, 'one_to_one': 0.0, 'coverage': 0.0, 'degenerate': False, 'n_out': len(os_), 'n_draft': 0}
    if not os_:
        return {'align_sim': 1.0, 'order_agree': 1.0, 'one_to_one': 1.0, 'coverage': 0.0, 'degenerate': bool((out or '').strip()), 'n_out': 0, 'n_draft': len(ds)}
    dt = [_align_tokens(s) for s in ds]
    idx, sims = [], []
    for s in os_:
        ot = _align_tokens(s)
        scores = [_jac(ot, x) for x in dt]
        best = max(range(len(ds)), key=lambda j: scores[j])
        sims.append(scores[best]); idx.append(best if scores[best] > 0 else -1)
    pairs = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1) if idx[i] >= 0 and idx[i + 1] >= 0 and idx[i] != idx[i + 1]]
    order = sum(1 for a, b in pairs if b > a) / len(pairs) if pairs else 1.0
    counts = {}
    for j in idx:
        if j >= 0: counts[j] = counts.get(j, 0) + 1
    matched = len(counts)
    one = sum(1 for j, c in counts.items() if c == 1) / matched if matched else 0.0
    return {'align_sim': round(sum(sims) / len(sims), 3), 'order_agree': round(order, 3), 'one_to_one': round(one, 3),
            'coverage': round(matched / len(ds), 3), 'degenerate': False, 'n_out': len(os_), 'n_draft': len(ds)}


def all_metrics(draft, out, body_only=True):
    """全部指标。body_only=True(默认):先把主题行/称呼/落款/签名块从两边剥掉,只在正文上量照抄与结构(这些行必须逐字保留,
       不该算"没改",也不该让删掉它们变成"改得深");格式要素单独给 format_recall。散文太少(<6 单位)时退回全文。"""
    d, o = draft, out
    fr = format_recall(draft, out)
    if body_only:
        db, ob = prose_lines(draft), prose_lines(out)      # 09-13:改用"只认散文"的口径,见 prose_lines 的注释
        if len(tokens(db)) >= 6 and len(tokens(ob)) >= 6:  # 散文太少(纯表格/纯代码文档)才退回全文,保住"整篇照抄"能被抓
            d, o = db, ob
    m = structure(d, o)
    vo, vd = verbatim_pair(d, o)
    m.update({'copy_5gram': round(copy_5gram(d, o), 3), 'draft_coverage': round(draft_coverage(d, o), 3),
              'verbatim_out': round(vo, 3), 'verbatim_draft': round(vd, 3), 'lcs_content': round(lcs_content(d, o), 3),
              'fact_recall': round(fact_recall(draft, out), 3), 'format_recall': round(fr, 3),
              'format_presence': round(format_presence(draft, out), 3), 'para_kept': para_kept(draft, out)})
    fc, fd_ = extractive_fragments(d, o)
    m['frag_coverage'] = round(fc, 3); m['frag_density'] = round(fd_, 2)   # 诊断用,见 extractive_fragments
    m['verbatim_sent'] = max(m['verbatim_out'], m['verbatim_draft'])
    m['copy'] = round(max(m['copy_5gram'], m['draft_coverage'], m['verbatim_out'], m['verbatim_draft']), 3)
    sk = skeleton_recall(d, o)
    m['skeleton_recall'] = None if sk is None else round(sk, 3)
    m['reuse'] = round(m['copy'] if sk is None else max(m['copy'], sk), 3)
    return m


if __name__ == '__main__':
    D = "Dr. Lin approved the request on Oct. 14. We measured density at three sites in June. No difference was found between groups. The budget is $3.5 million for 2019-2021. Please send the report by Friday."
    PAD = " There is a broader point worth keeping in view here. Everyone involved benefits when the shape of things is stated plainly. I do not think this needs much elaboration beyond what is already said."
    cases = {
        '原样照抄': D,
        '整句打乱顺序(零改写)': "Please send the report by Friday. The budget is $3.5 million for 2019-2021. No difference was found between groups. We measured density at three sites in June. Dr. Lin approved the request on Oct. 14.",
        '顺序不变·逐句换词': "Dr. Lin signed off on the request on Oct. 14. In June we sampled density at three locations. The groups did not differ. For 2019-2021 the budget comes to $3.5 million. Get the report to me by Friday.",
        '顺序不变·拆并句子': "Dr. Lin approved the request on Oct. 14, and in June we measured density at three sites. No difference was found between groups. The budget is $3.5 million for 2019-2021, so please send the report by Friday.",
        '重排+换词(真重组)': "Get the report to me by Friday. For 2019-2021 the budget comes to $3.5 million. In June we sampled density at three locations and the groups did not differ. Dr. Lin signed off on the request on Oct. 14.",
        '只保留数字的胡写': "Random words here about 2019-2021 and $3.5 million and Oct. 14 with three sites and Friday but nothing else matters at all today.",
        'S2 照抄+灌 3 段废话': D + PAD * 3,
        'S2 照抄+每句插填充词': "Dr. Lin in fact approved the request as such on Oct. 14. We in fact measured density as such at three sites in June. No difference in fact was found as such between groups. The budget in fact is $3.5 million as such for 2019-2021. Please in fact send the report as such by Friday.",
        'S9 每句加引号照抄': '"Dr. Lin approved the request on Oct. 14." "We measured density at three sites in June." "No difference was found between groups." "The budget is $3.5 million for 2019-2021." "Please send the report by Friday."',
        'S9 句号换 which is to say': "Dr. Lin approved the request on Oct. 14, which is to say we measured density at three sites in June, which is to say no difference was found between groups, which is to say the budget is $3.5 million for 2019-2021, which is to say please send the report by Friday",
        'S9 <p> 标签照抄': "<p>Dr. Lin approved the request on Oct. 14.</p><p>We measured density at three sites in June.</p><p>No difference was found between groups.</p><p>The budget is $3.5 million for 2019-2021.</p><p>Please send the report by Friday.</p>",
        'S10 同一句重复 5 次': "Dr. Lin approved the request on Oct. 14. " * 5,
        'S10 只写一句无关': "Nothing to report this week from anyone.",
        'S4 删一半句子': "Dr. Lin approved the request on Oct. 14. We measured density at three sites in June.",
        'S5 否定翻转照抄': D.replace('No difference', 'A difference'),
    }
    print('草稿分句:', sentences(D))
    print(f"{'情形':22s} copy  5gram dcov  vOut  vDrf  lcs   align order 1to1  cover facts 句数")
    for k, o in cases.items():
        m = all_metrics(D, o); print(f"{k:22s} {m['copy']:.2f}  {m['copy_5gram']:.2f}  {m['draft_coverage']:.2f}  {m['verbatim_out']:.2f}  {m['verbatim_draft']:.2f}  {m['lcs_content']:.2f}  {m['align_sim']:.2f}  {m['order_agree']:.2f}  {m['one_to_one']:.2f}  {m['coverage']:.2f}  {m['fact_recall']:.2f}  {m['n_out']}/{m['n_draft']}{' DEGEN' if m['degenerate'] else ''}")
    Z = "李老师您好,关于上周说的数据访问申请,还有一点补充说明。首先,这次分析仅需2019-2021年时间序列子集。其次,审批大概两到三周。祝好,小王"
    Z2 = "祝好,小王。其次,审批大概两到三周。首先,这次分析仅需2019-2021年时间序列子集。关于上周说的数据访问申请,还有一点补充说明。李老师您好。"
    Z3 = "李老师好,上周提到的数据访问申请我再补充一下:分析只用2019到2021年那段时间序列,审批大概要两三周。谢谢您,小王"
    PRE = "Below is the revised version of the text you sent over earlier today. I have reworked the argument throughout and tightened the prose considerably, taking care to preserve every claim and every figure from the original draft exactly as it was given to me. Please read the new version carefully and let me know whether the framing now works for the purposes you described, or whether you would like me to take another pass at it. "
    for k, o in (('中文原样', Z), ('中文整句打乱', Z2), ('中文真改写', Z3), ('S1 英文前言+中文原样一行', PRE + Z.replace('\n', ''))):
        m = all_metrics(Z, o); print(f"{k:22s} {m['copy']:.2f}  {m['copy_5gram']:.2f}  {m['draft_coverage']:.2f}  {m['verbatim_out']:.2f}  {m['verbatim_draft']:.2f}  {m['lcs_content']:.2f}  {m['align_sim']:.2f}  {m['order_agree']:.2f}  {m['one_to_one']:.2f}  {m['coverage']:.2f}  {m['fact_recall']:.2f}  {m['n_out']}/{m['n_draft']}")
    W = "Dr. Lin approved the request on October\n14 and we measured density at three\nsites in June. No difference was found\nbetween groups."
    print('S14 硬换行分句:', sentences(W))
    print('S9 引号分句:', sentences('He said "Fine." Then he left. Really?! Yes.'))
    print('缩写分句:', sentences('It was approved by the U.S. Dept. of Energy in Oct. 2019. See Fig. 3 and e.g. Table 2. Done.'))
