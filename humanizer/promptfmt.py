#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""改写任务的输入格式 —— 训练和推理【唯一】的事实源。

为什么要有这个文件(2026-08-29):
  基座模型只认它训练时见过的那个格式,差一个换行就崩。
  今天已经因为"提示词写在两处"做过一次逐字校验去防漂移 —— 那是补救,不是解决。
  现在两边都从这里 import,物理上不可能不一致。

指令为什么去掉(用户要求"把 prompt 融进权重",实测支持):
  拿现有模型对比三种输入,同一篇稿子各抽两次:
    完整指令 → 105/108 词,照抄 0.010/0.029
    只留分隔符 → 110/97 词,照抄 0.019/0.011   ← 没有差别
    指令换成废话 → 一版正常,一版只有 49 词且跑题   ← 说明它还在读,但不靠它
  也就是说 "### Rewritten:" 才是真正的开关,那 70 个 token 的指令是摆设。
  去掉之后:每次调用省 70 token,而且"提示词和训练不一致"这一整类故障消失。

分隔符为什么是这个(有数据):
  在 13,427 条真实样本里查过 —— "### Rewritten:" 完整串出现 0 次,零撞车。
  裸 "###" 在 AI 稿里出现 1.1-2.4%,在人类原文里 0.1%。
  所以:用完整串当触发器安全;但【停止条件必须靠 EOS】,
  拿 "###" 当停止符会让那 0.1% 的输出被截断。
"""
import hashlib

SEP = '\n\n### Rewritten:\n\n'


INSTR = """Rewrite the text below so it reads like a person wrote it, not a language model.

Reorganize it as you see fit. Vary sentence length on purpose. Cut hedging,
throat-clearing, and any sentence that only announces what comes next.
Prefer the concrete word over the abstract one. It is fine to sound uneven.

Every fact, number, unit, date, name and quotation must survive unchanged."""


def build_prompt(draft):
    """训练和推理都调这一个函数。绝不要在别处手拼。

    【2026-08-29 决定保留指令,不做融合】
    融合的收益是每次省约 70 token;代价是那句
    "Every fact, number, unit, date, name and quotation must survive unchanged"
    模型再也看不见。我实测过"有指令 vs 只留分隔符"输出没差别 ——
    但那测的是【训练时见过指令】的旧模型,"训练时就没见过"是另一回事,没验证过。
    用户判断:六行字复制粘贴不费事,不值得拿一个没验证的风险换这点 token。同意。
    """
    return INSTR + '\n\n' + draft.strip() + SEP


def fingerprint():
    """格式指纹。训练时记下来,推理时比对,不一致就该报警。"""
    return hashlib.sha256(build_prompt('X').encode()).hexdigest()[:16]


# ── 旧格式(2026-08-29 之前训的模型认这个)────────────────────────────
LEGACY_INSTR = """Rewrite the text below so it reads like a person wrote it, not a language model.

Reorganize it as you see fit. Vary sentence length on purpose. Cut hedging,
throat-clearing, and any sentence that only announces what comes next.
Prefer the concrete word over the abstract one. It is fine to sound uneven.

Every fact, number, unit, date, name and quotation must survive unchanged."""


def legacy_prompt(draft):
    """和 build_prompt 现在是同一个格式(2026-08-29 决定不融合)。留着兼容老的 spec 文件。"""
    return LEGACY_INSTR + '\n\n' + draft.strip() + SEP


SPEC_FILE = 'prompt_format.json'


def write_spec(model_dir, style='fused'):
    """训练完把格式写进模型目录 —— 让暗号跟着权重走。
    这样换模型不用改代码,别人下载了权重也自带格式说明。"""
    import json, os
    os.makedirs(model_dir, exist_ok=True)
    spec = {'style': style, 'sep': SEP, 'fingerprint': fingerprint(),
            'instr': None if style == 'fused' else LEGACY_INSTR,
            'note': ('推理时只需 draft + sep;指令已训进权重'
                     if style == 'fused' else '推理时需 instr + draft + sep')}
    with open(os.path.join(model_dir, SPEC_FILE), 'w', encoding='utf-8') as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)
    return spec


def load_for_model(model_dir):
    """按模型目录里的说明选格式。没有说明就当旧格式(2026-08-29 前的模型)。
    返回 (拼接函数, style)。"""
    import json, os
    p = os.path.join(model_dir or '', SPEC_FILE)
    if os.path.exists(p):
        try:
            spec = json.load(open(p, encoding='utf-8'))
            if spec.get('style') == 'fused':
                return build_prompt, 'fused'
            return legacy_prompt, 'legacy'
        except Exception:
            pass
    return legacy_prompt, 'legacy'


if __name__ == '__main__':
    print('SEP      =', repr(SEP))
    print('指纹     =', fingerprint())
    print('示例:')
    print(repr(build_prompt('Some AI draft text.')))
