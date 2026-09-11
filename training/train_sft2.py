#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二阶段:配对 SFT —— 在续训过的基座上,只教【任务接口】。

两阶段拆分的理由(2026-08-27 用户指出的结构性错误):
  文风要海量数据(第一阶段 200M token 人类文本),任务格式不要。
  之前用 429 条配对同时干两件事,两臂验证损失都在 0.6 轮见底就过拟合。
  这一阶段只教:看到 draft + "### Rewritten:" 该输出什么。所以几千条就够。

做法:先把第一阶段的 LoRA 合并进基座权重,再训一个新的任务 LoRA。
避免适配器叠加的复杂性,也让"文风"变成模型的底子而不是可插拔的东西。

损失只算 completion 那一段(prompt 部分遮掉),这点和第一阶段相反。
"""
import json, os, sys, argparse, random
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                          Trainer, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, PeftModel

# 【2026-08-29】指令融进权重:训练时不再拼 INSTR,只留 "### Rewritten:" 当触发器。
# 格式定义在 promptfmt.py —— 训练和推理【唯一】的事实源,物理上不可能漂移。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from promptfmt import build_prompt, fingerprint, write_spec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='google/gemma-4-E4B')
    ap.add_argument('--cpt_adapter', default='')      # 第一阶段产物,空则直接用基座
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rank', type=int, default=32)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=float, default=3.0)
    ap.add_argument('--maxlen', type=int, default=3072)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    pairs = [json.loads(l) for l in open(a.pairs, encoding='utf-8')]
    random.seed(11); random.shuffle(pairs)
    print(f'配对 {len(pairs)} 条', flush=True)

    print(f'格式指纹 {fingerprint()} (推理端必须一致)', flush=True)

    def build(p):
        prompt = build_prompt(p['ai'])
        pi = tok(prompt, add_special_tokens=False)['input_ids']
        ci = tok(p['human'], add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
        ids = pi + ci
        if len(ids) > a.maxlen: return None
        # prompt 段遮掉,只在人类原文那段算损失
        labels = [-100] * len(pi) + ci
        return {'input_ids': ids, 'labels': labels}

    built = [(p, b) for p, b in ((p, build(p)) for p in pairs) if b]
    ex = [b for _, b in built]
    print(f'可用 {len(ex)} 条(超长丢弃 {len(pairs)-len(ex)})', flush=True)
    # 【2026-08-29】原来是 shuffle 之后按下标切:ev, trn = ex[:n_ev], ex[n_ev:]。
    # 同一段人类原文如果出现在多条配对里(不同 provider / 不同 AI 风格),
    # 就会一半落进训练、一半落进验证 —— 验证损失被污染,早停就停错地方。
    # 改成【按人类原文哈希分组】切:同源的必定在同一侧。
    import hashlib
    # 【2026-09-01】哈希前先归一化(压空白 + 小写)。上一版直接哈希原文,
    # 只差空格/大小写的"归一化孪生"会得到不同哈希被拆到两侧 —— A/B 两臂
    # eval_loss 不可互比就是栽在这里(A 泄漏 8.7% / B 3.8%)。
    import re as _re
    # 【顺序要紧】必须【先归一化再截断】。写成 t[:600] 再 re.sub 是错的:
    # 两段只差空白量的原文,原始串第 600 字符切在不同的逻辑位置,归一化后内容
    # 仍然不同 → 哈希不同 → 同源被拆到训练/验证两侧。实测(pairs_vnext 9 万条):
    # 先截断 → 验证 658 条里 30 条(4.56%)在训练侧有逐字孪生;
    # 先归一化 → 验证 706 条,跨侧孪生 0 条。
    _nk = lambda t: _re.sub(r'\s+', ' ', t).strip().lower()[:600]
    grp = [hashlib.md5(_nk(p['human']).encode()).hexdigest() for p, _ in built]
    # 上限很要紧:len(ex)//20 在 8 万条上是 4000 条验证集,评测本身比训练还贵,
    # 500 条对 eval_loss 已经足够稳。
    n_ev = min(500, max(40, len(ex) // 20))
    # 【2026-08-30 审计发现的真 bug】上一版 seen 只记"已进 ev 的组",同一 human 的
    # 第 2 条配对走 else 掉进 trn —— 恰好把同源样本拆到两侧,和防泄漏的意图正相反。
    # 正确做法:先按组定去向,同组必然同侧。
    ev_groups = set()
    for g in grp:
        if len(ev_groups) * (len(ex) // max(1, len(set(grp)))) >= n_ev:
            break
        ev_groups.add(g)
    ev = [b for g, b in zip(grp, ex) if g in ev_groups]
    trn = [b for g, b in zip(grp, ex) if g not in ev_groups]
    # 【2026-09-01】原来这里是 `shared = ev_groups & {不在 ev_groups 的组}` ——
    # 按定义恒为空,断言结构上不可能触发,只提供虚假信心。换成【真检查】:
    # 用【全文】归一化(不截断)比对两侧的人类原文,这才抓得到"分组键截断后
    # 相同、全文其实不同"或反过来的漏网。
    _full = lambda t: _re.sub(r'\s+', ' ', t).strip().lower()
    _ev_full = {_full(p['human']) for (p, b), g in zip(built, grp) if g in ev_groups}
    _tr_full = {_full(p['human']) for (p, b), g in zip(built, grp) if g not in ev_groups}
    shared = _ev_full & _tr_full
    print(f'验证 {len(ev)} 条 / {len(ev_groups)} 组;训练 {len(trn)} 条;'
          f'两侧同文本泄漏 {len(shared)} 条(必须为 0)', flush=True)
    assert not shared, f'同源泄漏 {len(shared)} 条'
    assert len(ev) >= 40, f'验证集只有 {len(ev)} 条'

    print(f'训练 {len(trn)} / 验证 {len(ev)}', flush=True)

    class DS(torch.utils.data.Dataset):
        def __init__(s, d): s.d = d
        def __len__(s): return len(s.d)
        def __getitem__(s, i): return s.d[i]

    def collate(batch):
        m = max(len(b['input_ids']) for b in batch)
        pad = tok.pad_token_id
        return {
            'input_ids': torch.tensor([b['input_ids'] + [pad]*(m-len(b['input_ids'])) for b in batch]),
            'attention_mask': torch.tensor([[1]*len(b['input_ids']) + [0]*(m-len(b['input_ids'])) for b in batch]),
            'labels': torch.tensor([b['labels'] + [-100]*(m-len(b['labels'])) for b in batch]),
        }

    # 【2026-08-30 多卡】torchrun 启动时(WORLD_SIZE>1)走 DDP:
    # 每个进程各占一张卡,device_map 必须指到本进程的卡,不能写死 'cuda'
    # (写死会四个进程全挤到 GPU0)。单卡路径不变。
    import os as _os
    _ws = int(_os.environ.get('WORLD_SIZE', '1'))
    _dev = {'': int(_os.environ.get('LOCAL_RANK', '0'))} if _ws > 1 else 'cuda'
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                 device_map=_dev, attn_implementation='sdpa')
    if a.cpt_adapter:
        # 原来是 `if a.cpt_adapter and os.path.exists(...)` —— 路径打错一个字,
        # 整个合并就被静默跳过,在裸基座上训完 8 万条,loss 照常降、照常打印完成,
        # 只有最后人味评测才看得出不对。姊妹脚本 sft_s2/s6.sbatch 对同类问题都是
        # "找不到 adapter 就拒绝开训",这里跟上。
        if not os.path.exists(a.cpt_adapter):
            sys.exit(f'★ cpt_adapter 路径不存在: {a.cpt_adapter} —— 拒绝在裸基座上开训')
        print(f'合并第一阶段文风适配器: {a.cpt_adapter}', flush=True)
        model = PeftModel.from_pretrained(model, a.cpt_adapter)
        model = model.merge_and_unload()
    else:
        print('未提供 cpt_adapter,直接在基座上训练', flush=True)
    model.config.use_cache = False
    # use_reentrant=False:DDP + PEFT + 梯度检查点的标准组合,
    # reentrant 版会报 unused parameters / 不触发 hook。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.rank*2, lora_dropout=0.05,
        target_modules=(r'^(?!.*(vision|audio)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$' if 'gemma' in a.base.lower() else ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']),   # Gemma 4:只挂语言塔,跳过 vision/audio 的 Gemma4ClippableLinear
        task_type='CAUSAL_LM'))
    model.print_trainable_parameters()

    # eval/save 频率随数据量走:25 步是为几千条配对定的,8 万条时一个 epoch 有
    # 5000 步,每 25 步存一次检查点等于存 200 次。取"每 epoch 约 20 次"。
    spe = max(1, len(trn) // max(1, a.bs * a.accum * _ws))
    EVERY = max(25, (spe // 20 // 25) * 25 or 25)
    print(f'每 epoch {spe} 步;每 {EVERY} 步评估/存档一次', flush=True)

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.accum, learning_rate=a.lr,
        num_train_epochs=a.epochs, warmup_steps=30, lr_scheduler_type='cosine',
        logging_steps=10, eval_strategy='steps', eval_steps=EVERY,
        save_strategy='steps', save_steps=EVERY, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, bf16=True, report_to=[],
        ddp_find_unused_parameters=False,
        per_device_eval_batch_size=a.bs,
    )
    # 【2026-08-29 复盘上一次训练】patience 原来是 3。上次 3,580 步的计划停在 1,850 步
    # (epoch 1.03),而 eval_loss 到 1,775 步为止【一直在单调下降】,从没真的停过 ——
    # 只是后半程斜率降到每 100 步 0.0015,每次评估之间(25 步)该降 0.0004,
    # 比 1,506 条验证集的噪声还小,于是"连续 3 次没破纪录"变成了必然事件。
    # 现在评估间隔已随数据量放大(8 万条时 225 步一次),patience 再放到 5,
    # 等于要连续 1,125 步没有进步才停 —— 这才算得上一个信号。
    tr = Trainer(model=model, args=args, train_dataset=DS(trn), eval_dataset=DS(ev),
                 data_collator=collate,
                 callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])
    # 【2026-08-30 审计】free-gpu32 是可抢占分区,sbatch 带 --requeue ——
    # 但 transformers 不会自动续训(查过源码),被抢占后从零重跑等于白烧半天。
    # 输出目录里有 checkpoint 就接着训。
    import glob as _g
    cks = sorted(_g.glob(f'{a.out}/checkpoint-*'),
                 key=lambda x: int(x.rsplit('-', 1)[1]))
    if cks:
        print(f'发现断点 {cks[-1]},续训', flush=True)
        tr.train(resume_from_checkpoint=cks[-1])
    else:
        tr.train()
    # 【2026-08-30 复查抓到的必炸 bug】torchrun 四个进程都会跑到这里,
    # 对同一个 adapter 目录并发写 safetensors —— Trainer 内部的检查点有 rank0
    # 门控,这段手写收尾没有。只让 rank0 写,其余进程等它写完。
    if int(os.environ.get('RANK', '0')) == 0:
        model.save_pretrained(f'{a.out}/adapter'); tok.save_pretrained(f'{a.out}/adapter')
    # 把格式说明写进产物目录 —— 推理端据此自动选对格式,别人下载也自带说明
    # 2026-08-29:指令最终决定【保留】不融合,所以这里是 legacy 不是 fused。
    # 上一版写死 fused,导致产物的 prompt_format.json 说明和实际训练格式对不上。
        write_spec(f'{a.out}/adapter', style='legacy')
        print('SFT2_DONE', flush=True)
    if _ws > 1:
        torch.distributed.barrier()

if __name__ == '__main__':
    main()
