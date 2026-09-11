#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三阶段:DPO —— 把守卫的"重采一次"搬进权重里。

用户定的目标是"第一发就生成正确的",不是"生成很多次然后把不好的砍掉"。
SFT 教的是"能写对",DPO 教的是"别写错"——两者需要的信号不一样:
SFT 只看正例,模型不知道哪些写法是它自己爱犯的错;DPO 给成对的好坏,
把概率质量从坏那边搬走。

在 SFT 适配器之上继续训(增量检查点策略,用户:"每次从头训还有点太浪费了"):
  --sft_adapter 必填,先合并进基座,再训一个新的 DPO LoRA。
参考模型不单独加载(4B×2 会撑爆),用 peft 的适配器禁用做隐式参考:
trl 在 model 带 peft 且 ref_model=None 时会自动走这条路。
"""
import os, argparse, json, os, random, sys
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from promptfmt import fingerprint, write_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='google/gemma-4-E4B')
    ap.add_argument('--sft_adapter', required=True, help='vNext SFT 产物,先合并再训')
    ap.add_argument('--prefs', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--resume', default='', help="'auto' = 从 --out 里最新的 checkpoint-* 续训(09-10:被抢占后不再从头来)")
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--bs', type=int, default=1)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-6)     # DPO 要比 SFT 小一到两个量级
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--maxlen', type=int, default=2560)
    a = ap.parse_args()

    print(f'训练格式指纹 = {fingerprint()}', flush=True)
    if not os.path.exists(a.sft_adapter):
        sys.exit(f'★ sft_adapter 不存在:{a.sft_adapter} —— 拒绝在裸基座上做 DPO')

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(a.prefs, encoding='utf-8') if l.strip()]
    for r in rows:
        assert r.get('prompt') and r.get('chosen') and r.get('rejected'), '偏好行缺字段'
    random.seed(11)
    random.shuffle(rows)
    n_ev = min(300, max(40, len(rows) // 20))
    ev, trn = rows[:n_ev], rows[n_ev:]
    print(f'偏好对 训练 {len(trn)} / 验证 {len(ev)}', flush=True)

    ws = int(os.environ.get('WORLD_SIZE', '1'))
    dev = {'': int(os.environ.get('LOCAL_RANK', '0'))} if ws > 1 else 'cuda'
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16,
                                                 device_map=dev, attn_implementation='sdpa')
    print(f'合并 SFT 适配器:{a.sft_adapter}', flush=True)
    model = PeftModel.from_pretrained(model, a.sft_adapter)
    model = model.merge_and_unload()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()

    steps_per_epoch = max(1, len(trn) // (a.bs * a.accum * max(1, ws)))
    ev_steps = max(20, steps_per_epoch // 10)
    cfg = DPOConfig(
        output_dir=a.out,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        num_train_epochs=a.epochs,
        beta=a.beta,
        max_length=a.maxlen,
        bf16=True,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=ev_steps,
        save_steps=ev_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        # trl 0.29.1 的 DPOConfig 没有 warmup_ratio,只认 warmup_steps
        warmup_steps=max(10, int(steps_per_epoch * a.epochs * 0.1)),
        lr_scheduler_type='cosine',
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # peft 路径:禁用适配器即参考模型
        args=cfg,
        train_dataset=Dataset.from_list(trn),
        eval_dataset=Dataset.from_list(ev),
        processing_class=tok,
        peft_config=LoraConfig(
            r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05,
            target_modules=(r'^(?!.*(vision|audio)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$' if 'gemma' in a.base.lower() else ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']),
            task_type='CAUSAL_LM'),
    )
    resume = None
    if a.resume:
        import glob as _g
        cks = sorted(_g.glob(os.path.join(a.out, 'checkpoint-*')), key=lambda d: int(d.rsplit('-', 1)[-1]))
        resume = cks[-1] if a.resume == 'auto' and cks else (a.resume if a.resume != 'auto' else None)
        print('续训自', resume, flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    write_spec(a.out, style='legacy')   # 权重目录自带格式规格,推理端不会漂
    print('DPO_DONE', flush=True)


if __name__ == '__main__':
    main()
