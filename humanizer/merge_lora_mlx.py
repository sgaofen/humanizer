#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 PEFT 格式的 LoRA 合并进基座权重,产出 MLX 能直接加载的模型。

为什么要这个(2026-08-28):集群 free-gpu 排着 251 个作业,评测要等到明天下午。
但这是 4B 模型,Mac 本地就能跑 —— 只是本地只有 MLX,没有 torch/peft,
而 MLX 不认 PEFT 的适配器格式(报 'SimpleNamespace' has no attribute 'num_layers')。
所以自己算:W' = W + (alpha/r) · B @ A,存成普通权重,MLX 就能当普通模型加载。

支持叠加多个适配器(先续训后 SFT),顺序就是训练顺序。
"""
import argparse, glob, json, os, shutil, sys
import mlx.core as mx

def load_shards(d):
    w = {}
    for f in sorted(glob.glob(f'{d}/*.safetensors')):
        w.update(mx.load(f))
    return w

def merge(base_w, ad_dir):
    cfg = json.load(open(f'{ad_dir}/adapter_config.json'))
    scale = cfg['lora_alpha'] / cfg['r']
    aw = mx.load(f'{ad_dir}/adapter_model.safetensors')
    # PEFT 的键名: base_model.model.<路径>.lora_A.weight
    pairs = {}
    for k in aw:
        if '.lora_A.' in k or '.lora_B.' in k:
            stem = k.split('.lora_')[0].replace('base_model.model.', '')
            pairs.setdefault(stem, {})['A' if '.lora_A.' in k else 'B'] = aw[k]
    done = miss = 0
    for stem, ab in pairs.items():
        if 'A' not in ab or 'B' not in ab:
            continue
        # 【2026-08-28】Qwen3.5 是混合架构,权重路径是 model.language_model.layers.N...,
        # 而 PEFT 训练时按 model.layers.N... 记的 —— 差一个 language_model 前缀。
        # (顺带查清:36 层里只有 8 层是标准 self_attn,其余 28 层是 Mamba 式 linear_attn。)
        wk = None
        for cand in (f'{stem}.weight',
                     'model.language_model.' + stem.replace('model.', '', 1) + '.weight'):
            if cand in base_w:
                wk = cand
                break
        if wk is None:
            miss += 1
            continue
        # delta = scale * B @ A ; B:[out,r]  A:[r,in]  -> [out,in]
        delta = (ab['B'].astype(mx.float32) @ ab['A'].astype(mx.float32)) * scale
        base_w[wk] = (base_w[wk].astype(mx.float32) + delta).astype(base_w[wk].dtype)
        done += 1
    print(f'  合并 {ad_dir}: 改了 {done} 个权重,配不上 {miss} 个 (alpha/r={scale})', flush=True)
    if done == 0:
        sys.exit('★ 一个权重都没合上,键名对不上,停')
    if miss:
        # 只挡"全军覆没"是不够的。混合架构下最可能出的事是【合上了一部分】:
        # 8 层 self_attn 对上了、28 层 linear_attn 没对上,脚本照样打印成功退出,
        # 拿到手的是一个"训了但只生效了一小半"的模型 —— 而且看不出来。
        sys.exit(f'★ {miss} 个权重键名配不上(只合上了 {done} 个)。'
                 f'部分合并比不合并更危险,停。')
    return base_w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=None)
    ap.add_argument('--adapters', default='', help='逗号分隔,按训练顺序')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    base = a.base or glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/*'))[0]
    w = load_shards(base)
    print(f'基座 {len(w)} 个张量', flush=True)
    spec_src = None
    for ad in [x for x in a.adapters.split(',') if x]:
        w = merge(w, ad)
        cand = os.path.join(ad, 'prompt_format.json')
        if os.path.exists(cand):
            spec_src = cand
    os.makedirs(a.out, exist_ok=True)
    # 【2026-08-30 审计】"格式跟着权重走"此前只是碰巧成立 —— spec 是人工拷进
    # 合并产物目录的,下次换目录必忘。现在从 adapter 目录自动带过去,没有就硬停:
    # 一个不带格式说明的模型目录等于一颗地雷。
    if spec_src:
        import shutil
        shutil.copy(spec_src, os.path.join(a.out, 'prompt_format.json'))
        print(f'prompt_format.json <- {spec_src}', flush=True)
    else:
        sys.exit('★ 所有 adapter 目录都没有 prompt_format.json —— 拒绝产出无格式说明的模型')
    for f in glob.glob(f'{base}/*.json') + glob.glob(f'{base}/*.txt') + glob.glob(f'{base}/*.jinja'):
        if 'index' not in os.path.basename(f):
            shutil.copy(f, a.out)
    mx.save_safetensors(f'{a.out}/model.safetensors', w)
    print(f'存到 {a.out}', flush=True)
    print('MERGE_MLX_DONE')

if __name__ == '__main__':
    main()
