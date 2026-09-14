#!/bin/bash
# 把一个 RL checkpoint 合并成 MLX 模型并换上 8104 生产臂(2026-09-13)。用法: deploy_gemma_arm.sh <adapter_dir> <out_name> "<label>"
# 步骤:merge_lora_mlx(底座+scale_E20g+dpo_E20gd4+RL) → 删后 18 层共用 KV 的 54 个张量(scripts/gemma_e4b_drop_keys.json) → 换 webserver.py 路径 → 重启 8104。
set -e; cd ~/未命名文件夹/humantune; PY=~/.venvs/mlx-dspark/bin/python
AD=$1; OUT=models/$2; LABEL=$3; BASE=$(ls -d ~/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/*/ | head -1)
[ -s $AD/adapter_model.safetensors ] && [ -s $AD/adapter_config.json ] && [ -s $AD/prompt_format.json ] || { echo "★ $AD 缺 adapter/config/prompt_format"; exit 1; }
[ -e $OUT ] && { echo "★ $OUT 已存在,不覆盖"; exit 1; }
$PY scripts/merge_lora_mlx.py --base $BASE --adapters adapters/scale_E20g_adapter,adapters/dpo_E20gd4,$AD --out $OUT | tail -3
$PY - "$OUT" <<'PY'
import sys,json,mlx.core as mx; out=sys.argv[1]
drop=set(json.load(open('scripts/gemma_e4b_drop_keys.json'))); w=mx.load(f'{out}/model.safetensors')
n0=len(w); w={k:v for k,v in w.items() if k not in drop}; print(f'张量 {n0} → {len(w)}(删 {n0-len(w)},应为 54)')
assert n0-len(w)==54, '删的张量数不对'
import os; mx.save_safetensors(f'{out}/model.tmp.safetensors', w); mx.eval(); os.replace(f'{out}/model.tmp.safetensors', f'{out}/model.safetensors'); print('DROP_DONE')   # mx.load 是惰性读,不能原地覆盖(00:00 把文件写成 0 字节)
PY
$PY -c "
from mlx_lm import load; m,t=load('$OUT'); print('MLX_LOAD_OK')"
echo "MERGE_OK $OUT"
