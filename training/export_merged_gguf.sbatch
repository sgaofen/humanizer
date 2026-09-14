#!/bin/bash
#SBATCH -J export_r7
#SBATCH -p free-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH -c 8
#SBATCH --mem 80G
#SBATCH -t 3:00:00
#SBATCH -o /pub/stephjy1/humantune/logs/%x_%j.out
# 09-14 发布导出:底座 + scale_E20g + dpo_E20gd4 + grpo_R7/adapter 逐层 merge_and_unload → transformers bf16(4 分片)→ GGUF bf16/Q8_0/Q6_K。
# 配方与 09-10 的 E20gd4 导出一致(/dfs7/adl/stephjy1/humanizer 下 convert.log / quant_*.log 可对照)。
set -e
export HF_HOME=/pub/stephjy1/hf_cache TOKENIZERS_PARALLELISM=false
HT=/pub/stephjy1/humantune; W=/dfs7/adl/stephjy1/humanizer; PY=/pub/stephjy1/conda_envs/humantune/bin/python
OUT=$W/hf_bf16_r7; G=$W/gguf_r7; mkdir -p $G
$PY - <<PY
import torch, shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
HT='$HT'; base='google/gemma-4-E4B'
tok=AutoTokenizer.from_pretrained(base)
m=AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map='cuda')
for pa in [f'{HT}/runs/scale_E20g/adapter', f'{HT}/runs/dpo_E20gd4', f'{HT}/runs/grpo_R7/adapter']:
    print('merge', pa, flush=True); m=PeftModel.from_pretrained(m, pa).merge_and_unload()
print(type(m).__name__, flush=True)
m.save_pretrained('$OUT', safe_serialization=True, max_shard_size='5GB'); tok.save_pretrained('$OUT')
shutil.copy(f'{HT}/runs/grpo_R7/adapter/prompt_format.json', '$OUT/prompt_format.json'); print('HF_SAVED', flush=True)
PY
cd $W
$PY llama.cpp/convert_hf_to_gguf.py hf_bf16_r7 --outfile $G/humanizer-gemma-4-e4b-bf16.gguf --outtype bf16 > convert_r7.log 2>&1 && echo CONVERT_OK
for q in Q8_0 Q6_K; do llama.cpp/build/bin/llama-quantize $G/humanizer-gemma-4-e4b-bf16.gguf $G/humanizer-gemma-4-e4b-$q.gguf $q > quant_r7_$q.log 2>&1 && echo QUANT_OK $q; done
cp $OUT/prompt_format.json $G/
ls -l $OUT $G
echo EXPORT_R7_DONE
