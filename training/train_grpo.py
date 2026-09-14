#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""针对事实错误的在线 GRPO(2026-09-11,用户批准的 RL 设计):
  策略 = Gemma-4-E4B + SFT(scale_E20g)+ DPO(dpo_E20gd4)合并后的权重,再挂一个新 LoRA 训;参考模型 = 合并权重(LoRA 关掉)。
  每个草稿采 G 发,奖励 = rl_reward.reward(草稿, 输出, 事实清单):−(改/丢/编造事实) −反转 −格式丢 −落款丢 −0.2·轻微;
  照抄>.35 / 语言切换 / 退化 → 底分。没有风格项,没有检测器项。判定由计算节点直接调 GLM-5.3(节点能出网,已实测)。
用法(sbatch 里):python train_grpo.py --ledger data/rl_ledger.jsonl --out runs/grpo_R1 --max_steps 300 --num_generations 8
"""
import argparse, json, os, sys, time, random, threading
from concurrent.futures import ThreadPoolExecutor
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from promptfmt import load_for_model
import rl_reward
from textmetrics import is_cjk

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='google/gemma-4-E4B'); ap.add_argument('--sft_adapter', default='/pub/stephjy1/humantune/runs/scale_E20g/adapter'); ap.add_argument('--dpo_adapter', default='/pub/stephjy1/humantune/runs/dpo_E20gd4')
ap.add_argument('--ledger', required=True); ap.add_argument('--out', required=True); ap.add_argument('--fmt', default='')
ap.add_argument('--num_generations', type=int, default=8); ap.add_argument('--prompts_per_step', type=int, default=2); ap.add_argument('--max_steps', type=int, default=300)
ap.add_argument('--lr', type=float, default=5e-6); ap.add_argument('--beta', type=float, default=0.04); ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--max_prompt', type=int, default=1536); ap.add_argument('--max_completion', type=int, default=1024); ap.add_argument('--temp', type=float, default=0.85)
ap.add_argument('--n_prompts', type=int, default=0, help='只用前 N 篇草稿(0=全部);按 seed 洗牌'); ap.add_argument('--seed', type=int, default=11)
ap.add_argument('--judge_workers', type=int, default=16); ap.add_argument('--save_steps', type=int, default=25); ap.add_argument('--resume', default='')
ap.add_argument('--gen_batch', type=int, default=0, help='一次生成多少发(G 的倍数;大批量吃满 GPU,没有 vLLM 时这是唯一的提速手段);0=每步只生成本步的')
ap.add_argument('--micro_bs', type=int, default=0, help='每次前向反向的样本数(0=等于 num_generations)。80G 卡放得下 8,48G 卡要降到 4 否则 OOM;有效批由 prompts_per_step 补回来')
ap.add_argument('--span_beta', type=float, default=2.0, help='句级 credit 集中度:0=关闭(整篇均摊,R1 的做法);2=承载错误的句子权重约为其他句的 3 倍。总量守恒,不改变整体尺度')
ap.add_argument('--nll_floor_w', type=float, default=0.0, help='意外度地板项权重(0=关)。09-12 查明:RL 把输出推向"参考模型早就会写的那些词",'
                'NLL 单调下滑、过检率同步下滑。配对测量证明真人并不比 AI 更意外,所以不能写成"越意外越好"(那条路通向错字病句),'
                '只做单边地板:比出发点更好猜就扣分,比出发点更意外不给分——收益封顶,乱写拿不到好处')
ap.add_argument('--nll_warm', type=int, default=20, help='前多少步只记录不惩罚,用来定地板(这几步策略≈参考模型)')
ap.add_argument('--save_limit', type=int, default=14, help='保留多少个 checkpoint。原来写死 3,只剩末尾三个;09-12 查明奖励在 175 步就见顶而策略熵一路跌到底,必须留整条梯子才能事后挑步数')
ap.add_argument('--top_entropy_q', type=float, default=1.0, help='只在熵最高的前 q 比例 token 上更新(TRL 的 top_entropy_quantile)。1.0=全部(默认);0.2=只更新高熵 token,用来保住策略熵')
ap.add_argument('--save_merged', action='store_true', help='训完把 RL LoRA 合进(已合并 SFT+DPO 的)权重,存 <out>/merged 供 gen_cases 直接当 BASE 用')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# ---- 数据:草稿 + 事实清单 → 提示(与训练模板逐字一致) ----
build_fn, style = load_for_model(a.fmt or a.dpo_adapter)
print('prompt 格式', style, flush=True)
rows = [json.loads(l) for l in open(a.ledger)]
rows = [r for r in rows if 2 <= len(r.get('facts', [])) <= 30 and len(r['draft']) <= 6000]
random.Random(a.seed).shuffle(rows)
if a.n_prompts: rows = rows[:a.n_prompts]
ds = Dataset.from_list([{'prompt': build_fn(r['draft']), 'key': r['key'], 'draft': r['draft'], 'facts_json': json.dumps(r['facts'], ensure_ascii=False)} for r in rows])
print('提示数', len(ds), flush=True)

# ---- 模型:base + SFT + DPO 合并 → 新 LoRA ----
tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, a.sft_adapter).merge_and_unload()
model = PeftModel.from_pretrained(model, a.dpo_adapter).merge_and_unload()
print('SFT+DPO 已合并进权重', flush=True)
lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
                  target_modules=(r'^(?!.*(vision|audio)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$' if 'gemma' in a.base.lower() else ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))   # Gemma 4:只挂语言塔,跳过 vision/audio 的 Gemma4ClippableLinear(与 SFT/DPO 一致)

# ---- 奖励:节点内直接调 GLM,线程并行 ----
LOG = open(os.path.join(a.out, 'reward_log.jsonl'), 'a'); _lk = threading.Lock(); STEP = [0]
SPANS = {}     # completion 文本 → {承载错误的句子前八词: 扣分};句级 credit 用
def fact_reward(prompts, completions, key=None, draft=None, facts_json=None, **kw):
    jobs = list(zip(key, draft, facts_json, completions))
    def one(j):
        k, d, fj, c = j
        c = (c or '').strip()
        for stop in ('\n### Rewritten:', '\n\n\n\n'):
            if stop in c: c = c.split(stop)[0].strip()
        r, det = rl_reward.reward(d, c, json.loads(fj))
        if r is None: r = -1.0; det['judge_failed'] = True      # 判定失败给中性偏下,不给底分
        if det.get('spans'): SPANS[c] = det['spans']
        return r, det, k, c
    with ThreadPoolExecutor(a.judge_workers) as ex:
        res = list(ex.map(one, jobs))
    STEP[0] += 1
    with _lk:
        for r, det, k, c in res:
            LOG.write(json.dumps({'step': STEP[0], 'key': k, 'reward': r, 'copy': det.get('copy'), 'hard': det.get('hard_fail'), 'changed': det.get('changed'), 'dropped': det.get('dropped'), 'invented': det.get('invented'), 'critical': det.get('critical'), 'minor': det.get('minor'), 'fmt': det.get('format_kept'), 'n_spans': len(det.get('spans') or {}), 'len': len(c)}, ensure_ascii=False) + '\n')
        LOG.flush()
    rs = [x[0] for x in res]
    crit = sum(x[1].get('critical') or 0 for x in res); mino = sum(x[1].get('minor') or 0 for x in res)
    print(f'[step {STEP[0]}] reward 均值 {sum(rs)/len(rs):.2f} min {min(rs):.1f} max {max(rs):.1f} 硬失败 {sum(1 for x in res if x[1].get("hard_fail"))}/{len(rs)} 判定失败 {sum(1 for x in res if x[1].get("judge_failed"))} | 严重错 {crit} 轻微 {mino}', flush=True)
    return rs

cfg = GRPOConfig(
    output_dir=a.out, max_steps=a.max_steps, learning_rate=a.lr, beta=a.beta, seed=a.seed,
    per_device_train_batch_size=(a.micro_bs or a.num_generations),
    gradient_accumulation_steps=a.prompts_per_step * max(1, a.num_generations // (a.micro_bs or a.num_generations)),
    num_generations=a.num_generations, max_completion_length=a.max_completion, mask_truncated_completions=True,   # TRL 0.29 没有 max_prompt_length;被截断的采样不参与更新
    temperature=a.temp, top_p=0.95, bf16=True, gradient_checkpointing=True, logging_steps=1, save_steps=a.save_steps, save_total_limit=a.save_limit,
    report_to=[], remove_unused_columns=False, warmup_steps=5, lr_scheduler_type='constant_with_warmup', max_grad_norm=1.0,
    loss_type='grpo', scale_rewards=True, top_entropy_quantile=a.top_entropy_q,
    **({'generation_batch_size': a.gen_batch} if a.gen_batch else {}),
)
class SpanGRPOTrainer(GRPOTrainer):
    """把整篇一个标量的优势,按"哪几句承载了错误"重新分配到 token 上(总量守恒)。
       TRL 0.29 的 _compute_loss 明确支持 (B, T) 形状的 advantages(源码注释为子类预留),所以只在
       生成打分之后加一层,不动框架内部逻辑。advantage>=0 的样本不动 —— 好在哪里定位不了,只定位坏。"""

    def _tok_char_spans(self, ids):
        """每个 token 在解码文本里的近似字符区间(逐 token 解码累加;空格误差对定位句子无影响)。"""
        out = []; pos = 0
        for i in ids:
            t = self.processing_class.decode([int(i)], skip_special_tokens=True)
            out.append((pos, pos + len(t))); pos += len(t)
        return out

    def _weights(self, ids, mask, text, spans):
        import torch as _t
        n = int(mask.sum().item())
        w = _t.ones(len(ids), device=mask.device)
        if not spans or n == 0:
            return w
        low = text.lower(); charspan = self._tok_char_spans([int(x) for x in ids])
        hit = _t.zeros(len(ids), device=mask.device)
        mx = max(spans.values()) or 1.0
        for sp, pen in spans.items():
            j = low.find(sp.lower().strip())
            if j < 0:
                continue
            k = len(low)
            for end in ('. ', '! ', '? ', '\n', '。', '!', '?'):      # 该句从 span 起到下一个句末
                e = low.find(end, j + len(sp))
                if e >= 0: k = min(k, e + len(end))
            for t_i, (cs, ce) in enumerate(charspan):
                if ce > j and cs < k:
                    hit[t_i] = max(float(hit[t_i]), pen / mx)
        raw = 1.0 + a.span_beta * hit
        raw = raw * mask
        tot = raw.sum()
        return raw * (n / tot) if tot > 0 else w

    NLL_BUF = []; NLL_FLOOR = [None]

    def _nll_ref(self, out):
        """每条采样在【参考模型】眼里的平均意外度(nats/token)。ref_per_token_logps 是 KL 项本来就要算的,白拿。"""
        import torch as _t
        rp = out.get('ref_per_token_logps')
        if rp is None: return None
        m = out['completion_mask'].to(rp.dtype)
        n = m.sum(1).clamp(min=1)
        return -(rp * m).sum(1) / n

    def _floor_penalty(self, out):
        """单边地板:低于地板(= 热身期的平均意外度)才扣,高于不给分。返回 (B,) 的惩罚,已在组内去均值。"""
        import torch as _t
        nll = self._nll_ref(out)
        if nll is None or a.nll_floor_w <= 0: return None
        v = nll.detach().float()
        if SpanGRPOTrainer.NLL_FLOOR[0] is None:
            SpanGRPOTrainer.NLL_BUF.extend(v.tolist())
            if len(SpanGRPOTrainer.NLL_BUF) >= a.nll_warm * a.num_generations * a.prompts_per_step:
                b = sorted(SpanGRPOTrainer.NLL_BUF)
                SpanGRPOTrainer.NLL_FLOOR[0] = b[len(b) // 2]      # 用中位数,免得被个别长尾拉偏
                print(f'[意外度地板] 热身 {len(b)} 条采样,地板 = {SpanGRPOTrainer.NLL_FLOOR[0]:.4f} nats/token', flush=True)
            return None
        pen = _t.clamp(SpanGRPOTrainer.NLL_FLOOR[0] - v, min=0.0)   # 只罚"比出发点更好猜"
        g = a.num_generations
        if pen.numel() % g == 0:                                    # 组内去均值,保住 GRPO 的零均值约定
            pen = (pen.view(-1, g) - pen.view(-1, g).mean(1, keepdim=True)).view(-1)
        else:
            pen = pen - pen.mean()
        return pen * a.nll_floor_w

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        adv = out.get('advantages')
        import torch as _t
        pen = self._floor_penalty(out)
        if pen is not None and adv is not None and adv.dim() == 1 and pen.shape == adv.shape:
            if not hasattr(self, '_nll_log_n'): self._nll_log_n = 0
            self._nll_log_n += 1
            if self._nll_log_n % 25 == 1:
                v = self._nll_ref(out).detach().float()
                print(f'[意外度] 本步均值 {v.mean():.4f} 地板 {SpanGRPOTrainer.NLL_FLOOR[0]:.4f} '
                      f'低于地板 {int((v < SpanGRPOTrainer.NLL_FLOOR[0]).sum())}/{v.numel()}', flush=True)
            adv = adv - pen
            out['advantages'] = adv
        if adv is None or a.span_beta <= 0 or adv.dim() != 1:
            return out
        ids = out['completion_ids']; mask = out['completion_mask']
        W = []
        for b in range(ids.shape[0]):
            txt = self.processing_class.decode([int(x) for x, m in zip(ids[b], mask[b]) if m], skip_special_tokens=True)
            sp = SPANS.get(txt.strip()) or SPANS.get(txt)
            W.append(self._weights(ids[b], mask[b], txt, sp) if (sp and float(adv[b]) < 0) else _t.ones(ids.shape[1], device=ids.device) )
        out['advantages'] = adv.unsqueeze(1) * _t.stack(W)
        return out


trainer = SpanGRPOTrainer(model=model, reward_funcs=fact_reward, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
print('开训', time.strftime('%H:%M'), flush=True)
trainer.train(resume_from_checkpoint=(a.resume or None))
trainer.save_model(os.path.join(a.out, 'adapter'))
import shutil
try:
    shutil.copy(os.path.join(a.dpo_adapter, 'prompt_format.json'), os.path.join(a.out, 'adapter', 'prompt_format.json'))
except Exception as e: print('prompt_format 复制失败', e)
if a.save_merged:
    m = trainer.model.merge_and_unload() if hasattr(trainer.model, 'merge_and_unload') else trainer.model
    m.save_pretrained(os.path.join(a.out, 'merged'), safe_serialization=True, max_shard_size='5GB'); tok.save_pretrained(os.path.join(a.out, 'merged'))
    try: shutil.copy(os.path.join(a.dpo_adapter, 'prompt_format.json'), os.path.join(a.out, 'merged', 'prompt_format.json'))
    except Exception as e: print('prompt_format 复制失败', e)
    print('MERGED_SAVED', os.path.join(a.out, 'merged'), flush=True)
print('GRPO_DONE', a.out, flush=True)
