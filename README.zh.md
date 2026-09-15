# humanizer

[English](README.md) | [中文](README.zh.md)

一个把 AI 写的草稿改成"像人写的"的小模型,改写时保留每一个事实、数字、人名和日期。底座 `google/gemma-4-E4B`(约 4B 有效参数),Mac(MLX / llama.cpp)和 CUDA 显卡(transformers)都能本地跑。

**v2(2026-09-14):SFT + DPO + 强化学习(GRPO),奖励只看忠实度、语域和复用率。** **训练全程(包括 RL 阶段)没有任何 AI 检测器参与:不做奖励、不做过滤、不做训练信号、不做选样标准。** 模型学的只是"人是怎么写的"和"事实有没有保住";它能过检测器是副作用。下面的检测器数字只作外部核对。

## 它做什么

输入:语言模型写的草稿(邮件、作文、报告、论文段落、论坛帖,中英文皆可)。
输出:同样内容的改写。事实、数字、单位、日期、人名、引文应原样保留;结构、措辞、节奏、语气会变。

它是改写器,不是生成器:不会加论点、加例子、加废话,并且训练它保留问候、落款和主题行。

## 训练方法

1. **监督微调(28.6k 对)。** 每对是"AI 草稿 → 人写的原文"。人这一侧永远是真人文本(bioRxiv/PubMed 摘要、政府报告、学生作文、公司和邮件列表邮件、Reddit 和 Hacker News 帖子、知乎回答、中文正式文体等);AI 这一侧是当前一线模型(Claude Sonnet、GPT-5 级、GLM-5.3)根据人写原文写出的同内容草稿。模型学的是同一内容上"机器 → 人"的方向。从不使用合成的"人类"文本。
2. **两轮偏好优化(DPO,共 1,898 对)。** SFT 模型对每篇草稿写 6 个候选(3 个普通,3 个带解码期反照抄惩罚)。另一个模型(GLM-5.3)只判**忠实度**:事实是否全在、意思有没有改、有没有添加、问候落款有没有丢、是不是原样照抄,并分严重等级。*chosen* = 没有忠实度问题且与草稿逐字重合最低的候选;*rejected* = 有严重事实错或几乎原样照抄的候选。风格、长度、句式、检测器分数都不进入选样。
4. **解码期守卫(只在推理时)。** 第一发若照抄了草稿 35% 以上的 5-gram,就带惩罚重采一发:会拼出草稿里已有 5-gram 的 token 被降低 logit(数字 token 豁免)。这去掉了"偷懒原样照抄"的失败模式,不伤事实。

3. **强化学习(GRPO,300 步,v2)。** 在合并后的 SFT + DPO 模型上再训一个 LoRA(TRL 的 GRPO):每篇草稿采 8 发,600 篇同体裁草稿。奖励全部是忠实度项:判定模型(GLM-5.3)对照草稿的原子事实清单逐条核对每一发(严重事实错 −3.0/条、轻微 −0.15/条、凭空添加 −2.0、意思反转 −2.0、丢格式要素或问候落款 −1.5),草稿的语域被"规范化"(缩写展开、俚语改正式、故意的小写/缺标点被"修好")−1.0,再加一个超线性的**复用率**斜坡:复用率 = max(逐字 5-gram 照抄, 抹掉实词后的句法骨架 5-gram 复用),0.31 以下免罚,整篇照抄扣到 −4。错误的惩罚落在犯错的那句话上(token 级优势重加权)。每 25 步存一个 checkpoint,发布的那一个按留出集挑。两条留下来的经验:只罚事实错、不加语域项,RL 会把模型训得越写越"规范",检测器通过率从 81% 掉到 52%;复用率压不动之后继续训只会多出事实错(斜坡权重 8 而不是 4 时,300 步的版本严重错 7/62)。

训练脚本在 `training/`(LoRA SFT、DPO、GRPO 及 `rl_reward.py` + `textmetrics.py`、候选生成、判定提示词、两票判定)。训练数据不公开,因为人这一侧来源许可不一。

## 评测

39 例"日常使用"评测集(`eval_daily/`):作文、报告、论文段落、邮件、推特/Reddit/LinkedIn 帖,加 8 例中文;草稿由 Claude Sonnet 撰写。每例 2 个样本。

忠实度由两个独立判定模型(GLM-5.3 与 gpt-5.6)各判一次,两个都报的错才算;v1 用同一协议重判,列间可比。

| 指标(62 篇英文样本) | **v2(SFT + DPO + GRPO)** | v1(SFT + DPO) | 生产基线(Qwen3.5-4B 改写器) |
|---|---|---|---|
| 逐字 5-gram 照抄率,中位 | 0.14 | 0.15 | 0.12 |
| 复用率(逐字 ∨ 句法骨架),中位 | **0.29** | 0.34 | — |
| 照抄超过草稿 35% 的样本 | 0 | 1 | 33 / 93 |
| **严重事实错**(意思翻转、数字或事件改变) | **0 / 62** | 3 / 62 | 约 30% |
| 轻微 / 无错 | 20 / 42 | 15 / 44 | — |
| 丢格式要素(主题行、标题、列表、落款) | **5 / 62** | 12 / 62 | — |
| 中文通过判定 | **13 / 16** | 11 / 16 | 4 / 16 |
| Originality.ai 判"人写"*(仅外部核对,从未优化)* | **53 / 62 = 85%** | 50 / 62 = 81% | 53 / 93 = 57% |

同一检测器上,训练体裁的真人原文 9/9 判人写;草稿输入约 0/62。v2 最弱的体裁是社交帖(5/8)、论文段落(4/6)和报告(6/8);邮件 10/10。

### 已知失败模式(用之前请读)

* **意思翻转**(谁向谁建议、"收到"→"下单"、指标换名、比较方向反转)是 v1 的主要失败模式(约十分之一)。v2 在 62 篇样本上一例没有,但样本小:数字、日期和每个论断的方向仍请务必校对。
* v2 剩下的错误是**丢限定词**:"估计 4.2%" → "4.2%"、"结果提示" → "我们得出结论"、"12 小时或以上" → "超过 12 小时"。大约三分之一的输出有一处这样的偏移。
* 改写比 v1 更深;偶尔有一句会改坏("from thousands to thousands")。读着不对就重采一发。
* 中文弱于英文(13/16)。
* 120 词以下的短稿改写不稳。

## 样例

英文中文各一例,更多见 [docs/samples.md](docs/samples.md)(草稿为 Claude 所写;输出为模型原样,未经修改)。

### Email to a professor / 给教授的邮件

**Draft (Claude) / 草稿:**

> Dear Professor Lin,
> 
> I'm following up on the dataset access request I submitted on October 14th. I understand these reviews take time, and I appreciate you looking into it.
> 
> To give a bit more context: my analysis only uses the 2019–2021 subset, and I'm interested purely in the temporal patterns, so I won't need any of the personally identifying fields. If it would help, I'm happy to send a more detailed methodology document, or to walk through the scope of the request in person at whatever time suits you.
> 
> Please let me know if there's anything else you need from me to move the approval along. Thank you again for your time.
> 
> Best regards,
> Wei

**Humanizer output / 改写:** *(5-gram copy 0.14)*

> Dear Professor Lin,
> 
> Just following up on the access request for the data set that I submitted on October 14th. I know that the review process takes time and I appreciate the time you're taking to look into this.
> 
> To provide a little more context, I am only using the 2019-2021 subset for my analysis and I am only interested in the temporal patterns so I won't need any personally identifiable fields. If you would like, I can provide a more detailed methodology document or we can go through the scope of the request in person at your convenience.
> 
> Please let me know if there is anything else that I need to provide to help with the approval process. Thanks again for your time.
> 
> Best regards,
> Wei

### 中文邮件 / Chinese email

**Draft (Claude) / 草稿:**

> 李老师您好,
> 
> 关于上周提到的数据访问申请,我想再补充几点说明。
> 
> 首先,本次分析只涉及 2019 至 2021 年的时间序列子集,不需要任何身份识别字段,所以不会触及隐私数据。其次,考虑到审批通常需要两到三周,我已经把实验计划相应顺延,整体进度不会受影响。另外,如果您对数据的使用方式还有疑问,我可以提供一份更详细的方法说明,或者找个您方便的时间当面沟通。
> 
> 希望这件事能尽快推进。再次感谢您的时间和帮助。
> 
> 祝好
> 小王

**Humanizer output / 改写:** *(5-gram copy 0.31)*

> 李老师您好，
> 
> 上周关于数据访问申请的事情补充说明一下。
> 
> 1、本次只使用2019-2021的时间序列子集，不涉及任何身份识别字段，所以不存在涉及隐私数据的问题；
> 
> 2、考虑到审批周期大概需要两到三周，所以已经将实验计划顺延，不影响整体进度；
> 
> 3、如果对数据的使用方法还有疑问，我可以再提供一个更详细的方法说明，或者找个时间见面沟通。
> 
> 希望能尽快推进。再次感谢您的时间和帮助！
> 
> 祝好
> 小王


## 用法

### GGUF(llama.cpp、Ollama、LM Studio、llama-cpp-python)— 推荐

文件在 `jialinyyzz/humanizer-gemma-4-e4b` 的 `gguf/` 目录:`Q8_0`(8.0 GB)、`Q6_K`(6.2 GB)、`bf16`(14.9 GB)。Q8_0 和 Q6_K(v1 上实测,v2 同配方)的事实错都在 bf16 的噪声带内。**Q5_K_M 和 Q4_K_M 不发布**:Q5_K_M 严重错翻近三倍(17/62 对 6),Q4_K_M 和 MLX 4bit 一样退化成乱码(见 `docs/QUALITY.md`)。

```bash
pip install llama-cpp-python        # Mac 加 CMAKE_ARGS="-DGGML_METAL=on",N 卡加 "-DGGML_CUDA=on"
python humanizer/gguf_infer.py --gguf humanizer-gemma-4-e4b-Q8_0.gguf --format prompt_format.json draft.txt
```

`gguf_infer.py` 通过 logits 处理器实现反照抄守卫。`llama-cli`、Ollama、LM Studio 也能直接跑同一个文件,但加不了惩罚;用它们时若输出仍照抄草稿三分之一以上,再采一次即可。

### transformers(CUDA),合并后的 bf16

```bash
python humanizer/hf_infer.py --model jialinyyzz/humanizer-gemma-4-e4b draft.txt
```

### MLX(Apple 芯片),仅 bf16

`humanizer/mlx_nocopy_server.py` + `humanizer/humanize.py` 用合并后的 bf16 权重起服务并带守卫。注意:mlx_lm 的 Gemma 4 加载器会拒绝 HF 权重里 18 个共用 KV 层的 54 个无用 k/v 张量,加载前要去掉 `layers.24–41.self_attn.(k_proj|v_proj|k_norm)`。这个模型的 MLX 4/6 比特量化不可用(见 `docs/QUALITY.md`),Mac 上请用 GGUF 量化版。

### 提示词格式 —— 先读这一段

这是一个**基座续写模型,不是聊天模型**:没有 system prompt,没有 chat template,没有 `<start_of_turn>` 之类的回合标记。
你要把下面这段原样喂给它,让它接着写:

```
Rewrite the text below so it reads like a person wrote it, not a language model.

Reorganize it as you see fit. Vary sentence length on purpose. Cut hedging,
throat-clearing, and any sentence that only announces what comes next.
Prefer the concrete word over the abstract one. It is fine to sound uneven.

Every fact, number, unit, date, name and quotation must survive unchanged.

<你的草稿放这里>

### Rewritten:

```

指令和分隔符原样放在权重旁边的 `prompt_format.json` 里(字段 `instr` 和 `sep`;分隔符就是
`\n\n### Rewritten:\n\n`,注意后面那个空行)。**必须逐字节复刻** —— 模型就是在这个包装上训练的,
改写指令措辞或少一个空行都会实测变差。

生成到 EOS 停。采样:temperature 0.85,top-p 0.95,约 900 个新 token。

**用 Ollama / LM Studio 的注意**:这两个默认会套上基座的 chat template,把你的文本包进回合标记里,格式就废了。
要关掉。Ollama 的 Modelfile 这样写(原样透传提示词):

```
FROM ./humanizer-gemma-4-e4b-Q8_0.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.85
PARAMETER top_p 0.95
PARAMETER num_predict 900
```

然后把上面一整块(指令 + 草稿 + `### Rewritten:`)作为一个提示词发过去。我们自己的 llama.cpp 路径是
`humanizer/gguf_infer.py`,它从 `prompt_format.json` 拼提示词并带反照抄守卫;上面这个 Modelfile 是等价的手写版,
**我们没有实测过**。

## 权重

* `jialinyyzz/humanizer-gemma-4-e4b` — 一个仓库放全部变体:根目录是合并后的 bf16(transformers 格式,SFT + DPO + GRPO 已合入底座),`gguf/` 目录是 llama.cpp GGUF(Q8_0、Q6_K、bf16;Q5_K_M、Q4_K_M 因 6bit 以下忠实度崩塌不发)和 `prompt_format.json`。现在的文件是 **v2**;v1(只有 SFT + DPO)可从仓库的提交历史取回。

均派生自 `google/gemma-4-E4B`,受 Gemma 使用条款约束(见 `NOTICE`)。本仓库代码为 Apache-2.0。

## 复现

`training/train_sft2.py`(LoRA r=16,只挂语言塔,1 epoch,有效批 16)→ `training/gen_candidates.py` → `training/build_prefs.py --tiers --identity-bad`(需要 `~/.config/zai_key` 里的 GLM API key)→ `training/train_dpo.py`(β=0.1,lr 5e-6,1 epoch)→ `training/train_grpo.py`(GRPO,300 步,每步 2 篇 × 8 发,lr 5e-6,奖励见 `training/rl_reward.py`,`W_COPY=4`;每 25 步留 checkpoint,按留出集挑)。`training/export_merged_gguf.sbatch` 把三层 LoRA 合进底座并出 GGUF。评测:`training/gen_cases.py` 加 `COPY_PENALTY=2 COPY_N=5 ADAPTIVE_COPY=1 ADAPTIVE_THR=0.35`,再 `training/glm_eval_en.py` / `glm_eval_zh.py`。
