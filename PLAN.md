# 在 ChartQA 上应用 TON：实施计划

更新日期：2026-09-17  
状态：SFT 与 GRPO 工程代码已实施；GRPO 尚未在目标 A800 上完成 P0 或正式训练。
项目定位：复现 TON 的方法，并迁移到 ChartQA 与 Qwen3-VL；不追求复现原论文的全部任务或数值。

## 1. 目标与已经确认的决策

在图表问答中，让同一个模型自主选择是否生成显式推理，在保持答题质量的同时减少用户等待答案的时间。

| 项目 | 决策 | 状态 |
| --- | --- | --- |
| 方法 | TON：reverse thinking 冷启动数据、随机 thought dropout SFT、GRPO | 用户已确认 |
| 数据集 | ChartQA，使用已有 train / val / test 划分 | 用户已确认 |
| 基座 | `Qwen/Qwen3-VL-4B-Instruct` | 用户已确认 |
| GPU | 单张 NVIDIA A800 80GB，所有 GPU 作业串行 | 用户已确认 |
| 可用时间 | 不设硬性时长上限；仍记录每阶段 GPU 小时 | 用户已确认 |
| 训练方式 | LoRA SFT + LoRA GRPO，BF16，冻结视觉模块和基础权重 | 根据算力作出的工程选择 |
| 框架 | 优先 LLaMA-Factory 做 SFT、EasyR1 做 GRPO、vLLM 做采样和独立评测 | 需通过版本与单卡兼容性检查 |
| 主实验规模 | 2,000 个训练问答；两种 SFT 使用完全相同的题目与推理母版 | 项目初始配置 |
| 重复运行 | 主对照与 TON 各 3 个训练种子，配对比较 | 项目初始配置 |

采用 LoRA 是为了给单卡上的视觉输入、训练激活和多条 rollout 留出显存。80GB 不能自动保证任意 batch、分辨率和上下文都可运行；是否可行以阶段 P0 的实测为准。不因单卡实现困难静默改用第二张 GPU。

### 1.1 方法边界

必须保留：

- 同一模型、同一次自回归生成中选择空推理或非空推理。
- SFT 推理数据由基座结合训练图片、问题和标准答案生成；正式训练输入不包含标准答案提示。
- 以 0.5 概率随机清空推理内容，答案和外层标签保持不变。
- GRPO 根据格式与 ChartQA 答案正确性反馈训练。
- 所有任务相关输出都能自动解析和评测。

本计划不加入：短推理蒸馏、按难度或模式收益调整 dropout、长度/延迟奖励、强制 Think/NoThink 采样配额、额外路由器、DPO、额外视觉重感知、crop/zoom、OCR 工具、多模型级联。

“不思考”指不生成显式推理 token，不代表模型不进行视觉编码或内部计算。吞吐与延迟优化只涉及正常工程配置，不改变方法。

### 1.2 与原 TON 的区别

保留方法，替换数据集、基座、训练框架和参数更新方式。ChartQA verifier 是任务迁移所需的适配。静态随机 dropout 数据文件、LoRA 和单卡配置属于工程实现选择，报告中均需披露。不能将本项目称为 TON 原始实验的精确数值复现。

## 2. 研究问题与完成标准

需要回答：

1. 相比同数据的全推理 SFT，thought dropout 是否使模型在自由生成时更容易输出空推理？
2. GRPO 在 SFT 的基础上带来什么准确率、输出长度和跳过率变化？
3. 相比主对照，自适应模型是否降低实际答案延迟？
4. 相比始终直接回答，增加的推理是否保留了部分困难样本的准确性？

工程完成不要求出现正向研究结果。满足以下条件即可完成方法迁移：数据、训练和评测可复现；两种行为都被正确支持；奖励和 token mask 通过检查；主对照与 TON 都完成训练；独立测试、延迟分析和失败报告齐全。

预先登记的展示目标：相对主对照，测试 relaxed accuracy 点估计下降不超过 1 个百分点，并降低平均答案延迟。此数值是项目评价门槛，不是论文参数或对结果的保证。报告配对置信区间；仅点估计达标不能声称已经统计证明准确率不劣。若未达标，如实报告，不在 test 上选参数追逐目标。

## 3. 数据来源、划分与抽样

### 3.1 数据版本

- 主入口：`HuggingFaceM4/ChartQA`，在实施时固定 dataset revision。
- 当前数据卡记载：train 28,299；val 1,920；test 2,500。下载后实际检查并写入 manifest。[S3]
- 官方完整版和底层表格可用于核对数据来源与离线质量检查。[S4]
- 保存来源、revision、下载日期、原始字段、许可证与署名要求；不随代码默认提交图片和整套数据。
- ChartQAPro 不参与本轮训练或调参；跨数据集扩展不列为本计划完成的前提。

### 3.2 主实验数据分配

| 数据产物 | 来源及大小 | 用途 |
| --- | --- | --- |
| `train_candidates` | train 中固定顺序的候选池 | 推理生成、质量检查与必要补样 |
| `train_core` | 合格候选中的 2,000 个唯一问答 | full/drop50 SFT 与两组 GRPO |
| `dev_quick` | val 中固定抽取 256 题 | 训练过程监测和工程诊断 |
| `dev_full` | 全部 val | 参数/检查点选择 |
| `test_full` | 全部 test | 设置冻结后的最终评测 |
| `latency_test` | test 中预先按元数据选定 256 题 | 重复串行延迟测量 |

抽样种子固定为 `20260917`。训练池近似保留原 train 的 human/machine 来源比例与数值/文本答案比例；不依据基座是否答对来筛题，不将 human/machine 当成难易或动作标签。保存问答数量与独立图表数量。

同一图片可以包含多个不同问题，但重复问答仅保留一个。正式划分沿用已有 split，不将 test 或 val 移入训练。生成推理和 dropout 必须在划分之后进行。

### 3.3 泄漏审计

- 用稳定图片 ID、解码像素哈希、规范化问题检查精确重复；用感知哈希辅助发现改尺寸或重新编码后的近重复。
- 原始行号只有与 dataset revision 绑定后才能作为稳定来源信息。
- 若 train 与 val/test 存在确认的重复图表，从训练候选池排除对应图表并补样，保留审计清单。
- 若 val/test 存在图表重叠，不删除标准 test；隔离重叠验证图表的调参用途，并披露影响及实际开发集大小。
- 不把表格、OCR 真值、来源类别、生成时的答案提示、质量检查备注或 dropout 标志放入模型输入。
- 上述检查只控制本次后训练泄漏，不声称能证明基座预训练从未见过 ChartQA。

### 3.4 统一内部记录

```json
{
  "id": "chartqa:revision:split:row_id",
  "split": "train",
  "image_id": "stable_chart_id",
  "image_path": "relative/path/from/data_root.png",
  "image_sha256": "...",
  "question": "...",
  "answers": ["..."],
  "source": "human",
  "answer_kind": "numeric_or_text_or_list",
  "provenance": {"dataset": "HuggingFaceM4/ChartQA", "revision": "..."}
}
```

`answers` 保留原始列表语义；字符串形式的多值答案不能误拆成多个可任选参考答案。图表 ID、来源和答案类型只用于抽样、审计与分组统计。原始题目及答案保留，规范化值另存，不覆盖原始标签。

## 4. SFT 数据构造

### 4.1 Reverse thinking

使用固定 revision 的原始 Qwen3-VL-4B-Instruct，在离线阶段接收图片、问题和标准答案，生成推理。该机制来自 TON；不使用更大的外部教师模型。[S1]

生成提示作为 `prompts/reverse_thinking.txt` 单独版本化，要求根据图中证据说明计算或判断过程。不给出人为的极短长度目标，不要求所有题套用固定步骤，不做后续推理压缩。允许给出答案作为条件，但要求输出推理而不是复述“已知答案”。

初始生成设置：temperature 0.7、top_p 0.9、最大生成 2,048 token、每次生成 1 个候选；固定随机种子并保存原始响应。每题最多进行 2 次重试，仅用于格式错误、截断或明确的解释错误，不依据是否展示长推理进行筛选。记录每次尝试的种子与失败原因。

生成阶段的答案提示不得出现在正式 SFT 的 system/user 消息中。SFT 的 assistant 答案使用检查通过的参考答案，不能只因为强行接上了参考答案就把推理标为正确。

### 4.2 质量检查

自动检查：图片可读取、问题与答案非空、推理可解析、未截断、无重复控制标签、无答案提示残留；在可解析时检查算术一致性。

人工抽检至少 100 条，覆盖两种来源、数值和非数值答案、不同推理长度。检查图例/系列对应、年份、读数、单位、计算和结论。底层表格仅可作为检查辅助，不注入学生输入；颜色或空间关系不能仅凭表格判断。

建议放行条件：抽检中无残留答案提示，明确错误解释比例不高于 5%。未通过时，定位生成提示或数据处理问题，修订后重新抽检；这是数据质量门槛，不是模型效果调参。

失败样本、重试和补样均留档；不按模型回答难度删除样本。必须报告质量筛选前后的来源/答案类型分布，承认解释筛选可能改变训练分布。若重试后不足 2,000 条，从冻结候选顺序补样。

### 4.3 保存母版与随机 dropout

每条母版保存：原始 ID、完整推理、标准答案、生成模型/提示哈希/采样设置、质量标志。

从同一母版导出：

- `sft_full`：全部保留推理。
- `sft_drop50`：独立 Bernoulli(p=0.5) 决定是否清空该条推理，固定 dropout 种子 `17`。

dropout 内容为两个换行符；保留 `<think>`、`</think>`、`<answer>`、`</answer>`。样本数、顺序和答案与 full 版本相同。0.5 是抽样概率，不要求恰好 1,000 条为空；记录实际比例。静态物化一次后各 epoch 不再重抽，三组训练种子共用该版本以控制变量。已经 dropout 的文件不能再次 dropout。

```text
完整：<think>{rationale}</think><answer>{answer}</answer>
空推理：<think>\n\n</think><answer>{answer}</answer>
```

上例中的 `\n` 在实际训练字符串中是换行符，不是反斜杠字符。

### 4.4 统一学生提示与模板

SFT、GRPO 和自适应评测使用同一语义提示：根据图表回答；需要推理时在 think 标签内完成；可以留空；最终短答案放在 answer 标签内。该提示同时用于 full 和 drop50 分支，避免用不同指令制造行为差异。

保留基座的多模态 chat template 和图像占位逻辑。不要未经检查替换成纯文本模板，也不新增随机初始化的特殊 token。控制标签可使用现有 tokenizer 的普通 token 序列。

必须检查模板是否预填 think、是否自动删除空 think、是否在数据清洗中 strip 掉必要结构。空推理的闭合标签和后续答案 token 都要参与 assistant loss。SFT 不对 system/user/图像占位的输入 token 计算监督损失。

## 5. ChartQA verifier 与 TON reward

### 5.1 先固定评价规则

实现前审计 ChartQA 官方参考评价代码，保存具体 revision 与函数出处。[S5] 主指标在完成核验后采用 ChartQA relaxed accuracy；同时报告严格答案匹配作为诊断。不能直接使用 EasyR1 的通用数学 boxed-answer reward。[S8]

必须明确并测试：数值解析与容差、零与负数、百分号、千分位、科学计数法、文本大小写/空白、多值答案、多个参考答案、年份、NaN/Inf、空输出和截断。参考实现的边界行为需记录，不能静默加入“35、35%、0.35 全等”等自定义规则后仍声称标准成绩。若发现参考代码的可疑行为，标准兼容指标与严格诊断指标分开呈现。

### 5.2 两个奖励分量

`R_total = R_format + R_answer`，两个分量均为 0 或 1，复用 TON 的任务奖励结构；其中答案匹配替换为 ChartQA 判分。[S1]

- `R_format`：响应完整符合 think 后接 answer 的结构，标签各恰好一组，answer 非空；think 可为空白。仅允许约定的外层空白，不接受嵌套标签、重复答案块或尾部额外解答。
- `R_answer`：只从唯一且闭合的 answer 区域提取答案，再调用 ChartQA verifier。无法唯一解析时为 0。格式其余部分错误但答案可唯一解析时，可以独立获得答案分，保留加法奖励语义。
- 不从整个响应或 think 中搜索正确数字；不取多个答案候选中的最有利一个。
- 不添加长度、跳过率、推理质量、动作均衡或延迟奖励，不把正确性改为门控成本形式。

典型情况：格式合法且答案正确为 2；格式合法但答错为 1；答案正确且 think 结构非法但答案块可唯一解析为 1；无法解析且格式错误为 0。长度截断造成标签未闭合时按规则判分，不自动补闭合标签。

在正式训练前，用人工构造的边界样例和 ChartQA 样本核验奖励、评测与解析器的一致性。奖励模块返回 total/format/accuracy 及诊断数据；哪些字段可直接进入 EasyR1 回调，要按固定版本接口检查。

## 6. 单卡训练配置与兼容性检查

以下是项目起始值，不是声称已经在 A800 上实测成功的配置。通过 P0 后冻结成实际 YAML 和版本清单。所有主对照使用相同硬件、精度、分辨率与上下文上限。

### 6.1 环境

- Linux GPU 服务器：A800 80GB × 1；先记录驱动、CUDA、CPU 核数、内存、磁盘、GPU 时钟/功耗限制。
- 建议主机 RAM 至少 64GB、可用磁盘至少 100GB；若需大量 CPU offload 或保存更多检查点，按实测增加资源。
- Python 3.11 起步；固定 PyTorch、Transformers、PEFT、FlashAttention、vLLM、LLaMA-Factory、EasyR1 版本或 commit。
- SFT 与 RL 可使用独立环境，避免依赖强行混装；以模型/adapter/processor 导出兼容连接两个阶段。
- 不照搬 TON 旧版 vLLM 环境。EasyR1 的 Qwen3-VL LoRA 示例使用两张卡，不是单卡成功保证。[S6]
- 先记录 schema 与有效配置，再执行；以下 batch 描述为语义要求，不把不同框架的同名 batch 字段当成同一单位。

### 6.2 共用模型与视觉配置

| 参数 | 初始值 |
| --- | --- |
| precision | BF16，不做 4-bit/8-bit 量化 |
| LoRA rank / alpha / dropout | 64 / 128 / 0 |
| LoRA 位置 | 语言模块的 q/k/v/o、gate/up/down 投影；按完整模块路径过滤 |
| 冻结模块 | 原始基础权重、视觉编码器、多模态连接模块、词嵌入与输出头 |
| gradient checkpointing | 开启 |
| min_pixels / max_pixels | 65,536 / 1,048,576，保持宽高比，遵循 processor 的取整规则 |
| 最大输入 token | 4,096，按处理后的真实多模态 token 计数 |
| 最大响应 token | 2,048，包含推理、答案和生成的控制标签 |
| SFT 总序列上限 | 6,144，检查输入与输出都完整保留 |

这是允许上限，不要求把所有图片放大到 max_pixels。记录实际 resize 尺寸、视觉 token 与输入长度分布。禁止通过静默裁掉答案使 full/drop50 数据不再成对。训练超长样本需留档并对两个分支一致处理；测试超限须计入报告，不从分母删除。

按真实模块路径打印可训练参数清单并核对参数量。不要仅用 `all-linear` 就假设视觉模块不会被训练；EasyR1 配置也明确区分视觉 LoRA 排除项。[S7]

### 6.3 SFT

| 参数 | 初始值 |
| --- | --- |
| 数据 | `sft_full` 或 `sft_drop50`，各 2,000 条 |
| epoch | 2 |
| learning rate | 1e-5 |
| scheduler / warmup | cosine / 0.1 |
| micro batch | 1 条/卡 |
| gradient accumulation | 16，即有效 batch 16 条 |
| weight decay / max grad norm | 0.01 / 1.0 |
| packing | 第一版关闭，简化图像及 loss mask 检查 |
| train_on_prompt | 关闭 |
| 训练种子 | 42、43、44 |

full/drop50 使用相同训练题顺序与更新次数；它们的监督 token 总数不同是方法本身造成的差异，需要报告，不能靠额外 epoch 隐式补齐。

### 6.4 SFT 到 GRPO 的检查点衔接

对每个 SFT 分支导出独立的 merged BF16 模型，并同步 tokenizer、processor、chat template。验证合并前后在固定样本上的 logits/生成没有超出合理数值误差的偏差，确定容差并记录。

GRPO 从该 merged SFT 模型初始化新的 LoRA adapter，初始策略应与 SFT 模型一致；reference 固定为对应 SFT 模型。两个分支分别使用各自 reference。检查初始化等价、KL、reference 冻结和 rollout 权重同步，不把 full 分支的 reference 误用于 drop50 分支。

此做法属于 LoRA 工程实现，报告中披露；不同时叠加两个不明状态的 adapter。

### 6.5 GRPO

| 参数 | 初始值 |
| --- | --- |
| prompt pool | 与 SFT 相同的 `train_core`，模型输入不含解释或答案 |
| 每次采样的不同问题 | 16 |
| 每题 rollout 数 G | 4，即每次产生 64 条候选轨迹 |
| rollout temperature / top_p | 1.0 / 1.0 |
| 更新 micro batch | 1 条轨迹，必要时梯度累积 |
| 每轮采样后的更新 | 1 个优化 epoch；实际 optimizer steps 单独记录 |
| learning rate / scheduler | 1e-6 / constant |
| GRPO clip range | 0.2，核对实际版本对应字段 |
| KL coefficient | 0.04，作为起始值；明示相对 EasyR1 默认值的覆盖 |
| weight decay / max grad norm | 0.01 / 1.0 |
| 初始训练上限 | 200 个采样迭代，约 3,200 次题目抽取、12,800 条候选 |
| quick-val / save | quick-val 每 20 个采样迭代；可续训 checkpoint 每 100 个采样迭代，最多保留 2 份（50GB 数据盘约束） |
| GPU / tensor parallel | 1 / 1 |
| vLLM memory utilization | 从 0.35 起测，按训练/rollout 生命周期调整 |
| DAPO 式过滤、动作配额、额外长度奖励 | 全部关闭 |

EasyR1 的 trainer step、rollout batch、actor global batch 不保证恰好对应上表的采样迭代。P0 必须用日志确认题目数、轨迹数、每题分组、有效更新 batch 与实际 optimizer steps；将映射写入配置说明，不能只照抄数字。上表的样本量计算以明确的“16题 × 4候选 × 200次采样”为准。

全组同奖励时正确处理零优势，记录全对组、全错组、同奖励组比例。不得为制造优势而临时加入成本惩罚。保存每条轨迹的 prompt_id、policy_version、reward、token 数和 finish reason。

### 6.6 单卡 OOM 与性能回退顺序

1. 确认 SFT、RL、教师生成和评测进程没有同时占用 GPU；检查释放和 vLLM 生命周期。
2. 减少同时采样的问题数、micro batch、logprob 计算 batch；用累积保持约定的训练量。
3. 降低 rollout KV cache 占用，启用框架支持的 actor/reference/optimizer offload；实测主机内存和交换开销。
4. 若仍失败，评估框架版本兼容性，不直接认为模型太大。
5. 最后才考虑降低 LoRA rank、分辨率或响应上限；所有分支统一更改并重新建立基线，披露变化。

保留 G=4 和非量化优先级较高。任何实质配置调整必须写入决策记录；不静默改模型、方法、数据集或增加 GPU。

## 7. 实验矩阵与检查点选择

| ID | 条件 | 作用 |
| --- | --- | --- |
| B0 | 原始模型，提示直接给短答案 | 速度与无需显式推理的准确率参考 |
| B1 | 原始模型，提示先推理再回答 | 基座显式推理参考 |
| B2 | 原始模型，统一自适应提示 | 检查仅提示能否已有选择能力 |
| S0 | full SFT，统一自适应提示 | 无 dropout 冷启动对照 |
| S1 | drop50 SFT，统一自适应提示 | 分离 SFT 的作用 |
| R0 | full SFT + GRPO，统一自适应提示 | 主训练对照 |
| R1 | drop50 SFT + GRPO，统一自适应提示 | 完整 TON 迁移 |

S0/S1/R0/R1 各跑 3 个训练种子；同一种子成对比较。B0/B1/B2 的最终评测使用固定确定性设置。R0 应称为“全推理 SFT + GRPO 对照”，不冒称它等同于 TON 论文所有任务的 vanilla GRPO 初始化方式。直接从基座做 GRPO 可作为后续补充，不是本轮必做项。

S0/R0 自由生成时也允许空 think，不对其额外强制非空。B0/B1 是提示模式对照，报告指令遵循与实际 think 行为；若以后使用前缀强制控制，需要另命名并记录预填 token，不与普通提示结果混用。

SFT 使用固定最后 epoch 检查点。GRPO 每 20 次采样在 dev_quick 监测，但考虑 AutoDL 数据盘只有 50GB，仅在第 100、200 次采样保存包含 optimizer/RNG 的可续训 checkpoint，最多保留 2 份；结束后只用 dev_full 在实际保存的候选中按最高 relaxed accuracy 选择，准确率相同时取较早检查点。主指标不按 test 结果或最短响应挑模型。

先完成 seed=42 的整套流程；修复实现问题后冻结正式配置，再完成 42/43/44 配对实验。若首轮配置被修改，旧结果归档为 pilot，不混入正式均值。

## 8. 评测与真实延迟

### 8.1 答案质量与行为

- 主指标：ChartQA relaxed accuracy，总体及 human/machine 分项。
- 辅助：严格匹配、格式合法率、答案可解析率、空 think 比例、非空 think 条件下长度。
- 成本：实际生成的总 token、think token、答案 token、输入与视觉 token、截断率、GPU 小时。
- 跳过率用解析后的 think 内容 `strip()` 为空判断；格式非法样本单列，不能算成功跳过。
- 平均长度与延迟统计包含答错样本，不只报告正确回答；成功样本条件统计可作为附表。
- 所有条件使用同一 test 样本集与输入分辨率。缺失、失败、超长和超时均保留记录，不能从准确率分母静默移除。

### 8.2 延迟定义

主用户指标是首次实际答案内容出现的时间 `T_answer_first`；同时记录请求至响应结束的 `T_complete` 与普通 TTFT。think 标签、推理内容和 answer 开标签都不算实际答案内容。

- 请求开始：预热后、图片文件已在内存时，将图片和问题提交到本地推理服务的时刻。
- 时钟使用单调时钟；主计时包含请求序列化/本地传输、处理器、视觉编码、prefill、生成与解码。
- 网络下载图片和首次加载模型不计入主稳态延迟，单独报告冷启动。
- 使用流式输出或可定位输出 token 的时间戳；非流式返回总耗时不能冒充首次答案延迟。
- 标签可能跨 token/流式 chunk，解析器必须缓冲并正确定位首个答案字符。
- 未生成答案的请求不具有有效 `T_answer_first`：报告失败率及有效样本分布，并额外提供使用预先固定超时上限计罚的结果，避免幸存者偏差。

质量测试使用 greedy / temperature=0、每题 1 条输出，统一最大响应上限。延迟测试同样使用 merged BF16 模型、同一 vLLM 版本、batch=1、concurrency=1、相同输出上限，串行进行。vLLM prefix cache 对所有条件关闭；明确记录多模态缓存策略，避免重复请求命中缓存造成虚假提速。

延迟样本固定 256 题，按来源和输入长度分层选择，不按答案正确或动作选择筛样本。每个模型先用独立样本预热 20 次，然后重复 3 轮；各轮固定并打乱样本顺序，轮换模型执行顺序以减少温度/负载漂移。原始生成日志保留 finish reason 与时长。

报告 mean/P50/P95 的答案延迟和完整响应延迟；高并发吞吐只作为可选扩展，不能混入单请求体验指标。

### 8.3 统计与结论

三个训练种子报告均值、标准差与逐种子结果。主比较使用同题配对结果；置信区间按独立图表聚类 bootstrap，避免把同图多题当成完全独立样本。延迟重复测量与训练种子变化分别报告。

报告准确率—实际延迟散点图和训练中 accuracy/skip ratio/token 曲线。不能用 token 降幅替代延迟降幅；不能因有少量成功案例就宣称学会通用难度识别。失败分析至少覆盖读数、图例匹配、计算、单位、格式、无效长推理和跳过后答错。

## 9. 建议代码结构

以下均为后续拟创建内容；当前只交付本 PLAN.md。

```text
PLAN.md
README.md
pyproject.toml
configs/
  data.yaml
  generate_rationales.yaml
  sft_full.yaml
  sft_drop50.yaml
  grpo_full.yaml
  grpo_ton.yaml
  eval.yaml
prompts/
  reverse_thinking.txt
  adaptive.txt
  direct.txt
  cot.txt
src/ton_chartqa/
  data/                 # 下载、审计、抽样、导出、dropout
  generation/           # 离线推理生成、重试与质量记录
  rewards/              # 解析器、ChartQA scorer、EasyR1 适配
  evaluation/           # 准确率、流式延迟、聚类统计
scripts/
  prepare_data.py
  generate_rationales.py
  build_sft.py
  export_grpo.py
  train_sft.sh
  merge_sft.py
  train_grpo.sh
  evaluate.py
  benchmark_latency.py
  summarize_results.py
tests/
  test_data_isolation.py
  test_thought_dropout.py
  test_answer_parser.py
  test_chartqa_reward.py
  test_template_and_masks.py
  test_streaming_latency.py
reports/
  environment.md
  data_audit.md
  rationale_quality.md
  single_gpu_profile.md
  decisions.md
  results.md
data/                   # 不提交大文件，manifest 可提交
outputs/                # 日志、轨迹、检查点及统计
```

配置和 manifest 都有版本/hash。脚本支持显式 data/model/output 根目录、seed、resume；不硬编码个人机器路径。GPU 训练日志先落本地，外部跟踪服务不作为执行前提。

## 10. 执行阶段、依赖与验收

### P0：环境和单卡闭环

P0 使用少量原始训练题和临时响应检查基础设施；所需的最小 parser/reward 随 smoke test 实现。P1 再完成完整数据与 verifier 审计，P1 冻结版本后重新运行 P0 的奖励相关检查，随后才进入正式训练。

- [ ] 建立仓库结构、依赖记录、运行配置和本地日志。
- [ ] 检查 A800/主机资源，固定模型与框架版本。
- [ ] 检查 Qwen3-VL chat template、图片处理和控制标签 token。
- [ ] 用 32 条真实训练样本验证 SFT 前向/反向及空 think loss mask。
- [ ] 合并测试 adapter，校验导出和初始策略等价。
- [ ] 用临时检查点完成至少 5 次 GRPO 采样—打分—更新闭环，G=4。
- [ ] 测试 checkpoint 保存/恢复、LoRA 权重到 rollout 的同步、reference 冻结。
- [ ] 记录 GPU 峰值、主机内存、输入长度、rollout 吞吐及实际 batch 语义。

验收：单卡闭环可恢复，无 OOM/NaN/错分组/错误 mask；reward 与 parser 单元测试通过。这里的 toy/smoke 结果不作为正式模型成绩。若失败，先修实现和配置，不扩大训练或引入新方法。

### P1：数据与 verifier 冻结

- [ ] 下载并固定 ChartQA revision，执行重复与划分审计。
- [ ] 建立候选清单、开发/延迟子集清单和数据 schema。
- [ ] 核对 ChartQA 参考 evaluator，完成边界测试与奖励适配。
- [ ] 基座在 dev_quick 上跑 B0/B1/B2，记录真实行为、截断及时间。

验收：数据来源与划分清楚，verifier 可追踪，无训练输入答案泄漏。基座模式差异很小也保留结果，不借此选择有利测试题。

### P2：冷启动数据

- [ ] 小批量生成 100 条，检查 reverse thinking 提示与数据质量。
- [ ] 生成候选、记录重试，完成至少 100 条人工抽检并达到质量门槛。
- [ ] 冻结 2,000 条母版，导出 full/drop50 两套 SFT 和同题 GRPO 文件。
- [ ] 检查两套数据逐条配对、dropout 种子、空推理比例和完整标签。

验收：同题同答案、只有随机推理清空差异；所有生成配置和质量剔除均有记录。

### P3：配对 SFT

- [ ] 先跑 seed=42 的 full 与 drop50，比较 S0/S1。
- [ ] 验证模板和 loss 实际接收空 think；记录未触发/全部触发情况。
- [ ] 导出各自 merged SFT 模型与 processor，验证合并一致性。

验收：两分支训练和导出成功；模型不会因模板或 parser 被强制只能输出一种行为。不要求观测跳过率恰好 50%。

### P4：配对 GRPO

- [ ] 从各自 SFT 检查点训练 R0/R1，使用相同问题池、G、更新预算和 reward。
- [ ] 监测 accuracy、format、skip、长度、KL、梯度、同奖励组、截断、显存与耗时。
- [ ] 进行断点恢复检查并完成 seed=42 初轮。
- [ ] 冻结正式配置；若 pilot 有改动，按正式配置重新跑相应分支。
- [ ] 补齐 3 个种子的 SFT 与 GRPO 配对结果。

验收：所有正式运行有完整轨迹/配置/检查点，不能仅保存最好的一次。没有跳过或效果退化时仍进入失败分析，不私自加入奖励改进。

### P5：最终评测与交付

- [ ] 按预定 dev 规则选择检查点，冻结测试配置。
- [ ] 完成 B0/B1/B2/S0/S1/R0/R1 的完整 ChartQA test 评测。
- [ ] 完成单卡同条件延迟测试和统计。
- [ ] 输出总体、来源分组、每种子、误差类型、准确率—延迟对照。
- [ ] 整理 README、运行脚本、版本/数据 manifest、结果表、失败案例和可用模型产物。

验收：他人能从固定数据与配置重跑流程；结果包含负面发现和限制；明确区分 TON 方法、项目工程适配与实际实测结果。

## 11. 资源估算与故障决策

当前不承诺固定 GPU 小时。P0/P2 先测每题生成时长、每个 SFT step、每个 GRPO 采样迭代、每题评测时长，再据此估算数据生成、6 次 SFT、6 次 GRPO、验证和测试成本。每阶段写入预计/实际 GPU 小时，避免用论文多卡耗时推算单卡耗时。

| 现象 | 排查与处理 |
| --- | --- |
| SFT 后从不跳过 | 检查空 think 是否保留、mask、模板预填和提示一致性；实现正确则作为结果记录 |
| 全部跳过且准确率下降 | 查 verifier、采样、数据质量和训练稳定性；不加动作均衡或长度奖励 |
| 格式分上升但答案不改善 | 分开查看 reward 分量与全同奖励组；验证解析器没有漏洞 |
| 很多 full 推理被截断 | 在 dev/pilot 检查上限，统一调整所有分支并冻结，不让截断构成效率优势 |
| 图中细节读取明显受损 | 检查 resize 和 processor；必要时统一提高分辨率并重建基线 |
| reward 稳定但无有效更新 | 检查组内奖励方差、梯度、LoRA trainable 参数、rollout 权重版本与 KL |
| token 少但实际不快 | 分解视觉编码、prefill、解码与服务开销；报告无真实加速，不用 token 代替结论 |
| LoRA 跨框架不兼容 | 核对版本、模块命名、合并与 processor；必要时只替换训练适配层并记录 |

不因为结果不理想自动扩大到全部训练集、加入新数据域、改为全参数训练或恢复已否决的改进。此类范围变化另行讨论。

## 12. 参考资料与待实施核验项

链接核对日期为 2026-09-17；实施时必须把滚动的 main/模型页面解析成固定 revision。引用用于解释来源，本文其余数值若未注明论文出处均为本项目工程初始值。

- [S1：TON 论文，方法、数据构造与任务奖励](https://arxiv.org/html/2505.16854v3)
- [S2：TON 官方仓库](https://github.com/kokolerk/TON)
- [S3：ChartQA 整理版本与划分](https://huggingface.co/datasets/HuggingFaceM4/ChartQA/blob/main/README.md)
- [S4：ChartQA 官方数据说明与底层表格](https://github.com/vis-nlp/ChartQA)
- [S5：ChartQA 官方模型评价代码入口](https://github.com/vis-nlp/ChartQA/blob/main/Models/VL-T5/src/vqa_data.py)
- [S6：EasyR1 Qwen3-VL-4B LoRA GRPO 示例](https://github.com/hiyouga/EasyR1/blob/main/examples/qwen3_vl_4b_geo3k_grpo_lora.sh)
- [S7：EasyR1 配置与视觉 LoRA 排除项](https://github.com/hiyouga/EasyR1/blob/main/examples/config.yaml)
- [S8：EasyR1 reward 回调示例](https://github.com/hiyouga/EasyR1/blob/main/examples/reward_function/math.py)
- [S9：Qwen3-VL-4B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [S10：LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

待 P0/P1 实测锁定的事项：精确依赖版本、单卡 rollout/训练内存生命周期、有效 YAML 字段与 batch 单位、图像 token 分布、LoRA 合并数值容差、完整 evaluator 边界行为、请求超时上限和各阶段 GPU 小时。它们不是未决的研究方向，而是实施前必须完成的工程检查；目前没有已实测的吞吐或显存保证。
