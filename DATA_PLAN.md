# ChartQA 训练集构造计划

日期：2026-09-17。状态：方案，尚未下载数据、生成推理或启动训练。

本文细化 `PLAN.md` 第 3–5 节与 P1/P2；保留既定的 TON 方法、2,000 题规模、Qwen3-VL-4B-Instruct 基座和单 A800 约束。本文新增的候选批量、审查流程等均为工程建议，不是论文参数。当前项目只有总计划，数据流水线尚待实现。

## 1. 交付目标与数据流

构建一个可追溯的 2,000 题训练核心集，从同一份推理母版导出两个 SFT 版本及一个供两个 GRPO 分支共用的问题池：

```text
固定版本的 ChartQA train / val / test
  → 原始记录归档、图片与答案语义检查
  → 按图表审计跨划分泄漏、训练问答去重
  → 冻结训练候选顺序与开发/测试清单
  → 100 题 reverse-thinking 试生成与审查
  → 按冻结顺序批量生成、质检、重试、补样
  → 2,000 题 train_core + rationale_master
      ├─ sft_full：2,000 条完整推理
      ├─ sft_drop50：相同 2,000 条，随机清空约一半推理
      └─ grpo_train：相同 2,000 条，仅问题和图片进入策略输入
```

不将一题的 full/dropout 两个版本拼成 4,000 条用于同一个主实验分支。三组训练种子共享同一数据版本；数据种子与训练种子分离。

| 产物 | 数量 | 用途 |
| --- | --- | --- |
| train_candidates | 审计通过的训练池，保存完整候选顺序 | 生成及补样 |
| train_core / rationale_master | 各 2,000 | 固定题目池 / 推理母版 |
| sft_full / sft_drop50 | 各 2,000 | 成对 SFT 对照 |
| grpo_train | 2,000 | 两个 GRPO 分支共用 |
| dev_full_raw / dev_full | 原 val / 去除与 test 重叠图表后的调参视图 | 保留原始划分并隔离泄漏 |
| dev_quick | 从可调参的 dev_full 固定抽取 256 | 工程监测 |
| test_full | 原 test 全部 2,500，以实际下载核验为准 | 设置冻结后的最终评测 |
| latency_test | test 的固定 256 题子集 | 后续延迟测试 |

## 2. 数据来源与版本冻结

主来源为 `HuggingFaceM4/ChartQA`。当前数据卡列出 train=28,299、val=1,920、test=2,500，字段为 `image`、`query`、`label: list[string]`、`human_or_machine`，类别映射为 0=human、1=machine；实施时读取实际 features 核验，不能仅硬编码映射。[数据卡](https://huggingface.co/datasets/HuggingFaceM4/ChartQA/blob/main/README.md)

下载前将数据集、模型、tokenizer/processor 的 revision 解析为完整 commit SHA。manifest 保存来源 URL、revision、下载日期、文件校验和、软件版本及数据卡的许可证/署名信息。原始文件只读保留；派生文件另放目录，不覆盖原始标签。

官方完整版的图片、表格和 annotations 仅辅助来源核对与离线审查。连接时先确认图像对应关系，不能用问题文本模糊匹配后直接认定是同一图。官方说明 annotations 可能含噪声，因此辅助标注也不能替代目视判断。[官方说明](https://github.com/vis-nlp/ChartQA)

保留英文问题与标签，第一版不翻译、不改写问题、不增加合成问答、不加入外部推理数据。默认不向代码仓库提交图片、完整数据或模型权重。

## 3. 内部记录与答案语义

内部格式采用 JSONL，一行一题；图片独立存放，同图多题共享图片。字段至少包括：

```json
{
  "id": "chartqa:<dataset_revision>:train:<row_index>",
  "split": "train",
  "image_id": "<decoded_pixel_sha256>",
  "image_path": "images/<image_id>.png",
  "image_file_sha256": "<file_hash>",
  "image_pixel_sha256": "<pixel_hash>",
  "image_width": 800,
  "image_height": 600,
  "question": "<original query>",
  "answers": ["<original label>"],
  "source": "human",
  "answer_kind": "numeric",
  "answer_semantics": "single",
  "canonical_answer": "<verified training target>",
  "provenance": {
    "dataset": "HuggingFaceM4/ChartQA",
    "revision": "<full commit SHA>",
    "row_index": 0
  }
}
```

示例尺寸和行号仅用于说明 schema，不代表已抽取的真实样本。

原文、保守规范化值、审计结果分开存储。`answer_kind` 取 numeric/text/list，无法判断时暂记 unknown 并进入语义审查；它用于抽样统计，不决定评分规则或是否 dropout。

答案处理规则：

- 不能把 `label` 是列表直接理解成“多个都要输出”，也不能直接理解成“多个任选其一”；先核验真实数据和原始标注。
- 单参考答案直接保留。经确认的多个等价参考答案完整保留，SFT 按固定规则选择一个 canonical answer，评测保留全部参考。
- 如一个标签字符串表达多项答案，保留为整体答案；不按逗号拆成任选其一的多个参考，不自行排序或改写。
- 不将百分号、单位、正负号、年份、千分位进行不可逆清洗；`35`、`35%`、`0.35` 不能统一改写为同一个训练标签。
- 同图同问题出现不同标签时，先核验是否等价；无法确认则隔离整个冲突组，不任取一条，也不悄悄合并参考答案。记录数量与来源分布。
- canonical answer 必须有原始参考依据；不使用模型自己给出的答案替换数据集标签。

## 4. 先审计，再抽样

先读取全部 split 的图片与元数据做一次审计。test 只用于重复审计及预定元数据抽样，不生成解释、不根据 test 表现调整提示或筛选策略。

1. 检查解码、空问题/答案、尺寸、字段类型及引用路径。训练异常记录隔离；验证/测试异常保留在原始清单中，记录失败，不悄悄缩小评测分母。
2. 保存原文件哈希，以及按固定解码规则得到的像素哈希。像素哈希包括尺寸、色彩模式和像素；记录解码库、EXIF/透明度处理规则。ID 不依赖本地绝对路径。
3. 训练内部按“确认的图表身份 + 保守规范化问题”找重复，参考答案一致才自动合并；保留所有来源行映射与确定性的代表行规则。同图不同问题保留，不人为限制每图一题。
4. train 与 val/test 有确认重复图表时，排除训练侧该图全部问题，包括问题文本不同的情况。
5. 感知哈希仅召回近重复候选，例如初始采用 64-bit pHash 距离不超过 6；阈值是待 pilot 核验的工程值。对候选核对图中文字、数值和图例；外观相似的模板不能直接认定泄漏。
6. val/test 有确认图表重叠时保留标准 test，在验证调参视图中排除该图，并记录原始/有效验证数量。未决近重复候选涉及训练或调参时，先隔离相关训练/验证项并留档。

产出 `data_audit.md`、重复组与冲突清单、有效训练池、val 调参排除清单。分别统计原始 train、审计通过池、生成尝试池、最终 core 的来源、答案类型、独立图表数及每图问题数量分布。

## 5. 冻结候选与补样规则

抽样种子沿用 `20260917`。按 `source × answer_kind` 分层；目标配额优先近似原始 train 比例，以最大余数法将配额取整为总计 2,000。审计导致某层不足时记录缺额，以剩余可用容量确定性重新分配，不借助基座正确率调配额。

每层使用固定 RNG/版本打乱一次，保存完整有序 ID 清单；跨层按目标比例确定性穿插。保存原始比例、目标配额、候选顺序、最终顺序及全部选择/排除原因。

- 初始生成预算按约 2,500 个候选估算，相当于假设约 80% 可用；这只是容量估算，不是数据质量预测或硬性生成量。
- 先处理 100 条 pilot；提示冻结后按约 100–200 条一批推进，只生成满足配额所需的候选，避免无条件生成全部 train。
- 任一题最终失败时，优先从同层冻结队列的下一题补入。补样顺序不由完成时间、推理长度或答案正确率决定。
- 只按已定义的数据/解释质量门槛处理，不先自由答题再挑基座答对的题，不设置 Think/NoThink 或难度配额。
- 对 short/medium/long 推理仅做统计与审查覆盖，不作为保留比例或选样标准。上下文超限作为单独技术原因报告。

dev_quick 在隔离后的 val 中按来源与答案类型抽取 256 条。latency_test 按来源及实际学生输入 token 长度分层固定 256 条；输入长度由冻结的 processor 计算，不能依据输出、正确性或跳过行为选择。子集不足时明确记录，不能跨 split 凑数。

## 6. Reverse thinking 生成

用原始 `Qwen/Qwen3-VL-4B-Instruct` 的固定版本离线生成，输入仅为训练图片、原始问题、经核验的标准答案及生成提示。通过答案条件生成推理、随后执行随机 thought dropout，符合总计划采用的 TON 数据构造方法。[TON 方法](https://arxiv.org/html/2505.16854v3)

生成提示与学生提示分文件维护。生成提示初稿要求：依据图片解释所需读数、图例/类别对应及计算或判断；按题目需要展开，不固定步骤数，不设极短长度目标；不能仅以“提示已给出答案”为理由；无法支持参考答案时明确报告不一致，不编造视觉证据。批量前通过 pilot 检查实际指令遵循。

生成响应建议使用同一 `<think>…</think><answer>…</answer>` 结构，以便审查解释与结论是否一致。这里的 answer 只是生成质检信号；正式 SFT 答案来自 canonical answer，不能因为替换成标准答案就判定解释正确。拒答或证据冲突是质检记录，不成为新增训练答案类别。

| 设置 | 初始值 |
| --- | --- |
| precision | BF16 |
| temperature / top_p | 0.7 / 0.9 |
| 每次候选 | 1 |
| 最大生成 token | 2,048，包含所有生成标签 |
| 每题最大尝试次数 | 3，即首次加最多 2 次重试 |
| 建议独立生成种子 | 20260918，按题目 ID 和 attempt 派生并保存 |
| GPU 执行 | A800 单卡，生成与其他 GPU 作业串行 |

图像处理沿用总计划的 min_pixels=65,536、max_pixels=1,048,576，保留原图并由 processor 处理。生成阶段含答案提示，长度可能大于学生输入；两者分别计数，不能拿生成输入长度代替正式输入长度。

每次尝试记录：sample ID、model/processor revision、prompt hash、生成参数、派生 seed、attempt、raw response、finish reason、时间、token 数、失败原因。种子不等于跨硬件/推理引擎逐字复现保证，正式版本以归档响应为准。

只重试格式错误、截断和明确解释错误；不因推理短或直接读数而重试。取按 attempt 排序的第一个合格结果，禁止 best-of-N 挑长推理。基础设施故障单独重跑同一任务，不把它计成解释失败。

## 7. 质量门槛与人工审查

自动检查所有候选：

- 响应未截断，标签结构唯一且闭合，母版 think 非空，answer 非空。
- 无重复控制标签、角色标记或生成提示的明显回显。
- 无“已知标准答案”“根据提供的答案”等泄漏式解释；规则只能发现疑点，不能证明语义完全正确。答案数值正常出现在推理中不算泄漏。
- 可解析算术表达式使用受限解析器复算，检查操作数、计算结果和最终结论；不对任意文本使用 `eval`。不能解析时记为未验证，不能记为通过算术验证。
- 对结论与标准答案的一致性做检查，但明确“答案匹配”不等于“图像依据和中间步骤正确”。
- 记录读数、系列/年份、单位、计算、结论冲突等错误类别；可疑原始标签进入单独数据冲突队列，不强行让模型合理化。

人工审查分两轮：

1. **pilot 100 条**：分层抽取、逐条查看图像与解释；定位提示和格式问题，测通过率、耗时及 token 分布。pilot 引发提示修订时，旧输出归档；进入正式母版的 pilot 题须使用冻结版本重新生成。
2. **冻结前至少 100 条**：从暂定最终 core 重新随机分层抽查，覆盖来源、答案类型、推理长度和重试状态。另审自动规则标记的高风险项，额外审查与随机抽检分别统计。

pilot 用于诊断生成流程和估算可用率，不要求源数据与生成结果全部正确。完成 100 条审查后，所有已知 reject（包括原题/标签错误、生成错误和泄漏响应）按响应 hash 排除；只要仍有可用响应即可继续从冻结候选池全量生成并同层补样。reject 比例与泄漏数量必须报告，但被排除响应中的错误不阻断批量处理。

最终随机抽检沿用总计划：抽检中不得保留答案提示残留，明确错误解释比例不高于 5%。这是样本层面的工程门槛，不能表述为已证明全量错误率小于 5%。明确发现的错误均重试或剔除并补足 2,000 条，不因总体达标就保留已知错例；无法判断的争议项先处理，不算正确。发布前的刷新抽检不得含已知 reject。

若抽检失败，暂停批量发布，定位系统性原因；提示变更则为受影响数据建立新版本并重新生成/检查，随后重新抽检。无需用 dev/test 模型成绩来决定哪批解释更好。

审查表保存 reviewer、时间、sample/attempt ID、结论、错误类型、图像证据及备注。模型辅助标记不能冒充人工签核；人工抽检未完成时，产物只能标记为 candidate。可用底层表格复核数值，但颜色、空间关系、图例仍需看图。

## 8. 评分口径须先明确

核验发现：总计划 S5 指向的 `VQAEvaluator.evaluate_raw` 在当前可见版本中，对 `str(...).strip()` 做精确比较，没有实现 5% 数值容差。不能直接包装它后称为 relaxed accuracy。[代码依据](https://github.com/vis-nlp/ChartQA/blob/main/Models/VL-T5/src/vqa_data.py#L556)

建议将 Pix2Struct 的 `relaxed_correctness` 作为 relaxed 指标的具体参考实现，下载时固定 commit，并保存与 ChartQA 论文定义及原仓库 exact-match 代码的对照说明。该实现对非零数值使用 5% 相对误差、对百分数字符串做换算，零值落入文本比较；这些边界需要忠实记录。[参考实现](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L74)

实施时固定一个 scorer 版本供质检中的结论比较、GRPO 答案奖励和最终评测调用；另报告严格匹配。格式 parser 与答案 scorer 分离，不把 think 中的数字当作答案。明确解析出的答案边界空白策略，记录相对上游实现的任何适配。

测试覆盖：数值容差内外、0/0.0、负数、百分号、千分位、科学计数法、文本大小写/空白、年份、整体多项答案、等价多参考、NaN/Inf、空输出、重复标签与截断。年份被当作数值时可能获得宽松容差，不能私自改规则再报标准指标；额外严格诊断单独呈现。

## 9. 冻结母版与导出

母版在语义质检及学生模板长度检查后冻结。每题保留 core 原始字段、完整 rationale、canonical answer、生成 attempt/config 引用、质量状态、实际输入与输出长度。

full 和 drop50 共用相同 system/user、图片、题目 ID、顺序及 answer。学生提示仅要求按图表回答、允许空 think、最终短答案放入 answer；保留多模态 chat template 的角色及图像处理，不暴露来源类别、表格、答案提示、质检备注或 dropout 标志。

```text
full：
<think>{rationale}</think><answer>{canonical_answer}</answer>

dropout 命中时：
<think>

</think><answer>{canonical_answer}</answer>
```

dropout 用独立 RNG、seed=17，对冻结顺序逐条抽 Bernoulli(0.5)。保存实际 bool mask、RNG 实现及版本；不凑整到 1,000，不分难度设置概率，不跨 epoch/训练种子重抽，也不对已导出文件再次 dropout。JSON 中的 `\n\n` 解码后必须是两个真实换行符。

内部母版与框架导出分离：SFT 可导出 LLaMA-Factory 使用的 messages/images 数据及字段映射；GRPO 可导出 EasyR1 需要的 Parquet 及适配配置。具体框架字段以固定版本实际 loader 为准，不能把示意字段当成已验证接口。

GRPO 中只有统一学生提示、图片和问题送入策略模型；标准答案留在独立 reward 字段或 sidecar，推理母版不导入 rollout prompt。两个分支共享同一个 GRPO 文件。对实际渲染后的请求做字段白名单检查，而非只对输入字符串搜索答案数字。

使用冻结 tokenizer/processor 对全量 core 检查学生输入≤4,096、完整响应≤2,048、SFT 总长度≤6,144，并包含模板开销。禁止自动 truncation 静默裁掉答案。超长样本按同一规则对两个 SFT 分支和 GRPO core 一致处理、补样并披露；若超长成为普遍问题，先统一评估上限，不进行推理压缩。

抽取至少 32 条覆盖 full、空 think、不同答案类型和图像长度的样本，通过真实训练数据加载器检查：图像占位数/路径正确、空 think 未被删掉、answer 与闭合标签未丢失、system/user/图像输入不计 assistant loss、空 think 的闭合标签和答案参与 loss。正式训练仍须通过总计划 P0 的单卡闭环；完整 GRPO smoke 不作为先写数据处理脚本的前提。

## 10. 拟实现文件与恢复机制

```text
configs/data.yaml                      # 来源、规模、种子、审计/长度规则
configs/generate_rationales.yaml       # 固定生成配置
prompts/reverse_thinking.txt           # 仅离线生成使用
prompts/adaptive.txt                   # SFT / GRPO / 自适应评测共用
scripts/prepare_data.py                # 下载、规范化、审计、冻结清单
scripts/generate_rationales.py         # pilot、批量、重试、恢复
scripts/build_sft.py                   # 冻结母版、full/dropout 导出
scripts/export_grpo.py                 # 同题 GRPO 导出
scripts/validate_dataset.py            # 配对、隔离、模板和统计验收
data/manifests/<version>/              # 来源、候选/core/subset ID、hash、mask
data/raw/                             # 原始快照
data/images/                          # 原分辨率图片
data/normalized/                      # 标准化元数据
data/rationales/<run_id>/              # 每次尝试、状态与审核记录
data/processed/<version>/              # 母版及框架导出
reports/data_audit.md
reports/rationale_quality.md
reports/data_release.md
```

所有脚本支持显式 data/model/output 根目录及 resume，路径相对 data root 可迁移。任务键至少由 sample ID、model revision、prompt hash、generation config hash、attempt 组成；改变配置时创建新运行版本，不能用旧缓存冒充新结果。

每题状态区分 pending/generated/needs_review/accepted/retry/rejected；原始响应与状态变更留档。持久化采用原子写入与任务去重，中断后从未完成任务继续。最终排序根据冻结清单，不受批处理/并发完成顺序影响。CPU 可并行处理图片，GPU 作业保持串行。

## 11. 实施顺序与资源安排

| 阶段 | 工作 | 可执行位置 | 完成标志 |
| --- | --- | --- | --- |
| D0 | schema、配置、版本 manifest、答案/parser/scorer 边界规则 | 本地 CPU | 数据与评分契约明确 |
| D1 | 下载、图片/答案审计、分层候选、开发/测试清单 | 本地或服务器 CPU | 数据审计与清单冻结 |
| D2 | 模型/processor 加载、100 条 pilot、人工审查 | A800 + 审查环境 | 提示与视觉/长度配置冻结 |
| D3 | 批量生成、自动质检、重试、补样 | A800 + CPU | 暂定 2,000 条完整母版 |
| D4 | 最终随机抽检、问题修复、重新验收 | 人工 + 必要 GPU 重试 | 质量门槛满足 |
| D5 | full/drop50/GRPO 导出、真实 loader 检查、发布 manifest | CPU，必要时 A800 | 成对文件与发布报告齐全 |

现阶段优先完成 D0–D1；A800 推理环境就绪后推进 D2–D5。无需先启动正式 SFT/GRPO 来决定数据是否合格。总计划中基座 dev 对照与完整 P0 仍保留为后续训练前置检查，不依据其正确率重选 core。

GPU 时间在 pilot 后估算：`后续生成时间 ≈ 预计剩余尝试次数 × pilot 每次平均耗时`；分别记录初次生成、重试和废弃 pilot 的 GPU 小时。批量吞吐、通过率及人工审查时间均实测报告，不承诺未测得的耗时。

## 12. 发布验收

- [ ] 数据集/模型/processor/scorer/prompt/config 版本固定；原始文件与最终产物 hash 齐全。
- [ ] core 恰好 2,000 个唯一问答，仅来自 train；与 val/test 无已确认的重复图表，未决审计项按规则隔离。
- [ ] 所有 core 图像可读、答案语义明确，原始标签完整保留；补样和分布变化可追溯。
- [ ] 2,000 条母版均有完整、非空推理及生成尝试记录；质量状态清楚，不把自动通过称为逐条人工验证。
- [ ] 至少 100 条最终随机人工抽检完成，满足门槛，已知错误已处理；抽检前不能发布为 final。
- [ ] full/drop50 逐条 ID、输入、答案、顺序完全相同；唯一允许差异是命中 mask 的 think 内容。
- [ ] dropout seed=17，实际比例与 mask 已保存；所有训练种子复用该版本。
- [ ] GRPO 与 SFT core 的题目 ID 集完全相同；策略输入无法访问 reward 答案或生成提示。
- [ ] 全量真实模板长度通过；32 条 loader/loss-mask 检查通过，无隐式截断和空 think 删除。
- [ ] parser/scorer 边界测试、跨 split 隔离测试、配对/dropout 测试通过。
- [ ] resume 后不重复收录、不改变冻结顺序；从冻结母版可重复导出同一训练内容与清单。
- [ ] 发布报告列出各层数量、独立图表数、错误/重试/剔除原因、token 分布、通过率、抽检结果与剩余限制。

本轮完成状态以数据产物和验收报告为准；是否提升模型准确率或减少延迟留给后续正式实验验证。
