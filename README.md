# bisight-rl

ChartQA 上的 TON 数据构造。研究设定见 [PLAN.md](PLAN.md)，详细数据规则见 [DATA_PLAN.md](DATA_PLAN.md)。

## 当前已完成

- Git 初始化；本地 Conda 环境 `bisight-rl`（Python 3.11）。
- 原始 ChartQA 已下载并校验：train 28,299 / val 1,920 / test 2,500。
- 数据 revision：`b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`。
- 生成模型：`Qwen/Qwen3-VL-4B-Instruct`，revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`。
- 规范化、精确重复/近重复审计、冲突标签隔离和分层抽样已完成。
- 28 项测试通过；20,220 张相关图片通过 hash/解码检查；32 条真实样本通过 Qwen processor/空 think 模板检查，大图像素上限实测为 1,048,576。
- 合格训练候选 27,223 题；首批 2,000 题来自 1,929 张图，human 523 / machine 1,477。
- pilot 100 题已固定，覆盖 human/machine、数值/文本/整体多项答案。
- 有效调参 val 1,897 题，dev_quick 256 题；原始 val 和标准 test 均保留。

**目前推理母版数量为 0。** 本地只完成前置处理；在 A800 生成并完成审查后，才能发布 2,000 条母版。`train_initial_2000.jsonl` 是初始题目清单，不是已生成、已质检的解释集；最终 core 会因失败重试与同层补样而变化。

## 数据目录

| 路径 | 内容 |
| --- | --- |
| `data/manifests/source.json` | 固定来源、数据/模型 revision、原始文件 hash |
| `data/manifests/build.json` | 清洗和抽样配置、配额、产物 hash、统计 |
| `data/raw/` | 原始 Parquet 快照，本地保留，不随 A800 包传输 |
| `data/images/` | 原分辨率 RGB 图像，按解码像素 hash 命名 |
| `data/normalized/` | 原始 train/val/test 的标准化记录 |
| `data/candidates/train_candidates.jsonl` | 全量 27,223 题固定候选顺序，含备用题 |
| `data/candidates/train_initial_2000.jsonl` | 初始 2,000 题清单 |
| `data/candidates/pilot_100.jsonl` | 100 题试生成清单 |
| `data/reviews/` | 跨划分图表对、训练排除原因、数据异常 |
| `data/rationales/v1/` | 首轮 pilot 的格式失败诊断，仅保留审计，不继续写入 |
| `data/rationales/v2/` | 第二轮 pilot 诊断；部分响应缺少 `</think>`，不继续写入 |
| `data/rationales/v3/` | 自动规范化缺失闭合标签后的生成结果、逐次尝试和人工审查记录 |
| `data/processed/v1/rationale_master.jsonl` | 最终审查通过后才创建的母版 |
| `reports/data_audit.md` | 实际数据统计 |

近重复规则：64-bit pHash 距离≤6 先召回，再将两图缩放到 128×128，比对 RGB 平均绝对差≤0.01（归一化至 0–1），以及任一通道差>32 的像素占比≤0.04。满足条件的非精确重复仍标为“待确认近重复”并隔离。规则及阈值已固定在 `configs/data.yaml`，不依据模型正确率选择。此启发式不能保证发现所有重复，也可能隔离不同但相似的图表。

精确图像重复 16 对；疑似近重复 651 对。训练侧隔离涉及 549 张图的 772 道题，另去掉 165 条重复问答和 139 条冲突标签记录。全量 pHash 召回结果保留，因此 `cross_split_pairs.jsonl` 大于最终隔离对列表。相似模板被标为未达到像素阈值时，不声称经过逐项人工确认。

## 1. 将准备包上传到 A800

本地已提供 `outputs/bisight-rl-a800.tar.gz`，包含代码、提示、固定清单、审计记录和图片，不含模型权重、Git 历史或原始 Parquet。修改代码后可重新执行 `python scripts/package_a800.py` 打包。

以下 SSH 别名 `a800` 需替换为自己的实际连接方式；若使用非默认端口，对 ssh/scp/rsync 配置对应端口。

在本地项目根目录执行：

```bash
conda activate bisight-rl
ssh a800 'mkdir -p ~/work'
scp outputs/bisight-rl-a800.tar.gz a800:~/work/
```

在 A800 执行：

```bash
cd ~/work
tar -xzf bisight-rl-a800.tar.gz
cd bisight-rl
conda create -n bisight-rl python=3.11 pip -y
conda activate bisight-rl
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-generation.txt
python -m pip install -e '.[dev]'
python -m pytest -q
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print("BF16:", torch.cuda.is_bf16_supported())'
mkdir -p outputs
python -m pip freeze > outputs/requirements-a800.freeze.txt
python scripts/validate_dataset.py --verify-images
python scripts/check_processor.py
```

如已有同名环境，直接激活并安装所需依赖。CUDA wheel 须与服务器驱动兼容；安装示例来自 [PyTorch 官方版本页](https://pytorch.org/get-started/previous-versions/)。本地未执行 CUDA 前向，A800 的可用性以上述检查和 pilot 为准。

当前生成使用 Transformers BF16、batch=1、SDPA，方便单卡运行和逐题恢复；无需先安装 vLLM。没有量化或外部教师模型。首次生成会从 Hugging Face 下载固定 revision 的模型权重。

## 2. 先生成 100 条 pilot

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_rationales.py --stage pilot --run-dir data/rationales/v3 --limit 5 --dry-run
CUDA_VISIBLE_DEVICES=0 python -u scripts/generate_rationales.py --stage pilot --run-dir data/rationales/v3 --limit 5
# 检查 5 条结果格式后，复用它们并继续完成全部 100 条
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u scripts/generate_rationales.py --stage pilot --run-dir data/rationales/v3 2>&1 | tee outputs/pilot-v3.log
python scripts/review_rationales.py --stage pilot --action export --run-dir data/rationales/v3
```

中断后重复同一生成命令即可。每次尝试单独原子写入文件；已经自动通过的题不会重复采样，每题最多 3 次尝试。OOM 或网络等基础设施异常会停止并保留已有结果，不混作解释质量失败。

生成时标准答案只加入离线教师提示。保存 raw response、seed、finish reason、token 数、耗时、格式/答案/可解析算术检查；自动检查不证明视觉解释正确。最终母版的答案使用原始标签，解释结论必须在数值等价或文本规范化意义上与参考一致，不能仅因落在 5% 容差内就强接标准答案。

生成目录中：

- `attempts/*.json`：所有生成尝试及自动检查记录。
- `pilot_candidate_master.jsonl`：自动通过的候选，可能不足 100。
- `pilot_exhausted.jsonl`：自动检查失败的题目与原因。
- `pilot_review.html`、`pilot_review.csv`：全部 100 题的人工审查材料。

在本地项目根目录下载：

```bash
rsync -av a800:~/work/bisight-rl/data/rationales/v3/ data/rationales/v3/
```

打开 `data/rationales/v3/pilot_review.html`，逐项填写同目录 CSV 的 `decision=pass/reject`、`hint_leak=yes/no`、`reviewer`、`notes`。不确定项填写 reject 并注明原因；不得批量填充虚构的人工审查结果。HTML 使用本地 `data/images/` 的相对路径。

把填写好的 CSV 传回 A800：

```bash
scp data/rationales/v3/pilot_review.csv a800:~/work/bisight-rl/data/rationales/v3/
```

A800 上导入：

```bash
python scripts/review_rationales.py --stage pilot --action import --run-dir data/rationales/v3
```

pilot 是完整性和流程就绪检查，不要求 100 条全部通过：100 条必须全部完成审查，至少存在一条可用响应，且任何标为 pass 的响应都不得有答案提示泄漏。所有 reject（包括原题/标签错误、生成错误及泄漏响应）均按响应 hash 记录并排除，不进入全量候选母版；reject 比例和泄漏数量作为诊断指标报告，不阻断对剩余候选的全量生成。若错误呈系统性或通过率低到不值得继续，应先修订提示；修改提示或生成配置须另用一个全新的 run-dir（例如 `data/rationales/v4`）。

## 3. 生成 2,000 条候选母版

pilot 审查通过后，在 A800 执行：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u scripts/generate_rationales.py --stage full --run-dir data/rationales/v3 2>&1 | tee outputs/full-v3.log
python scripts/review_rationales.py --stage final --action export --run-dir data/rationales/v3
```

程序按照冻结候选顺序与来源/答案类型配额生成，复用同一配置下 pilot 中合格的尝试，并跳过人工判定 reject 的响应。失败或被拒后从同层候选顺序继续补样，直到自动通过且不属于已知 reject 的候选达到 2,000；不会把所有 27,223 题无条件生成一遍。如果无法凑齐，则显式失败并报告，不静默降低质量门槛。

`full_candidate_master.jsonl` 此时仍是待最终抽检的候选母版。`final_review.html` / CSV 固定抽取 100 条，覆盖来源、答案类型、推理长度和重试状态。

生成全部结果后在本地下载：

```bash
rsync -av a800:~/work/bisight-rl/data/rationales/v3/ data/rationales/v3/
rsync -av a800:~/work/bisight-rl/outputs/ outputs/a800/
```

## 4. 回到本地完成最终质检和发布

填写 `final_review.csv` 后：

```bash
conda activate bisight-rl
python scripts/review_rationales.py --stage final --action import --run-dir data/rationales/v3
python scripts/finalize_master.py --run-dir data/rationales/v3
```

只有最终抽检通过且候选母版中不存在已知 reject 时，才生成 `data/processed/v1/rationale_master.jsonl` 和带 hash 的发布 manifest。未经逐条人工检查的记录明确标为 `auto_checked_in_sample_audited_release`，不会声称全部 2,000 条均经人工验证。

若最终抽检发现错例：先导入以保存 reject 记录；把 `human_reviews.json` 同步回 A800，重新运行 full，让已知 reject 被排除并从同层候选中补足到 2,000。候选变化后，将旧的 `final_review*` 和 `final_gate.json` 移至归档目录，再重新导出抽检；若发现系统性错误或抽检错误比例>5%，应修订提示、创建新 run-dir 重新生成，不反复换抽检名单追求通过。pilot 中旧的 pass 后来被改为 reject 时，重新导入 pilot 审查记录即可更新排除集合与门禁摘要。

本轮交付到推理母版为止；full/drop50 SFT、GRPO 框架数据导出及实际训练 loss-mask 检查按总计划在后续完成。

## 本地复现前置处理

```bash
conda activate bisight-rl
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/validate_dataset.py --verify-images
python scripts/check_processor.py
python -m pytest -q
```

`prepare_data.py` 重跑时复用已校验的规范化数据。配置变化必须显式 `--rebuild`，旧 build manifest 自动归档；已有生成 run 的契约随之失效，须新建 run-dir。默认 data 根目录为项目下 `data/`，各数据脚本支持 `--data-root`，生成/审查脚本支持 `--run-dir`。

## 评分及限制

`src/bisight_rl/quality.py` 实现严格响应结构解析、答案一致性、受限算术检查和 relaxed scorer 边界测试。relaxed scorer 参考 [Pix2Struct 实现](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L74)，保留零值回退字符串比较、百分数换算及非数值忽略大小写行为，额外明确拒绝 NaN/Inf。外层空白由答案块解析器去除；不删除千分位或猜测单位。

生成质检中的数值等价检查比 relaxed scorer 更严格，避免错误解释仅靠 5% 容差过关；后者不能证明推理或图像依据正确。完整 SFT/RL verifier 接入、真实 GPU forward、训练 loss mask、延迟子集和模型效果评测尚未执行。

ChartQA 数据卡标注 GPL-3.0，原始数据与署名信息随 source manifest 保留；图片/完整数据不会随代码默认提交。数据源：[HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)。
