# 数据构造实施记录

日期：2026-09-17。

1. 项目目录已由用户更名为 `bisight-rl`，环境名使用用户指定的 `bisight-rl`。原有 PLAN.md / DATA_PLAN.md 内容保留。
2. 用户明确由自己在 A800 执行 think 生成；本轮只执行本地前置处理、processor 检查和脚本测试，没有下载模型权重、生成推理或执行 GPU 训练。
3. 单独 pHash≤6 在 ChartQA 中召回 124,082 个跨划分图表对（包括精确重复），存在大量相同图表模板导致的假阳性；初版过度隔离的 build manifest 已归档，不用于当前清单。目视比较了同图重新编码、不同年份同模板和不同主题同模板的例子后，增加固定的归一化像素复核：128×128、MAE≤0.01、任一通道差>32 的像素占比≤0.04。未按模型效果调阈值。当前保守隔离 16 对像素重复和 651 对未决近重复，仍可能误排或漏检，不声称完成逐项人工核验。
4. 模型生成采用原始 Qwen3-VL-4B-Instruct、Transformers、BF16、batch=1、SDPA，以减少部署依赖。采样参数仍为 temperature=0.7、top_p=0.9、max_new_tokens=2048；每题至多 3 次尝试。A800 真实 forward/显存/速度待用户执行 pilot。
5. 自动质量检查要求生成结论与 canonical answer 数值等价或文本一致；5% relaxed scorer 只记录诊断，不作为强行替换结论后保留推理的依据。任何自动检查均不能证明视觉依据正确。算术只检查受限语法中可识别的表达式，未识别记作 not_verifiable。
6. 最终人工抽检保留至少 100 条要求。明确发现的错例必须修复或补样，自动检查和真实人工审查状态分开保存。review CSV 不自动填写；未达门槛只保留 candidate，不发布 final master。
7. Transformers 4.57.1 的 processor 具有 size 与 min/max_pixels 两套字段。代码显式同步设置两者，并对 2048×2048 合成边界图确认处理面积不超过 1,048,576；32 条真实训练题同时验证了图片占位和空 think 的编码/解码保留。
8. 100 条 pilot 对稀少的 human:list 层至少保留 1 条，以覆盖整体多项答案；正式 2,000 条配额仍按原 train 比例分配，未按难度或模型答对情况选样。
9. 运行契约包括数据构建 hash、模型 revision、生成脚本/质量模块 hash、提示 hash、参数与运行环境。相同准备任务重跑不改写 build manifest；变更代码/配置须显式 rebuild 并失效旧生成 run。
10. 本轮不生成 SFT/dropout 或 GRPO 框架数据；真实训练 loader 的 loss-mask、完整 GPU 闭环和按处理后输入长度分层的 latency_test 仍是后续工作。
