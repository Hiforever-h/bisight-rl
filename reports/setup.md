# 本地前置处理完成状态

日期：2026-09-17。

- 项目：`/Users/hiforever/Documents/AiProject/bisight-rl`
- Git：已初始化；未创建远程仓库、未推送，工作文件尚未提交。
- Conda：`/opt/miniconda3/envs/bisight-rl`，Python 3.11.16。
- 平台：macOS / arm64。仅使用 CPU 做数据和 processor 检查。
- 本地依赖：`requirements-local.lock.txt`；GPU 环境安装见 README，不能将 macOS 环境导出直接当成 CUDA 环境。
- 数据及模型固定版本：见 `data/manifests/source.json`。
- 自动测试：28 项通过，覆盖图表隔离、冲突/重复问答、分层容量、pHash 检索、评分边界、格式/算术和人工发布门槛。
- 候选完整性：通过所有产物 hash、行数、配额、split 隔离检查；20,220 张候选/有效验证/测试涉及的唯一图片通过文件 hash 和解码校验。
- Qwen processor：32 条真实训练题通过；输入长度 271–862 token；图片占位与空 think 保留；2048×2048 边界图缩放后为 1,048,576 像素。
- 生成命令 dry-run：通过；未加载模型权重。
- 推理生成数量：0。人工推理审查数量：0。最终推理母版：尚未创建。

下一步：用户在 A800 运行 pilot，下载并人工审查 100 条；通过后执行 full，生成 2,000 条自动检查候选，再回本地抽检和冻结母版。运行与同步命令已写入 README。
