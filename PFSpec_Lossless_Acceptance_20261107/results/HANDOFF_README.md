# PFSpec 接受能力研究交付

本阶段完成七个所选新机制及规定调整，正式评价与全部输出验证结束。最终新方法为 DFlash 第 2800 步、完整 Draft 概率范围、64 个首 Feature 候选。自然 BOS/Completion 集合接受为 71.76%/66.75%，首 P/F 联合接受为 67.38%/61.46%，每轮推进为 3.3470/3.2276；相对 fusion_long 的两任务增量区间均为正。

实际模型有 71,809,280 个可训练参数，使用约四分之一、二分之一和最后一层的三个 Target 状态，读取 64 个已提交 token，预测两个 masked Feature，当前推理消费第一份预测。模型输入包含已知上一 Feature 和两个因果 Ranker Position 提案，未读取未来 Target 真值。第一、第二 masked 预测共享模型，训练中第二份教师分布来自原提案分支；当前有效方案使用单层接受结构。

| 配置 | 实际值 |
|---|---|
| Target | 原 `gpt_vfront.ckpt`，FP32、TF32 关闭、batch=1 |
| Target temperature / top-p | 0.9 / 0.9 |
| Draft temperature / top-p | 0.9 / 1 |
| 首 Feature 候选 | 63 个确定候选与一次独立剩余抽样，共 64 个 |
| 额外层预测消费 | 第一份 Feature；第二份未参与接受 |
| 完整 Target 树 | 至多 130 个节点，最多输出四个 token |
| 自然输入 | 30 BOS seeds、30 Completion 场景 × 2 seeds，16002 上限 |
| 辅助输入 | 同 30 场景，64/256 前缀与两 seeds，共 120 条，512 上限 |

权重在 `paper06_dflash/training/diffusion_context_w64/selected.pt`，287266879 字节，内部 59 个成员已完整读取。原 Fusion 初始化依赖在 `dependencies/fusion_long/selected.pt`，13560843 字节，21 个内部成员已完整读取。原训练权重的采样字段保留原值，实际 Full64 推理通过 `paper06_dflash/training/diffusion_full_w64/inference.json` 指定 top-p=1；复现时使用这个配置。

服务器实际模型路径为 `/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash/training/diffusion_full_w64/`，其 `selected.pt` 链接到同一步训练权重。算法选择记录 `selection_before_final.json` 于 2026-11-07 01:03:05 UTC 保存，当时未读取正式科学结果。正式数据没有参与训练或权重选择。

完整逐篇记录为 `FEATURE_DRAFT_LITERATURE_STUDY.md`，源码为主工程 `patches/pfspec_lossless_acceptance_20261106/`。七个机制分别为 Draft-OPD、LK Losses、Optimal Multi-draft、Block Verification、DFlash、Acceptance-Aware P/F/P Window、UniVer。跨文献表保留主要方案、科学调整、训练对照、来源区间、原始计数及实际费用。VSD 和 Global Resolution 未深入实施，未赋予负面结论。

交接包的 `results/` 保存权重、配置、逐请求结果、逐篇记录与图，`project/` 保存当前研究源码、旧 Feature 适配源码及共享后端参考源码，`literature/` 保存七篇原始 PDF 与引用依据。包内只保留最终新模型与必要 Fusion 权重；未选中的大权重仍在原服务器，清单见 `handoff_contents.json`。恢复时将 DFlash 权重放入上述 `diffusion_context_w64/selected.pt`，并在 `diffusion_full_w64/` 中保留指向它的 `selected.pt` 链接与实际 `inference.json`。Fusion 初始化继续使用原服务器的 `fusion_long/selected.pt`；包内副本可在原件缺失时恢复该位置。

## 无损依据与实现证据

第一 Position 用确定提案的标准接受与剩余采样。Feature 固定集合 S 来自 Draft 的最高概率 63 个 token，q′ 是移除 S 后归一的完整正概率分布；余下候选独立取自 q′。精确集合耦合保持 Feature 输出分布为原 Target p，集合接受概率为 p(S)+Σ min(p,q′)。首 P/F 联合概率为对应 Position 概率乘该集合接受概率。

选择候选分支后使用其完整条件 Target 分布、状态及 KV；未选中集合时通过实际 token 恢复 Target。Position 拒绝与 EOS 使用原 Target 剩余分布，后续 Feature 继续读取完整实际条件。逐前缀归纳得到完整 Target 条件输出分布。该结论使用论文的精确耦合条件；数值实现证据涵盖已测分支和历史，没有声称枚举所有 GPU 可达历史。

数学枚举、完整分支、所选 KV、下一 Target 调用、短输出与逐轮动作对照保存在研究记录及验证文件中。新数值案例表明单层 130 节点方案满足原要求；DFlash 与 UniVer 的更宽深树在正概率 BOS 状态出现超过 1e-5 的策略总变差，已从有效候选中排除。零 Position 概率的 Completion 差异另列，不作为可达输出失败的依据。没有提高误差要求，也没有舍弃小正质量。

## 复现位置

继续使用服务器 `/root/workspace/GaussianGPT` 及既有环境 `/root/miniconda3/envs/gaussiangpt/bin/python`。原 GaussianGPT 版本为 `031d9160622f73dae69324a25d656ce4eb086524`；既有共享后端位于 `/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04`，旧 Feature 研究依赖位于 `/root/autodl-tmp/pfspec_feature_states_20261105`。本次源码位于 `/root/autodl-tmp/pfspec_lossless_acceptance_20261106/source`。

Target、VQ、Ranker 及原项目运行依赖复用原服务器位置。大型训练状态与未选中的训练权重仍保留在服务器的各篇结果目录，路径见各篇记录；最终新方法权重、必要 Fusion 初始化依赖、配置和正式 token 输出保存到本机。默认 Target 与 W1 配置保持原值。

在一个新的结果目录进行短复现：

```bash
/root/miniconda3/envs/gaussiangpt/bin/python /root/autodl-tmp/pfspec_lossless_acceptance_20261106/source/evaluate_final.py \
  --root /root/autodl-tmp/pfspec_lossless_acceptance_20261106/formal_acceptance_01 \
  --methods diffusion_full_w64 --capacity 32768 --expected-gpu 5090 --validate \
  --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json \
  --out /root/autodl-tmp/pfspec_lossless_acceptance_20261106/reproduction_full_validation_01
```

正式来源、种子与完整预算在 `natural_complete_inputs.json`、`capped_inputs.json` 中。完整生成入口为 `run_formal_references.sh`、`run_formal_best.sh`；统计入口为 `report_final.py`。重复执行时使用新的输出目录。

## 下一阶段可用结果

单 Feature overlap、候选集合接受和实际推进分别统计；64 候选的增量包含候选覆盖，不能解释为单次预测准确度。实际费用包括生成时间、首次运行创建、Target 调用与 token 数、候选实例和峰值显存；可靠的独立 Draft 提案时间尚未得到。当前研究没有开展算子优化，也没有修改论文。

后续低时延工作可以复用第 2800 步模型、130 节点接受结构及已测条件概率，先以实际候选与 Target 工作量定位费用。更宽深树需要在原采样策略下解决已发现的可达数值差异，不能直接将其数学推进上限当作有效部署结果。


## 完整正式结果

| 任务与指标 | W1 | fusion_long | DFlash Full64 |
|---|---:|---:|---:|
| BOS 首 Feature 条件接受 (%) | 21.02 [18.52, 23.47] | 27.63 [25.23, 30.02] | 71.76 [68.35, 75.06] |
| BOS 首 P/F 联合接受 (%) | 19.63 [17.20, 22.02] | 25.94 [23.52, 28.38] | 67.38 [63.76, 70.98] |
| BOS 每轮推进 | 2.3921 [2.3436, 2.4399] | 2.5183 [2.4699, 2.5671] | 3.3470 [3.2744, 3.4188] |
| BOS token/s | 229.77 [223.45, 236.47] | 240.74 [231.37, 250.61] | 38.92 [36.14, 41.99] |
| Completion 首 Feature 条件接受 (%) | 24.49 [22.39, 26.82] | 26.92 [24.71, 29.46] | 66.75 [63.57, 69.87] |
| Completion 首 P/F 联合接受 (%) | 22.57 [20.57, 24.81] | 24.84 [22.68, 27.32] | 61.46 [57.88, 64.96] |
| Completion 每轮推进 | 2.4499 [2.4102, 2.4944] | 2.4956 [2.4526, 2.5453] | 3.2276 [3.1563, 3.2970] |
| Completion token/s | 258.06 [248.30, 268.82] | 258.03 [247.47, 269.87] | 48.33 [43.51, 53.46] |

三方法共 270 次自然请求全部 EOS，360 次定长输出也完成；630 份 token 均逐文件读取，输入、长度与 EOS 一致。正式 W1 共同历史 11448 个状态、三方法 34344 行，联合概率为 DFlash BOS 61.16%、Completion 63.07%。单次 Feature overlap 为 23.65%/27.01%，fusion_long 为 23.54%/28.88%，主要增量来自候选覆盖与精确耦合。来源区间、完整原始计数、计算费用和未定义条件行见主研究文档与 `formal_acceptance_01/`。

实际 DFlash 速度低于 fusion_long；未宣称低时延收益。所有 GPU 可达历史尚未逐项枚举，证据范围见无损依据。原 Target、默认 W1、论文和运行服务未因本研究切换。
