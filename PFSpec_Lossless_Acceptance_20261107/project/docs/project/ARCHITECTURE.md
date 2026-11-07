# GaussianGPT 架构说明

## 2026-11-06 文献接受率研究的 DFlash 接入

研究源码位于 `patches/pfspec_lossless_acceptance_20261106/`，服务器为 `/root/autodl-tmp/pfspec_lossless_acceptance_20261106/source/`。共享 Target、Ranker、三层状态读取、词表及生产入口保持原位置。

DFlash Draft 有五层、1024 维注意力，共 71809280 参数。每层从最近 64 个已提交 Target token 的三层状态读取 key/value；已知 F0、Ranker P1、掩码 F1、Ranker P2、掩码 F2 构成五个查询位置。一次前向得到两个完整 Feature 分布，Target embedding 和输出层共享且不训练。在线上下文只接入已提交路径，输出尚未消费的最后 Feature 仍作为下一轮输入。

首版及第一调整只用首 Feature，沿用 Greedy 固定 63 个候选加一次剩余分布采样，共 64 候选、130 个完整 Target 节点。第二调整用四个共享次 Feature，组合为 256 条 Feature 路径，共 642 节点。首、次 Feature 依次使用实际 p/q 接受和完整差额采样；每个后续 Target 分布包含此前实际 Position 与 Feature。最多输出六个 token，EOS、预算和恢复前向全部计入。

642 节点的图执行曾改变一个 Top-p 有效范围，当前只对这棵树使用普通执行。原 130 节点树与普通 Target 调用继续复用已有方式。原始失败结果保留；普通两前缀的 6152 项状态、概率与 KV 对照已通过，完整解码继续。`long_metrics.py` 单独计算首 P/F 与第二 Feature 指标，第二 Feature 的验证次数不进入首 Feature 分母。全部追加计算费用逐请求保存。

## 2026-11-06 文献驱动接受率研究

新增研究源码 `patches/pfspec_lossless_acceptance_20261106/`。真实轮起点记录器采集已有 Target 三层状态、实际 P/F 条件分布及接受结果；它不增加 Target 计算，也不消耗解码随机数。Draft-OPD 适配复用原网络；LK 另通过权重中的 `draft_top_p` 选择实际 Draft 分布。Target 继续使用原 0.9/0.9，单候选旧方法继续使用原 W1；新集合方法复用原 Target 树计算与 KV 提交，并按原文选择实际候选分布。块方法只联合首个 P/F 对，再复用所选真实分支，其完整输出分布有独立数值证据。当前默认模型仍保留，研究详见[逐篇记录](FEATURE_DRAFT_LITERATURE_STUDY.md)。

2026-11-05 新增研究代码 `patches/pfspec_feature_states_20261105/`：通过 `StateTargetGraph` 将已消费 Position 的少量中间层状态纳入原 Target 图计算输出，再交给位置条件 Feature Draft。它沿用既有四节点 W1 验证和状态选择，不修改原 Target 权重或默认入口。当前在 5090 开发，尚未建立自然生成收益；详见 [研究记录](PFSPEC_FEATURE_STATES_2026-11-05.md)。

## 2026-10-28 单步 Feature Transformer 实验入口

新增 `patches/single_step_transformer_20261028/`，沿用原 B1-E：上一 Target Position 状态加到上一 Feature 与候选 Position 的两 token 表示，经一层宽度 256、4 头因果 Transformer，输出 1024 维状态残差，再通过固定 Feature 输出权重预测。两 token 因果遮罩提前分配；不持有跨轮 Draft KV。在线直接引用 Target 的 embedding 和 Feature 输出权重，独立持有可训练参数及提案图；接受与恢复仍由现有 W1 负责。生产默认保留原 step12000，三种子续训与在线结果见 [实验记录](SINGLE_STEP_TRANSFORMER_2026-10-28.md)。

## 2026-10-23 实验性概率树与同状态操作探针

新隔离目录patches/recovery_value_evidence_20261023使用纯Python浅树推理，从正式轻量上下文预测主/备用概率，按读长度操作费用选择零/单备用。Target标签只进入训练和观察记录。operation_probe为可选诊断回调；测量验证、非空路径提交、单步Target及EOS提交，退出恢复cache、树图、共享线性运行对象及捕获输出。正式计时不启用该回调。策略EOS质量按未命中费用近似，首次建图没有单独估值；原生产入口不变。结构合法与实际收益分开论证，见[结果](artifacts/recovery_value_evidence_20261023/RESULTS.md)。

## 2026-10-23 实验解码器的正式与观察路径

恢复价值实验的adaptive_decoder现在将正式选择输入与观察记录分开：正式路径仅构造标量、候选与图状态，空候选跳过回调；观察路径保留完整前缀、标签和事件。固定0/1动作与动态动作使用同一解码器和通用恢复运行对象，原W1/快速单备用保留参照入口。selection.py支持reference与compiled两种等价路径，compiled在创建时预读线性参数；模型改变后需重建选择器。成本解析使用实际计时状态，价值输入仍限定在验证前。未替换生产入口。实现与验证见[本轮报告](artifacts/recovery_value_execution_20261023/RESULTS.md)。

2026-10-21 新增三种提案侧专用图，隔离源码在patches/proposal_graphs_20261021。MLP捕获FeatureStateDraft前向，Medusa LowRank捕获heads.raw；输入复制到固定缓冲，输出在下一次同形状调用前消费。EAGLE捕获单token Draft递推与一维Feature读出头，长历史/T>1保留普通执行；按读长度分桶，固定staging KV从原缓存同步，连续同缓存递推可复用，回退/普通前向使同步状态失效。原decoder仍负责采样、接受、拒绝、缓存截短和Ranker。每个执行臂独立持有Target图与提案图，原权重和生产默认保留；首次建图进入外层同步墙钟。若外部同位置重写Draft KV，调用invalidate_cache(cache)。详见artifacts/proposal_graphs_20261021/PLAN.md。

2026-10-19 Feature EAGLE v4 为 `speculative/pf_engine.py` 增加可选持久 Draft KV。每轮按 Target 已消费前缀截短或追加 Draft 状态，局部重建新增 token 的坐标；默认关闭时沿用 v3 全历史重放。三种子运行表明 Draft 输入显著减少，但追加状态时间抵消节省，未形成稳定端到端收益。详见 [v4 性能诊断](FEATURE_EAGLE_V4_2026-10-19.md)。

2026-10-19 Feature EAGLE v3 沿用 v2 网络与验证，新增 `speculative/pf_distill.py`，在原训练入口提供条件样本、密集 Feature KL 与两处 Feature 递推。训练区分真实末尾 F 已由 Target 消费、或尚待消费；后一种先由 Draft 更新该 F 状态。相邻预测节点停止梯度传播，当前节点独立学习；Target 和 Ranker 参数不变。在线新增 F1/F2 与首次拒绝槽位计数，接受算法没有重写。详见 [v3 执行记录](FEATURE_EAGLE_V3_2026-10-19.md)。

2026-10-19 新增 EAGLE v2：`GPT.forward(return_hidden=True)` 可同时返回原 logits 和最终 RMSNorm 状态。`speculative/pf_draft.py` 共享 1L-256 因果模块，按实际 P/F token 和四维坐标递推；Joint 输出 P/F，Feature-only 只输出 F，但两类输入都参与状态更新。`speculative/pf_engine.py` 共用一次 Target 候选验证、实际提案概率的接受采样，以及待消费 token 的恢复逻辑。详见 [实现记录](EAGLE_PF_V2_2026-10-19.md)，本地源码副本为 `patches/eagle_pf_v2/`，运行版本仍在远端主工程。

## 1. 原版目标与数据流

本地 `GaussianGPT` 目录目前保存的是项目交接层，不是完整源码 checkout。下面的架构和实验说明来自对 Autodl 工作区与 GitHub 原版的核对；路径只有在源码同步到本地后才可直接执行。

原版 GaussianGPT 的核心任务是学习 3D Gaussian Splatting 场景的离散生成分布。一个场景先经过 VQ-VAE 编码成有空间结构的 token 序列，GPT 按自回归方式预测下一个 token，生成结果再由 VQ-VAE 解码为 Gaussian 参数并渲染成点云或图像。

```text
3D-FRONT / PhotoShape 场景
        │
        ▼
VQ-VAE / sparse 3D CNN encoder
        │  位置 token + 特征 token
        ▼
tokenized scene chunks
        │
        ▼
GPT with 3D RoPE, autoregressive next-token prediction
        │
        ├─ generate_chunks.py / complete_chunks.py / generate_scene.py
        ▼
VQ-VAE decoder → Gaussian Splatting renderer
        │
        ▼
scene point cloud / orbit render / evaluation artifact
```

VQ-VAE 训练阶段同时关注稀疏 3D 特征重建和 Gaussian re-render loss。`tokenize_dataset.py` 保持 encoder 参数不变，把场景写成 token 流，GPT 训练则使用这些 token 流做交叉熵。位置 token 决定 3D 结构和扫描顺序，特征 token 表示该位置的 Gaussian 语义/外观信息，因此 token 不是普通的一维文本词表。

## 2. 代码目录职责

| 目录 | 作用 |
| --- | --- |
| `model/vqvae/` | 稀疏 3D VQ-VAE、编码器、解码器和 Gaussian 重建相关模块 |
| `model/gpt/` | GPT 主干、3D RoPE、采样和 position-selective speculative 原型 |
| `serialization/` | 位置/特征 token 的序列化、反序列化和数据约定 |
| `scripts/` | 训练、tokenize、生成、渲染、测速和诊断入口 |
| `conf/` | Hydra 数据、模型、训练和实验配置 |
| `data/`, `data_splits/` | 本地数据目录与 train/val 列表；大文件不应提交 |
| `checkpoints/` | 可直接使用的基础 checkpoint；需以状态文档中的清单为准 |
| `profiling/` | 训练日志、checkpoint、端到端测速 JSON 和路线实验 |
| `outputs/` | Hydra 运行目录、渲染和临时输出 |
| `fig/` | 论文/报告图表或可视化资源 |
| `.codex/` | AI 交接说明、项目配置和状态刷新脚本 |

## 3. 数据与坐标约定

3D-FRONT/ASE 使用 z-up 坐标，PhotoShape 使用 y-up 坐标。数据配置通过 `conf/data/tokenized_ase_s2.yaml` 等文件指定数据集、split、token 上限和 chunk 策略。修改序列化或坐标变换时必须同步核对 decoder、renderer 和数据预处理，否则训练 loss 可能正常但渲染方向错误。

VQ-VAE token 保存的是 latent 网格整数坐标。默认配置有 3 个下采样阶段、每阶段缩小 2 倍，`grid_size=0.025` 米，因此 20³ latent chunk 对应名义边长 `20 × 8 × 0.025 = 4` 米。解码时先将 latent 坐标乘以 stride，再把连续 `coord_offset` 加到体素坐标上，最后乘以 `grid_size` 得到 Gaussian 中心；`coord_offset` 使用恒等映射，没有裁剪到单元格范围。因此 `[0,4]` 表示网格覆盖范围，不是 Gaussian 中心必须满足的硬限制。实现见远端 `model/vqvae/cnn/configurable_vqvae.py` 的 `decode_idxs`，参数见 `conf/model/vqvae_cnn.yaml`，偏移表示见 `conf/internal_representations.py`。

当前 S2 配置把 `max_tokens` 设为 16000，`min_chunk_occupancy` 为 0.3，chunk 起点约束为 `[null, null, -1]`。这些约束会直接影响序列长度、显存和 draft/teacher 的可比性。

## 4. GPT 与 draft 模型

原版 teacher 配置在 `conf/model/gpt.yaml` 一类配置中定义词表、序列化和 3D RoPE。当前 S2 draft 配置保持相同的词表、序列化和上下文长度，只缩小 Transformer 宽度：

| 配置 | embedding | layers | heads | context |
| --- | ---: | ---: | ---: | ---: |
| `gpt_draft_d4.yaml` | 256 | 4 | 4 | 16384 |
| `gpt_draft_d6.yaml` | 384 | 6 | 6 | 16384 |
| `gpt_draft_d8.yaml` | 512 | 8 | 8 | 16384 |
| `gpt_draft_d12.yaml` | 768 | 12 | 12 | 16384 |

draft 必须和 teacher 使用相同 token grammar；只缩小网络容量，不能改变位置/特征 token 的交替关系或 RoPE 位置解释，否则 speculative acceptance 的统计没有意义。

## 5. S3 与 position-selective speculative decoding

S3 的 feature distillation 在保留标准 token-level cross entropy 的同时，对 feature-token 行加入 teacher→draft 的 KL 约束，使用温度 `T²` 缩放，并记录 feature CE/KL、top-1 agreement、raw/policy overlap。on-policy 版本先保持 teacher 参数不变并采样 trajectory corpus，再训练 D8 student，避免只在 teacher-forced 前缀上对齐。

`model/gpt/position_selective_speculative.py` 当前只用 draft 提出 position；feature 始终由 target 采样。位置被拒绝时按 target/draft 的最终 position policy 做 residual correction，并回滚/重建相应 KV cache。统计中 `position_accepts/rejects` 用于实际接受率，`position_overlap` 衡量 position 分布重合；`feature_overlap` 来自 draft 的 feature logits，仅是诊断值，不是 feature 接受率，也不表示当前解码器会因 feature rejection 回滚。

关键不变量：

1. draft 和 target 的 token grammar、BOS/EOS 与位置编码必须兼容，缓存各自对应正确的输入前缀；
2. Target 验证会暂时消费候选 P，拒绝后须移除该候选并重新取得最终 P 条件下的 F 分布；
3. 完整一对提交后，Target 缓存长度为输出长度，神经 Draft 为输出长度加 BOS；Target 的尾 F 在下一轮消费；
4. 接受使用最终概率比率，拒绝使用归一化 `max(p-q,0)`，不能只按 top-1 决定；
5. speedup 应注明长度、采样参数、预热、硬件、随机轨迹和同步方式；同 seed 不保证两条路线输出相同。

## 6. 环境与复现风险

当前记录的环境包括 Python 3.10.21、PyTorch 2.8.0+cu129、MinkowskiEngine 0.5.9、gsplat 1.5.0、PyTorch3D 0.7.8、FlashAttention 2.8.2、Lightning 2.5.2 和 RTX 4090。README 提醒 MinkowskiEngine 编译建议 GCC ≤13，而当前记录的 GCC 为 14.3；重建环境或重新编译扩展前必须单独验证这一兼容性。


## 7. Draft proposer 接口（2026-10-05）

Position-selective engine 现在可以通过正式 proposer 接口取得 Position proposal：

```text
DraftProposal.propose_position(state)
DraftProposal.commit(token)
DraftProposal.rewind(state)
```

`DraftProposalState` 携带 Position history、最近 Feature、可选的 D8 `q_position_logits`、合法 Position token ids、EOS、sampling policy、随机数生成器和当前 step。engine 统一执行：

```text
proposer q_D^P
        ↓
Target p_T^P
        ↓
ratio acceptance / residual correction
        ↓
Target Feature commit + cache invariant
```

`NeuralDraftProposal` 只封装 D8 的 logits proposal，保留 Draft prefill/update forward；`StructuredPositionProposal` 调用结构化 table adapter，跳过 Draft Transformer forward，仅维护 Draft cache 的 BOS/位置 bookkeeping。两者都经过同一 Target acceptance/correction/rollback 路径，因此可以直接比较 proposal 成本、acceptance 和 wall-clock。Feature proposer 与 joint `[P,F]` 状态机仍不属于默认接口路径。

### 7.1 Structured lookup 实现层

`StructuredPositionGapDraft` 的正式接口保留三种 lookup mode，runner 通过 `STRUCTURED_LOOKUP_MODE` 选择：

```text
python      → tuple/dict backoff，兼容基线
tensor      → integer context code + device gap/count tensors
vectorized  → integer code + query_batch 的 batched support mapping
```

三种实现都先选择相同的 backoff table，再将有效 gap 计数归一化为原始 Position 概率。正式 proposer 将其取对数并应用 temperature/top-k/top-p，接受函数使用重新处理后的 q。`vectorized` 已提供 batch API，但当前 engine 的 `propose_position` 是单 query 调用；要取得 batch microbenchmark 的收益，必须先把 engine 的 proposal 调度改成批量。已有材料未记录 fused CUDA extension 的注册，fused lookup 仍属于后续方案。

### 7.2 直接源码核对后的实现说明（2026-10-08）

完整算法见 [D8 与结构化 Position 实现汇报](POSITION_ONLY_IMPLEMENTATION_REPORT_2026-10-08.md)。本轮通过 Computer Use 直接读取远端源码，确认查表键使用最近 gap 和最后一个 Position id 的低位，当前没有使用 F token。源码字段 `morton_low_bits` 的实际计算为对 Position id 按位与，没有额外 Morton 编码调用。

查表路线跳过 Draft Transformer 前向，当前仍加载兼容 D8 对象并分配缓存；其 F 诊断使用零 logits 占位，不能解释成查表模型的 F 预测质量。当前共用引擎只提出一个 P，离线 K=4 mode 指标没有成为四位置生成策略。

已读取的 `run_structured_position_e2e.py` 通过 profiler 对纯 Target 每次前向进行 CUDA 同步，候选引擎则关闭逐次前向计时；两条路径的输出头范围也不同。引用速度时须注明入口与计时条件，本轮没有重新测量这些差异。


## 2026-10-21 Feature W1/W2 树解码原型

隔离源码 `patches/feature_tree_20261021/` 新增 `tree_decoder.decode` 与 `TargetTreeRuntime`。树节点为 pendingF、共享P1及每分支的F/P2；注意力读取共享历史和自身祖先，RoPE与位置限制按逻辑路径计算。验证暂存KV，提交一条祖先路径后才推进逻辑位置；第二分支先暂存再压入连续槽。Feature候选无放回抽样，每次拒绝都更新完整剩余分布。沿用既有Ranker和12k MLP，不新增训练。批量RoPE版与逐节点版在实测中保持输出和业务计数相同。该原型未进入生产默认，机制与数值范围见[实现说明](artifacts/feature_tree_20261021/IMPLEMENTATION.md)。

## 2026-11-06 接受率研究候选（未替换默认）

新增候选 fusion / fusion_long 读取 Target 第 5、11、23 层（零起始）的已有表示，通过图内 3,072 维输出提供每次重放的新状态。低中层经 RMSNorm 和 2,048→256 映射加入旧 W1 分支，保留 Position/Feature token 条件、Target Feature 头、Ranker、验证和残差抽样。fusion_long 仅在训练中混入自然状态监督，推理结构与 fusion 相同。

候选实现见 `patches/pfspec_feature_states_20261105/state_draft.py`；共用原 `source_run04` 后端。本阶段结果支持 fusion_long 接受率提升，原 W1 和 Position-only 的入口及权重保持。细节与数据路径见 `FEATURE_DRAFT_LITERATURE_STUDY.md`。


2026-11-07，两 Feature 无损采样依据：在每个实际已提交历史中，Position 点提案按 Target 概率 c 接受，拒绝时按完整剩余 Position 分布采样。Feature 的确定性集合 S 与剩余分布 q′ 的一次随机提案都在 Target 判定前生成；标准 min(1,p/q′) 接受和完整 (p−q′)+ 恢复保证该 Feature 输出分布为 p。可复用分支的输出质量为 B(f)=p(f)（f∈S），其他 f 为 min(p(f),q′(f))。已选首 Feature 分支后的 Position 和第二 Feature 使用该分支完整的 Target 条件；第二候选集合独立于首 Feature 的选择，不能把其他分支的 Target 条件用于当前输出。逐位置条件保持 p，因此完整输出序列保持原 Target 分布；EOS 和预算在每个实际提交位置处理。

两层树一次调用验证 64×4 个 Feature 组合和 642 个 Target 节点，首、次 Feature、各自对应的 Position 和后续 Target 额外 token 保持真实路径。数学例子见 `validate_long_coupling.py`，真实分支及 KV 例子见 `validate_long_branches.py`，自然动作与 token 对照见 `paper06_dflash/development_validation_05/validation.json`。普通执行的 6152 项真实对照覆盖两个前缀，不构成所有可达历史的浮点逐位相同证明。原始概率差、有效输出范围、隐藏状态和 KV 误差均保存；图执行出现有效范围变化的结果保留为失败。


2026-11-07 接受率研究：P/F/P 窗口的完整条件重要性目标已完成，开发在线无稳定增量。新增 UniVer 使用每个父节点的独立剩余抽样，按前缀接受概率分配完整 Target 质量，再按子节点先于父节点的顺序选择路径。浅层 P/F/P 与已有 Block 的预期量等价；新深层 16×4 结构包含 162 个 Target 节点。独立小词表完整概率枚举、1544 项真实 Target 分支与 KV 对照，以及两个已有数值案例通过。原始 Target 参数、0.9/0.9 采样和 P/F 条件保持，未来 Target 条件只在计入费用的真实验证树中计算。完整解码与自然评价尚在执行。


## 2026-11-07 接受能力研究已完成

七个所选新机制与规定调整结束，正式三方法共 270 自然、360 capped，630 份 token 全部验证，270 自然全部 EOS；正式共同历史 11448 状态、34344 行。最终 DFlash 第 2800 步、64 首 Feature 候选、Draft 0.9/1、单层 130 节点，集合接受 BOS 71.76%、Completion 66.75%，联合接受 67.38%/61.46%，每轮推进 3.3470/3.2276，相对 fusion_long 两任务增量区间均为正。单 Feature overlap 未共同提升，主要收益包含候选覆盖；实际速度较低，保留后续低时延依据。正式数据未训练或选权重。

当前主记录为 `docs/project/FEATURE_DRAFT_LITERATURE_STUDY.md`，源码 `patches/pfspec_lossless_acceptance_20261106/`，权重、配置、原始输出、科学图及交接说明 `docs/project/artifacts/lossless_acceptance_20261106/`。服务器正式进程 139559 返回值 0、04:25:12 UTC 结束。原 Target、默认 W1 与论文保持已有状态。
