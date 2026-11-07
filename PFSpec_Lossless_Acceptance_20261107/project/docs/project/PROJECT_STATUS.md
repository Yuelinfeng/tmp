# GaussianGPT 当前状态快照

> 由 `.codex/scripts/update_status.py` 自动生成。更新时间：`2026-11-07T12:56:30+08:00`
> 
> 稳定的架构与实验解释请看 `ARCHITECTURE.md`、`EXPERIMENTS.md` 和 `OPERATIONS.md`；本文件只保存易变事实。

## Git

- HEAD: `95040911af92a68cc3943db63f9ddc956f3ac26b`
- 当前分支: `main`
- upstream: `origin/main`
- 工作区改动数: `758`（tracked 修改/新增 `14`，未跟踪 `744`）

### 工作区摘要

- `M .codex/AI_START_HERE.md`
- ` M docs/project/ARCHITECTURE.md`
- ` M docs/project/EXPERIMENTS.md`
- ` M docs/project/LOGIC_VALIDATION.md`
- ` M docs/project/OPERATIONS.md`
- ` M docs/project/PROJECT_STATUS.md`
- ` M docs/project/RESEARCH_SOLUTIONS.md`
- ` M docs/project/artifacts/paper_design_20261021/THEORY_NOTES.md`
- ` M docs/project/artifacts/paper_v4_20261023/PAPER_MAIN.md`
- ` M docs/project/artifacts/paper_v4_20261023/overleaf/README.md`
- ` M docs/project/artifacts/paper_v4_20261023/overleaf/sections/abstract.tex`
- ` M docs/project/artifacts/paper_v4_20261023/overleaf/sections/conclusion.tex`
- ` M docs/project/artifacts/paper_v4_20261023/overleaf/sections/experiments.tex`
- ` M docs/project/artifacts/paper_v4_20261023/overleaf/sections/introduction.tex`
- `?? .codex/config.toml`
- `?? docs/project/DRAFT_A_ROLLOUT_V2_2026-10-13.md`
- `?? docs/project/FEATURE_DRAFT_LITERATURE_STUDY.md`
- `?? docs/project/FEATURE_RESIDUAL_BLOCKS_2026-10-28.md`
- `?? docs/project/GAUSSIANGPT_5090_2026-10-28.md`
- `?? docs/project/GAUSSIANGPT_AUTODL_2026-10-28.md`
- `?? docs/project/GAUSSIANGPT_DIRECTORY_REVIEW_2026-10-15.html`
- `?? docs/project/MULTICHUNK_EXACTNESS_EXPERIMENT_PLAN_2026-10-28.md`
- `?? docs/project/MULTICHUNK_LOSSLESS_VISUAL_2026-10-28.md`
- `?? docs/project/MULTICHUNK_POSITION_SURVEY_2026-10-28.md`
- `?? docs/project/PAPER_EXPERIMENT_PLAN_2026-11-04.md`
- `?? docs/project/PFSPEC_4090D_REPRODUCTION_PLAN_2026-10-28.md`
- `?? docs/project/PFSPEC_ADAPTIVE_2026-10-29.md`
- `?? docs/project/PFSPEC_FEATURE_ALLOCATION_2026-10-29.md`
- `?? docs/project/PFSPEC_FEATURE_STATES_2026-11-05.md`
- `?? docs/project/PFSPEC_FORMAL_4090D_2026-10-28.md`

## 环境记录

```text
未找到 `gaussiangpt_versions.txt`。
```

## 源码工作区

当前目录是文档交接工作区，未检测到 model/scripts/conf/serialization 源码目录。

## 权重文件索引

| 路径 | 大小 |
| --- | ---: |
| （未找到） | — |

## 端到端 profiling 索引

吞吐单位为 tokens/s；speedup 为提议路线 / Target。

| JSON | Target tok/s | 提议路线 tok/s | speedup |
| --- | ---: | ---: | ---: |
| （未找到 `profiling/e2e*.json`） | — | — | — |

## 交接提示

- 先读 `.codex/AI_START_HERE.md`，再结合 `ARCHITECTURE.md`、`EXPERIMENTS.md` 和 `OPERATIONS.md` 判断任务类型。
- 运行或引用实验前，确认 teacher、VQ-VAE、draft checkpoint、seed、token 数、temperature、top-p、warmup 和 GPU。
- 当前状态脚本不会判断 checkpoint 是否语义可用；`.corrupt`、未完成或没有配置对应关系的文件仍需人工排除。
- 代码、配置、checkpoint 或环境变更后重新运行本脚本。



## 2026-11-07 接受能力研究已完成

七个所选新机制与规定调整结束，正式三方法共 270 自然、360 capped，630 份 token 全部验证，270 自然全部 EOS；正式共同历史 11448 状态、34344 行。最终 DFlash 第 2800 步、64 首 Feature 候选、Draft 0.9/1、单层 130 节点，集合接受 BOS 71.76%、Completion 66.75%，联合接受 67.38%/61.46%，每轮推进 3.3470/3.2276，相对 fusion_long 两任务增量区间均为正。单 Feature overlap 未共同提升，主要收益包含候选覆盖；实际速度较低，保留后续低时延依据。正式数据未训练或选权重。

当前主记录为 `docs/project/FEATURE_DRAFT_LITERATURE_STUDY.md`，源码 `patches/pfspec_lossless_acceptance_20261106/`，权重、配置、原始输出、科学图及交接说明 `docs/project/artifacts/lossless_acceptance_20261106/`。服务器正式进程 139559 返回值 0、04:25:12 UTC 结束。原 Target、默认 W1 与论文保持已有状态。
