# UniVer 原文依据

| 编号 | 论文标题 | 作者 | 年月 | BibTeX | 原文支撑 | 出处 | 链接 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 7 | UniVer: A Unified Perspective for Multi-step and Multi-draft Speculative Decoding | Yepeng Weng, Qiao Hu, Takehisa Yairi | 2026-05 | `@article{weng2026univer, title={UniVer: A Unified Perspective for Multi-step and Multi-draft Speculative Decoding}, author={Weng, Yepeng and Hu, Qiao and Yairi, Takehisa}, year={2026}, eprint={2605.04543}}` | 前缀接受概率参与各层条件质量分配，再从子节点向父节点决策；原文证明该条件框架中的无损性与接受最优性。 | §3；Eq. 3–10；Theorems 3–6；Appendix B；Tables 1–7 | [Official PDF](https://arxiv.org/pdf/2605.04543v1) / [Page](https://arxiv.org/abs/2605.04543) / [Local PDF](/Users/lief/.codex/research/pfspec_lossless_20261106/07_UniVer.pdf) | v1 共 19 页，方法、全部证明、实验与消融已读取；真实 P/F 两层实现完成八次开发生成与共同历史评价，来源区间已汇总；两项扩宽均在可达 BOS 条件违反原数值要求，主要方案不替代 DFlash 64 |

原文结论适用于指定候选生成方式及条件框架；不能扩大为任意候选结构的全局最优结论。浅层确定 P/F/P 的独立枚举与已有 Block 接受概率相同。本项实际采用第一 Feature 16 个候选、各父节点独立生成四个第二 Feature 候选的两层树。

公开仓库 susan-zhu/univer 的 ReadMe.md 明确描述复现，尚未确认其作者身份。已读取提交 098d79572000b86010fe2a7b81494873641fe130 的核心分配与决策源码；它仅作实现参考，公式与证明依据来自上述原文。本项目使用双精度质量分配，保留全部正质量。
