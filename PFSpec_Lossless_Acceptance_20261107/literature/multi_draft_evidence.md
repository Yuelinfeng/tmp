# 多候选精确采样原文依据

| 编号 | 论文标题 | 作者 | 年月 | BibTeX | 原文支撑 | 出处 | 链接 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 3 | Towards Optimal Multi-draft Speculative Decoding | Zhengmian Hu 等 | 2025-02 | `@article{hu2025optimalmultidraft, title={Towards Optimal Multi-draft Speculative Decoding}, year={2025}, eprint={2502.18779}}` | §5 用前 n−1 个最高 Draft 概率候选与一个剩余分布候选构造集合；与剩余 Draft 分布的精确单步耦合保持原 Target，达到该集合分布的最优接受率。 | §3–5，式(20)–(22)，Theorem 8 证明第18页，Appendix D 式(77)，§6 温度与候选数消融 | [Official PDF](https://arxiv.org/pdf/2502.18779v2) / [Page](https://arxiv.org/abs/2502.18779) / [Local PDF](/Users/lief/.codex/research/pfspec_lossless_20261106/03_Optimal-Multi-Draft.pdf) | 23页作者 v2（2026-11-02）；方法、核心证明与消融已读取 |

此最优性限于原文的确定性候选加一个随机候选的分布。不同候选构造的全局最优性没有据此成立；PFSpec 的集合接受增量与单 Feature 预测增量分别测量。公开源码检索尚未找到对应仓库，接入依照原公式与伪代码。
