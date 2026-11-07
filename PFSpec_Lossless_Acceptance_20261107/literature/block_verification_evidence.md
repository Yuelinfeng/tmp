# 块验证原文依据

| 编号 | 论文标题 | 作者 | 年月 | BibTeX | 原文支撑 | 出处 | 链接 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 4 | Block Verification Accelerates Speculative Decoding | Ziteng Sun 等 | 2025-04 | `@inproceedings{sun2025blockverification, title={Block Verification Accelerates Speculative Decoding}, year={2025}, booktitle={ICLR}, eprint={2403.10444}}` | 整段提案协调最长接受前缀，按缩放的 Target 与 Draft 差额采样额外 token，保持完整 Target 输出分布；适用给定草稿路径条件分布和标准外层采样。 | Algorithm 2，式(4)–(5)，Theorem 1–2，Appendix A/B；§6 温度、长度、模型消融 | [Official PDF](https://arxiv.org/pdf/2403.10444v3) / [Page](https://arxiv.org/abs/2403.10444) / [Local PDF](/Users/lief/.codex/research/pfspec_lossless_20261106/04_Block-Verification.pdf) | 30页作者 v3（2025-04-10）；方法、证明、作者示意代码和相关消融已读取 |

作者 Appendix A 明确示意代码不适合实际使用，当前工程根据 Algorithm 2 数学公式实现。PFSpec 只协调首个 P/F 对，随后复用已验证分支；组合的整体最优性需要另行论证，不沿用原文的路径信息最优结论。
