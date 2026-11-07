# LK Losses 原文依据

| 编号 | 论文标题 | 作者 | 年月 | BibTeX | 原文支撑 | 出处 | 链接 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 2 | LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding | Alexander Samarin 等 | 2026-07 | `@inproceedings{samarin2026lk, title={LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding}, booktitle={ICML}, year={2026}}` | 负 log 接受目标按 1/α 改变状态梯度；自适应 KL/TV 使用不接收梯度的 λ=exp(−ηα)，单头默认 η=3；接受比值依赖实际 p/q。 | §4.1–4.4，Table 1，Appendix A/B/D；作者 v2 PDF 14 页 | [Official PDF](https://arxiv.org/pdf/2602.23881) / [Page](https://proceedings.mlr.press/v306/samarin26a.html) / [Local PDF](/Users/lief/.codex/research/pfspec_lossless_20261106/02_LK-Losses.pdf) | 作者 v2（2026-06-01）正文与公式已读取；出版 PDF 下载未完成 |

PFSpec 实现保留 Target 0.9/0.9，当前 LK 候选的 Draft 使用 0.9/1.0。原出版论文的 LLM 数字与本项目开发接受率分别记录。公开仓库查询尚未找到官方损失训练源码，自适应目标依据完整作者公式实现。
