# DFlash 原文依据

| 编号 | 论文标题 | 作者 | 年月 | BibTeX | 原文支撑 | 出处 | 链接 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 5 | DFlash: Block Diffusion for Flash Speculative Decoding | Jian Chen, Yesheng Liang, Zhijian Liu | 2026-05 | `@inproceedings{chen2026dflash, title={DFlash: Block Diffusion for Flash Speculative Decoding}, author={Chen, Jian and Liang, Yesheng and Liu, Zhijian}, year={2026}, booktitle={ICML}, eprint={2602.06036}}` | 每层 Draft 注意力直接读取 Target 上下文特征；单次前向预测多个掩码位置，使用实际 Draft 概率进行无损接受采样。 | §3–4；§5.3–5.5；Appendix A.1/A.3/A.5；作者实现 `DFlashDraftModel` 和 `_rejection_sample` | [Official PDF](https://arxiv.org/pdf/2602.06036v2) / [Page](https://arxiv.org/abs/2602.06036) / [Local PDF](/Users/lief/.codex/research/pfspec_lossless_20261106/05_DFlash.pdf) | 13 页 v2；方法、相关消融及作者原始 DFlash 推理实现已读取 |

作者仓库目前同时包含 DFlash 和 DFlash 2。此次依据原始 DFlash 的每层上下文读取与掩码块预测，未采用 DFlash 2 的动态卷积和候选选择器。当前仓库没有训练入口，训练规则依据原文 Appendix A.1；现有环境中自行适配，不安装仓库列出的新依赖版本。

PFSpec 的首个实现使用三个现有 Target 层的特征、最近 64 个已提交 token、五层 Draft 注意力和两个 Feature 掩码。起始权重来自 fusion_long，新增注意力输出作为其隐状态增量。已知的 Ranker P1/P2 作为块输入；第二个 Feature 的训练标签来自原始提案分支的完整 Target 条件分布。首轮在线评估使用第一个 Feature 的实际分布及原有 Greedy 64 接受算法；第二个预测只测量，尚未扩大生成轮次。这些属于工程适配，不能套用作者原表中的加速倍数。

已保存[作者代码](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py)的当前版本，阅读了原始 DFlash 的上下文读取、块预测和非零温度接受采样。

本篇三项实际推理设置已完成自然请求：原 Draft Top-p=0.9、同权重 Top-p=1、同权重 64×4 两 Feature 组合，各八次 EOS。每次最多四/六个输出 token，完整条件及有效输出范围由真实 Target 对照记录；第二层在相同新 KV 上的数值定位继续，不能把历史差异直接归因于网络架构。
