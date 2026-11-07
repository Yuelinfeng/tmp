# Draft-OPD 原文依据

| 编号 | 论文标题 | 作者 | 年月 | BibTeX | 原文支撑 | 出处 | 链接 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 1 | Draft-OPD: On-Policy Distillation for Speculative Draft Models | Haodi Lei 等 | 2026-05 | `@article{lei2026draftopd, title={Draft-OPD}, year={2026}, eprint={2605.29343}}` | 接受部分使用 forward KL，拒绝部分使用 reverse KL，后续拒绝位置指数降权；重放输入为相同草稿前缀。 | Method，公式(7)–(13)，PDF 第5–6页；Appendix B | [Official PDF](https://arxiv.org/pdf/2605.29343) / [Page](https://aclanthology.org/2026.emnlp-main.1139/) / [Local PDF](/Users/lief/.codex/research/pfspec_lossless_20261106/01_Draft-OPD.pdf) | 已读取公式与消融；月份对应预印本，出版身份为 EMNLP 2026 |

当前单 Feature 只适配真实轮起点反馈，原长草稿块的拒绝后缀输入未对应到此状态结构。官方版本 d6f0ec859321959785c65b59eb311fd25f042158 的扩展训练代码与论文目标分别记录；实际收益由本机统一研究记录中的开发数据支持。
